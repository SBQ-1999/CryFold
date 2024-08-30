"""
MIT License

Copyright (c) 2022 Kiarash Jamali

This file is modified from: [https://github.com/3dem/model-angelo/blob/main/model_angelo/gnn/inference.py].

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions.
"""
import os.path
import sys
from collections import namedtuple
import torch
from scipy.spatial import cKDTree
import tqdm
from CryFold.utils.mrc_tools import load_map,make_model_grid
from CryFold.utils.flood_fill import final_results_to_cif, normalize_local_confidence_score
from CryFold.utils.fasta_utils import is_valid_fasta_ending
from CryFold.utils.protein import Protein, load_sequence_from_fasta, get_protein_empty_except, load_protein_from_prot, \
    get_lm_embeddings_for_protein, get_protein_from_file_path
from CryFold.utils.affine_utils import init_random_affine_from_translation, get_affine_translation, get_affine, get_affine_rot
from CryFold.utils.pdb_untils import load_cas_from_structure
import numpy as np
from CryFold.CryNet.CryFolder import CryFolder

def get_module_device(module: torch.nn.Module) -> torch.device:
    return next(module.parameters()).device

def init_protein_from_see_alpha(see_alpha_file: str, fasta_file: str,end_flag:bool=False) -> Protein:
    ca_locations = torch.from_numpy(load_cas_from_structure(see_alpha_file)).float()
    rigidgroups_gt_frames = np.zeros((len(ca_locations), 1, 3, 4), dtype=np.float32)
    rigidgroups_gt_frames[:, 0] = init_random_affine_from_translation(
        ca_locations
    ).numpy()
    rigidgroups_gt_exists = np.ones((len(ca_locations), 1), dtype=np.float32)
    unified_seq, unified_seq_len = load_sequence_from_fasta(fasta_file)

    return get_protein_empty_except(
        rigidgroups_gt_frames=rigidgroups_gt_frames,
        rigidgroups_gt_exists=rigidgroups_gt_exists,
        unified_seq=unified_seq,
        unified_seq_len=unified_seq_len,
    )
def argmin_random(count_tensor: torch.Tensor):
    rand_idxs = torch.randperm(len(count_tensor))
    corr_idxs = torch.arange(len(count_tensor))[rand_idxs]
    random_argmin = count_tensor[rand_idxs].argmin()
    original_argmin = corr_idxs[random_argmin]
    return original_argmin
def update_protein_gt_frames(
    protein: Protein, update_indices: np.ndarray, update_affines: np.ndarray
) -> Protein:
    protein.rigidgroups_gt_frames[update_indices][:, 0] = update_affines
    return protein
def get_inference_data(protein, grid_data, idx, crop_length=200):
    grid = ((grid_data.grid - np.mean(grid_data.grid)) / np.std(grid_data.grid)).astype(
        np.float32
    )
    backbone_frames = protein.rigidgroups_gt_frames[:, 0]  # (num_res, 3, 4)
    ca_positions = get_affine_translation(backbone_frames)
    picked_indices = np.arange(len(ca_positions), dtype=int)
    if len(ca_positions) > crop_length:
        random_res_index = idx
        kd = cKDTree(ca_positions)
        _, picked_indices = kd.query(ca_positions[random_res_index], k=crop_length)

    return {
        "affines": torch.from_numpy(backbone_frames[picked_indices]),
        "cryo_grids": torch.from_numpy(grid[None]),  # Add channel dim
        "sequence": torch.from_numpy(np.copy(protein.residue_to_lm_embedding)),
        "cryo_global_origins": torch.from_numpy(
            grid_data.global_origin.astype(np.float32)
        ),
        "cryo_voxel_sizes": torch.from_numpy(grid_data.voxel_size.astype(np.float32)),
        "indices": torch.from_numpy(picked_indices),
        "num_nodes": len(picked_indices),
    }
@torch.no_grad()
def run_inference_on_data(
    module,
    data,
    run_iters: int = 3,
    seq_attention_batch_size: int = 200,
):
    device = get_module_device(module)

    affines = data["affines"].to(device)
    result = module(
        sequence=data["sequence"][None].to(device),
        sequence_mask=torch.ones(1, data["sequence"].shape[0], device=device),
        positions=get_affine_translation(affines),
        batch=None,
        cryo_grids=[data["cryo_grids"].to(device)],
        cryo_global_origins=[data["cryo_global_origins"].to(device)],
        cryo_voxel_sizes=[data["cryo_voxel_sizes"].to(device)],
        init_affine=affines,
        run_iters=run_iters,
        seq_attention_batch_size=seq_attention_batch_size,
    )
    result.to("cpu")
    return result


def init_empty_collate_results(num_residues, unified_seq_len, device="cpu"):
    result = {}
    result["counts"] = torch.zeros(num_residues, device=device)
    result["pred_positions"] = torch.zeros(num_residues, 3, device=device)
    result["pred_affines"] = torch.zeros(num_residues, 3, 4, device=device)
    result["pred_torsions"] = torch.zeros(num_residues, 7, 2, device=device)
    result["aa_logits"] = torch.zeros(num_residues, 20, device=device)
    result["local_confidence"] = torch.zeros(num_residues, device=device)
    result["existence_mask"] = torch.zeros(num_residues, device=device)
    result["seq_attention_scores"] = torch.zeros(
        num_residues, unified_seq_len, device=device
    )
    # result["edge_logits0"] = torch.zeros(
    #     num_residues, num_residues, device=device
    # )
    # result["edge_logits1"] = torch.zeros(
    #     num_residues, num_residues, device=device
    # )
    # result["edge_counts"] = torch.zeros(
    #     num_residues, num_residues, device=device
    # )
    return result


def collate_nn_results(
    collated_results, results, indices, protein,end_flag=False,crop_length=300,repeat_num:int=3
):
    num_pred_residues = crop_length//3
    if end_flag:
        repeat_logits = (collated_results["counts"][indices[:num_pred_residues]] > -1)
    else:
        repeat_logits = (collated_results["counts"][indices[:num_pred_residues]]<repeat_num)
    collated_results["counts"][indices[:num_pred_residues][repeat_logits]] += 1
    collated_results["pred_positions"][indices[:num_pred_residues][repeat_logits]] += results[
        "pred_positions"
    ][-1][:num_pred_residues][repeat_logits]
    collated_results["pred_torsions"][indices[:num_pred_residues][repeat_logits]] += torch.nn.functional.normalize(
        results["pred_torsions"][:num_pred_residues][repeat_logits], p=2, dim=-1
    )

    curr_pos_avg = (
        collated_results["pred_positions"][indices[:num_pred_residues][repeat_logits]]
        / collated_results["counts"][indices[:num_pred_residues][repeat_logits]][..., None]
    )
    collated_results["pred_affines"][indices[:num_pred_residues][repeat_logits]] = get_affine(
        get_affine_rot(results["pred_affines"][-1][:num_pred_residues][repeat_logits]).cpu(),
        curr_pos_avg
    )
    collated_results["aa_logits"][indices[:num_pred_residues][repeat_logits]] += results[
        "cryo_aa_logits"
    ][-1][:num_pred_residues][repeat_logits]
    collated_results["local_confidence"][indices[:num_pred_residues][repeat_logits]] = results[
        "local_confidence_score"
    ][-1][:num_pred_residues][repeat_logits]
    collated_results["existence_mask"][indices[:num_pred_residues][repeat_logits]] = results[
        "pred_existence_mask"
    ][-1][:num_pred_residues][repeat_logits]
    collated_results["seq_attention_scores"][indices[:num_pred_residues][repeat_logits]] += results[
        "seq_attention_scores"
    ][:num_pred_residues][repeat_logits]
    # edge_index = results["cryo_edges"][-1]
    # collated_results["edge_logits0"][indices[edge_index[1][:num_pred_residues*20]],indices[edge_index[0][:num_pred_residues*20]]] += results[
    #     "cryo_edge_logits"
    # ][-1][:num_pred_residues,:,1].sigmoid().flatten()
    # collated_results["edge_logits1"][indices[edge_index[1][:num_pred_residues*20]],indices[edge_index[0][:num_pred_residues*20]]] += results[
    #     "cryo_edge_logits"
    # ][-1][:num_pred_residues,:,2].sigmoid().flatten()
    # collated_results["edge_counts"][indices[edge_index[1][:num_pred_residues*20]],indices[edge_index[0][:num_pred_residues*20]]] += 1

    protein = update_protein_gt_frames(
        protein,
        indices[:num_pred_residues].numpy(),
        collated_results["pred_affines"][indices[:num_pred_residues]].numpy(),
    )
    return collated_results, protein

def get_final_nn_results(collated_results):
    final_results = {}

    final_results["pred_positions"] = (
        collated_results["pred_positions"] / collated_results["counts"][..., None]
    )
    final_results["pred_torsions"] = (
        collated_results["pred_torsions"] / collated_results["counts"][..., None, None]
    )
    final_results["pred_affines"] = get_affine(
        get_affine_rot(collated_results["pred_affines"]),
        final_results["pred_positions"],
    )
    final_results["aa_logits"] = (
        collated_results["aa_logits"] / collated_results["counts"][..., None]
    )
    final_results["seq_attention_scores"] = (
        collated_results["seq_attention_scores"] / collated_results["counts"][..., None]
    )
    final_results["local_confidence"] = collated_results["local_confidence"]
    final_results["existence_mask"] = collated_results["existence_mask"]

    final_results["raw_aa_entropy"] = (
        final_results["aa_logits"].softmax(dim=-1).log().sum(dim=-1)
    )
    final_results["normalized_aa_entropy"] = final_results["raw_aa_entropy"].add(
        -final_results["raw_aa_entropy"].min()
    )
    final_results["normalized_aa_entropy"] = final_results["normalized_aa_entropy"].div(
        final_results["normalized_aa_entropy"].max()
    )

    # final_results["edge_logits"] = (collated_results["edge_logits0"]+collated_results["edge_logits1"].transpose(1,0))/(collated_results["edge_counts"]+collated_results["edge_counts"].transpose(1,0)+1e-8)

    return dict([(k, v.numpy()) for (k, v) in final_results.items()])
MRCObject = namedtuple("MRCObject", ["grid", "voxel_size", "global_origin"])

def infer(args):
    output_dir = os.path.dirname(args.output_dir)
    device = torch.device(args.device)
    state_dict = torch.load(args.model_dir,map_location=torch.device('cpu'))
    module = CryFolder(1280,256,num_layers_former=18,num_layers_ipa=7,attention_heads=8)
    module.load_state_dict(state_dict)
    module.to(device)
    module.eval()
    protein = None
    if args.struct.endswith("cif") or args.struct.endswith("pdb"):
        if not is_valid_fasta_ending(args.fasta):
            raise RuntimeError(f"File {args.fasta} is not a fasta file format.")
        protein = init_protein_from_see_alpha(args.struct, args.fasta,end_flag=args.end_flag)
        protein = get_lm_embeddings_for_protein(protein,device=device)
    if protein is None:
        raise RuntimeError(f"File {args.struct} is not a supported file format.")
    grid_data = None
    if args.map_path.endswith("map") or args.map_path.endswith("mrc"):
        grid, voxel_size, global_origin = load_map(args.map_path, multiply_global_origin=True)
        grid, voxel_size, global_origin = make_model_grid(
            grid,
            voxel_size,
            global_origin,
            target_voxel_size=1.1,
        )
        grid_data = MRCObject(grid,voxel_size,global_origin)
    if grid_data is None:
        raise RuntimeError(
            f"Grid volume file {args.map_path} is not a cryo_em density map file format."
        )
    num_res = len(protein.rigidgroups_gt_frames)

    collated_results = init_empty_collate_results(
        num_res,
        protein.unified_seq_len,
        device="cpu",
    )

    residues_left = num_res
    total_steps = num_res * args.repeat_per_residue
    steps_left_last = total_steps

    pbar = tqdm.tqdm(total=total_steps, file=sys.stdout, position=0, leave=True)
    while residues_left > 0:
        idx = argmin_random(collated_results["counts"])
        data = get_inference_data(protein, grid_data, idx, crop_length=args.crop_length)

        results = run_inference_on_data(
            module, data, seq_attention_batch_size=args.seq_attention_batch_size
        )
        collated_results, protein = collate_nn_results(
            collated_results,
            results,
            data["indices"],
            protein,
            end_flag=True if args.aggressive_repeat else args.end_flag,
            crop_length=args.crop_length,
            repeat_num=args.repeat_per_residue
        )
        residues_left = (
            num_res
            - torch.sum(collated_results["counts"] > args.repeat_per_residue - 1).item()
        )
        steps_left = (
            total_steps
            - torch.sum(
                collated_results["counts"].clip(0, args.repeat_per_residue)
            ).item()
        )

        pbar.update(n=int(steps_left_last - steps_left))
        steps_left_last = steps_left

    pbar.close()

    final_results = get_final_nn_results(collated_results)
    output_path = os.path.join(args.output_dir, "model_net.cif")
    final_results_to_cif(
        final_results,
        output_path,
        protein.unified_seq.split("|||"),
        verbose=True,
        aggressive_pruning=args.aggressive_pruning,
        mask_threshold=args.mask_threshold,
        end_flag=args.end_flag,
    )

    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--map-path", "--m", required=True, help="The path to the input map")
    parser.add_argument(
        "--fasta", "--f", required=True, help="The path to the sequence file"
    )
    parser.add_argument(
        "--struct", "--s", required=True, help="The path to the structure file"
    )
    parser.add_argument("--model-dir", required=True, help="Where the model at")
    parser.add_argument("--output-dir", default=".", help="Where to save the results")
    parser.add_argument("--device", default="cpu", help="Which device to run on")
    parser.add_argument(
        "--crop-length", type=int, default=400, help="How many points per batch"
    )
    parser.add_argument(
        "--repeat-per-residue",
        default=3,
        type=int,
        help="How many times to repeat per residue",
    )
    parser.add_argument(
        "--aggressive-pruning",
        action="store_true",
        help="Only build parts of the model that have a good match with the sequence. "
        + "Will lower recall, but quality of build is higher",
    )
    parser.add_argument(
        "--seq-attention-batch-size",
        type=int,
        default=300,
        help="Lower memory usage by processing the sequence in batches.",
    )
    args = parser.parse_args()
    infer(args)
