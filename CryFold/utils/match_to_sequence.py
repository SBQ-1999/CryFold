"""
MIT License

Copyright (c) 2022 Kiarash Jamali

This file is from: [https://github.com/3dem/model-angelo/blob/main/model_angelo/utils/match_to_sequence.py].

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions.
"""
import numpy as np
from numpy import ndarray
from scipy.spatial import KDTree

from CryFold.utils.misc_utils import assertion_check


class MatchToSequence:
    def __init__(
        self,
        new_sequences,
        residue_idxs,
        sequence_idxs,
        key_start_matches,
        key_end_matches,
        match_scores,
        hmm_output_match_sequences,
        exists_in_sequence_mask,
    ):
        self.set_vals(
            new_sequences,
            residue_idxs,
            sequence_idxs,
            key_start_matches,
            key_end_matches,
            match_scores,
            hmm_output_match_sequences,
            exists_in_sequence_mask,
        )

    def sort_with_idx(self, idxs: ndarray):
        self.new_sequences = [self.new_sequences[i] for i in idxs]
        self.residue_idxs = [self.residue_idxs[i] for i in idxs]
        self.sequence_idxs = self.sequence_idxs[idxs]
        self.key_start_matches = self.key_start_matches[idxs]
        self.key_end_matches = self.key_end_matches[idxs]
        self.match_scores = self.match_scores[idxs]
        self.hmm_output_match_sequences = [
            self.hmm_output_match_sequences[i] for i in idxs
        ]
        self.exists_in_sequence_mask = [self.exists_in_sequence_mask[i] for i in idxs]

    def set_vals(
        self,
        new_sequences,
        residue_idxs,
        sequence_idxs,
        key_start_matches,
        key_end_matches,
        match_scores,
        hmm_output_match_sequences,
        exists_in_sequence_mask,
    ):
        self.new_sequences = new_sequences  # Sequence len
        self.residue_idxs = residue_idxs  # Sequence len
        self.sequence_idxs = np.array(sequence_idxs)  # Not sequence len
        self.key_start_matches = np.array(key_start_matches)  # Not sequence len
        self.key_end_matches = np.array(key_end_matches)  # Not sequence len
        self.match_scores = np.array(match_scores)  # Not sequence len
        self.hmm_output_match_sequences = hmm_output_match_sequences  # Sequence len
        self.exists_in_sequence_mask = exists_in_sequence_mask  # Sequence len

    def concatenate_chains(self, new_chain_ids):
        new_sequences = []  # Sequence len
        residue_idxs = []  # Sequence len
        sequence_idxs = []  # Not sequence len
        key_start_matches = []  # Not sequence len
        key_end_matches = []  # Not sequence len
        match_scores = []  # Not sequence len
        hmm_output_match_sequences = []  # Sequence len
        exists_in_sequence_mask = []  # Sequence len

        for merged_chain_id in new_chain_ids:
            new_sequences.append(
                np.concatenate([self.new_sequences[i] for i in merged_chain_id])
            )
            residue_idxs.append(
                np.concatenate([self.residue_idxs[i] for i in merged_chain_id])
            )
            sequence_idxs.append(self.sequence_idxs[merged_chain_id[0]])
            key_start_matches.append(self.key_start_matches[merged_chain_id[0]])
            key_end_matches.append(self.key_end_matches[merged_chain_id[-1]])
            match_scores.append(
                sum(
                    [
                        self.match_scores[i] * len(self.new_sequences[i])
                        for i in merged_chain_id
                    ]
                )
                / sum([len(self.new_sequences[i]) for i in merged_chain_id])
            )
            hmm_output_match_sequences.append(
                "".join([self.hmm_output_match_sequences[i] for i in merged_chain_id])
            )
            exists_in_sequence_mask.append(
                np.concatenate(
                    [self.exists_in_sequence_mask[i] for i in merged_chain_id]
                )
            )

        self.set_vals(
            new_sequences,
            residue_idxs,
            sequence_idxs,
            key_start_matches,
            key_end_matches,
            match_scores,
            hmm_output_match_sequences,
            exists_in_sequence_mask,
        )

    def prune_chains(
            self,
            chains,
            chain_prune_length=4,
            aggressive_pruning=False
    ):
        assertion_check(
            len(chains) == len(self.new_sequences),
            f"Chains: {len(chains)} not same size as sequences: {len(self.new_sequences)}",
        )
        new_sequences = []  # Sequence len
        residue_idxs = []  # Sequence len
        sequence_idxs = []  # Not sequence len
        key_start_matches = []  # Not sequence len
        key_end_matches = []  # Not sequence len
        match_scores = []  # Not sequence len
        hmm_output_match_sequences = []  # Sequence len
        exists_in_sequence_mask = []  # Sequence len
        new_chains = []

        # More convenient
        chains = [np.array(c) for c in chains]
        for chain_id in range(len(self.new_sequences)):
            if (
                aggressive_pruning and
                np.sum(self.exists_in_sequence_mask[chain_id] > 0.5)
                < chain_prune_length
            ) or (
                not aggressive_pruning and
                len(chains[chain_id]) < chain_prune_length
            ):
                continue
            if aggressive_pruning:
                new_sequences.append(
                    self.new_sequences[chain_id][
                        self.exists_in_sequence_mask[chain_id].astype(bool)
                    ]
                )
                residue_idxs.append(
                    self.residue_idxs[chain_id][
                        self.exists_in_sequence_mask[chain_id].astype(bool)
                    ]
                )
                exists_in_sequence_mask.append(
                    self.exists_in_sequence_mask[chain_id][
                        self.exists_in_sequence_mask[chain_id].astype(bool)
                    ]
                )
                new_chains.append(
                    chains[chain_id][
                        self.exists_in_sequence_mask[chain_id].astype(bool)
                    ]
                )
            else:
                new_sequences.append(self.new_sequences[chain_id])
                residue_idxs.append(self.residue_idxs[chain_id])
                exists_in_sequence_mask.append(self.exists_in_sequence_mask[chain_id])
                new_chains.append(chains[chain_id])

            sequence_idxs.append(self.sequence_idxs[chain_id])
            key_start_matches.append(np.min(residue_idxs[-1]))
            key_end_matches.append(np.max(residue_idxs[-1]))
            match_scores.append(self.match_scores[chain_id])
            hmm_output_match_sequences.append(self.hmm_output_match_sequences[chain_id])

        self.set_vals(
            new_sequences,
            residue_idxs,
            sequence_idxs,
            key_start_matches,
            key_end_matches,
            match_scores,
            hmm_output_match_sequences,
            exists_in_sequence_mask,
        )

        return new_chains

    def remove_duplicates(
        self,
        chains,
        ca_pos,
    ):
        kdtree = KDTree(ca_pos)

        new_sequences = []  # Sequence len
        residue_idxs = []  # Sequence len
        sequence_idxs = []  # Not sequence len
        key_start_matches = []  # Not sequence len
        key_end_matches = []  # Not sequence len
        match_scores = []  # Not sequence len
        hmm_output_match_sequences = []  # Sequence len
        exists_in_sequence_mask = []  # Sequence len
        new_chains = []

        idx_info = {}
        for chain_id, chain in enumerate(chains):
            for i, res in enumerate(chain):
                idx_info[res] = {
                    "used": False,
                    "seq_idx": self.sequence_idxs[chain_id],
                    "res_idx": self.residue_idxs[chain_id][i],
                }

        # More convenient
        chains = [np.array(c) for c in chains]
        for chain_id, chain in enumerate(chains):
            keep_list = []
            for idx in chain:
                keep_list.append(not idx_info[idx]["used"])

                close_ids = kdtree.query_ball_point(ca_pos[idx], r=2)
                for close_id in close_ids:
                    if (
                        close_id in idx_info
                        and idx_info[close_id]["res_idx"] == idx_info[idx]["res_idx"]
                        and idx_info[close_id]["seq_idx"] == idx_info[idx]["seq_idx"]
                    ):
                        idx_info[close_id]["used"] = True

            keep_arr = np.array(keep_list)
            if len(chain) < 2 and not keep_arr[0]:
                continue

            new_sequences.append(self.new_sequences[chain_id][keep_arr])
            residue_idxs.append(self.residue_idxs[chain_id][keep_arr])
            exists_in_sequence_mask.append(
                self.exists_in_sequence_mask[chain_id][keep_arr]
            )
            new_chains.append(chains[chain_id][keep_arr])
            sequence_idxs.append(self.sequence_idxs[chain_id])
            key_start_matches.append(np.min(residue_idxs[-1]))
            key_end_matches.append(np.max(residue_idxs[-1]))
            match_scores.append(self.match_scores[chain_id])
            hmm_output_match_sequences.append(self.hmm_output_match_sequences[chain_id])

        self.set_vals(
            new_sequences,
            residue_idxs,
            sequence_idxs,
            key_start_matches,
            key_end_matches,
            match_scores,
            hmm_output_match_sequences,
            exists_in_sequence_mask,
        )

        return new_chains
    def prune_short_chains(
            self,
            chains,
            match_original_seq_len=None,
            chain_prune_length=12
    ):
        assertion_check(
            len(chains) == len(self.new_sequences),
            f"Chains: {len(chains)} not same size as sequences: {len(self.new_sequences)}",
        )
        new_sequences = []  # Sequence len
        residue_idxs = []  # Sequence len
        sequence_idxs = []  # Not sequence len
        key_start_matches = []  # Not sequence len
        key_end_matches = []  # Not sequence len
        match_scores = []  # Not sequence len
        hmm_output_match_sequences = []  # Sequence len
        exists_in_sequence_mask = []  # Sequence len
        new_chains = []

        # More convenient
        chains = [np.array(c) for c in chains]
        for chain_id in range(len(self.new_sequences)):
            CF_seq_len = np.sum(self.exists_in_sequence_mask[chain_id] > 0.5)
            if CF_seq_len<chain_prune_length and CF_seq_len/(match_original_seq_len[chain_id])<0.26:
                continue
            new_sequences.append(self.new_sequences[chain_id])
            residue_idxs.append(self.residue_idxs[chain_id])
            exists_in_sequence_mask.append(self.exists_in_sequence_mask[chain_id])
            new_chains.append(chains[chain_id])

            sequence_idxs.append(self.sequence_idxs[chain_id])
            key_start_matches.append(np.min(residue_idxs[-1]))
            key_end_matches.append(np.max(residue_idxs[-1]))
            match_scores.append(self.match_scores[chain_id])
            hmm_output_match_sequences.append(self.hmm_output_match_sequences[chain_id])

        self.set_vals(
            new_sequences,
            residue_idxs,
            sequence_idxs,
            key_start_matches,
            key_end_matches,
            match_scores,
            hmm_output_match_sequences,
            exists_in_sequence_mask,
        )

        return new_chains