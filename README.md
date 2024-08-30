# CryFold

## Overview
CryFold is a software that automatically constructs full-atom 3D structural models of proteins based on cryo-EM density maps and sequence information.

It has two main stages: the first step predicts the C<span>&alpha;</span> atom coordinates from the density map, and the second step builds the full-atom model by combining the sequence and density map information. Finally, the full-atom model will undergo a post-processing program to generate the final protein model. This post-processing program comes from [ModelAngelo](https://github.com/3dem/model-angelo).

<div align=center><img src="example/figure/framework.png" width="100%" height="100%"/></div>

For more details on CryFold, please refer to the manuscript.

## Hardware requirements
CryFold requires at least 3GB of disk space for its own weight files plus the weight files of the ESM language model. 
It also requires at least 13GB of GPU memory.

## Installation

<details>
<summary>Install CryFold</summary>
<br>

**Step 1: Install Conda**

It requires to use conda to manage the Python dependencies, which can be installed following https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html#regular-installation.

**Step 2: Clone this repository**

Now, you need to clone this Github repository with
```
git clone https://github.com/SBQ-1999/CryFold.git
```

**Step 3: Check if the GPU is available**

Ensure that the device you are currently running on has a GPU .it must be able to run the command:
```
nvcc -V
```
If you are in a cluster system, make sure that the node you are on has a GPU (this can be checked using the command 'nvcc -V'). If it does not have one, please first log into any compute node that has a GPU to execute the following commands.

**Step 4: Install CryFold**

Navigate to the CryFold installation directory and run the installation script:
```
cd CryFold
source install.sh
```
Once the installation script has finished running, you will have an CryFold execution environment.
Finally, you can run the command
```
build -h
```
to check if the installation was successful.
<br>
</details>   
    
## Usage

First, use the command
```
build -h
```
to check some basic parameters of CryFold. 

Additionally, since the first run requires downloading a 2GB ESM language model weight file, the waiting time is relatively long. 
However, this issue does not occur in subsequent runs.
Below are a few simple examples to illustrate how to use CryFold.

<details>
<summary>Use a cryo-EM density map and a FASTA sequence</summary>
<br>

First, we need the density map and the fasta file:
```
wget -P ./example https://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-33306/map/emd_33306.map.gz
wget https://www.rcsb.org/fasta/entry/7xmv -O ./example/rcsb_pdb_7XMV.fasta
cd ./example
gzip -d emd_33306.map.gz
```
Then, run CryFold:
```
conda activate CryFold
build -s rcsb_pdb_7XMV.fasta -v emd_33306.map -o out
```
</details> 

<details>
<summary>Extra use of mask map</summary>
<br>

let's assume we already have the density map and the fasta file. we also need to obtain the mask map:
```
cd ./example
wget https://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-33306/masks/emd_33306_msk_1.map
```
Then, run CryFold:
```
build -s rcsb_pdb_7XMV.fasta -v emd_33306.map -m emd_33306_msk_1.map -o use_mask
```
</details> 

<details>
<summary>Specify GPU device and inference acceleration</summary>
<br>

If you want to specify the GPU number, you can set it using the parameter -d, for example, to specify GPU 3:
```
build -s protein.fasta -v map.mrc -o output_dir -d cuda:3
```
If you want to infer 900 residues at once (the default is 300), you can set it using the parameter -n:
```
build -s protein.fasta -v map.mrc -o output_dir -n 900
```
</details> 

## Citation
<span id="citation"></span>
Su et al, Accurate de novo modeling of atomic structures from cryo-EM maps using an enhanced transformer, submitted, 2024.