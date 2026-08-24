#!/bin/bash
#SBATCH --job-name=calo_v2_smoke
#SBATCH --time=00:45:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --qos=gpu-test
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
export MAMBA_ROOT_PREFIX=/scratch/gpfs/IOJALVO/gnn-tracking/object_condensation/micromamba
export MAMBA_EXE=/home/lv7805/bin/micromamba
cd /home/lv7805/genpu
# Plumbing smoke test only -- 3000 steps. Verifies pos_dim=3 is inferred from the (P,4) slice, that
# the point flow trains on 3D positions, and that the energy head reads the LAST column (not index
# 2, which under v2 would silently be DEPTH).
$MAMBA_EXE run -n genpu2 python scripts/train_calo_flow.py \
    --slice /scratch/gpfs/IOJALVO/lv7805/genpu_data/calo_slice/electron_v2.npz \
    --out /scratch/gpfs/IOJALVO/lv7805/genpu_data/checkpoints/calo_flow/electron_v2_smoke \
    --steps 3000 --no_wandb
