#!/bin/bash
#SBATCH --job-name=train_surface_v3
#SBATCH --time=00:55:00
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --qos=gpu-test
#SBATCH --output=logs/%x-%j.out
# NO --partition line (errors on this grid).
set -euo pipefail

export MAMBA_ROOT_PREFIX=/scratch/gpfs/IOJALVO/gnn-tracking/object_condensation/micromamba
export MAMBA_EXE=/home/lv7805/bin/micromamba
export HF_HOME=/scratch/gpfs/IOJALVO/lv7805/genpu_cache

cd /home/lv7805/genpu
STEPS=${STEPS:-60000}

$MAMBA_EXE run -n genpu2 python scripts/train_tracker_module.py \
    --slice   /scratch/gpfs/IOJALVO/lv7805/genpu_data/tracker_slice/surface_pion.npz \
    --module_geometry /scratch/gpfs/IOJALVO/lv7805/genpu_data/module_geometry.npz \
    --out     /scratch/gpfs/IOJALVO/lv7805/genpu_data/checkpoints/tracker/surface_pion_v3 \
    --steps $STEPS --batch 512 --lr 3e-4 \
    --save_every 10000 --log_every 500 --no_wandb \
    --run_name tracker_surface_v3

echo "=== training done ==="
