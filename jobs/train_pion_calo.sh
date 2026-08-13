#!/bin/bash
#SBATCH --job-name=train_pion_calo
#SBATCH --time=00:55:00
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --qos=gpu-test
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
export MAMBA_ROOT_PREFIX=/scratch/gpfs/IOJALVO/gnn-tracking/object_condensation/micromamba
export MAMBA_EXE=/home/lv7805/bin/micromamba
export HF_HOME=/scratch/gpfs/IOJALVO/lv7805/genpu_cache
cd /home/lv7805/genpu
$MAMBA_EXE run -n genpu2 python scripts/train_calo_flow.py \
    --slice /scratch/gpfs/IOJALVO/lv7805/genpu_data/calo_slice/pion_ctr_v2.npz \
    --out /scratch/gpfs/IOJALVO/lv7805/genpu_data/checkpoints/calo_flow/pion_qtd_v2 \
    --pos_transform quantile --count_dither --steps 40000 \
    --save_every 20000 --run_name pion_qtd_v2 --no_wandb
echo "=== training done ==="
