#!/bin/bash
#SBATCH --job-name=calo_metrics_pion
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --qos=gpu-test
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
export MAMBA_ROOT_PREFIX=/scratch/gpfs/IOJALVO/gnn-tracking/object_condensation/micromamba
export MAMBA_EXE=/home/lv7805/bin/micromamba
export HF_HOME=/scratch/gpfs/IOJALVO/lv7805/genpu_cache
cd /home/lv7805/genpu
CK=/scratch/gpfs/IOJALVO/lv7805/genpu_data/checkpoints/calo_flow
$MAMBA_EXE run -n genpu2 python scripts/calo_metrics.py \
    --ckpt $CK/pion_qtd_v2/checkpoint_040000.pt --pdg_class 3 4 --tag pion --shard 5
echo "=== done ==="
