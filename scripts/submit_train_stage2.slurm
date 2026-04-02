#!/bin/bash
#SBATCH --job-name=genpu-train-s2
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/scratch/gpfs/IOJALVO/genpu_data/logs/train_stage2_%j.out
#SBATCH --error=/scratch/gpfs/IOJALVO/genpu_data/logs/train_stage2_%j.err

# Train Stage 2 model on preprocessed data.
#
# Usage:
#   sbatch scripts/submit_train_stage2.slurm
#
# To override config values:
#   sbatch scripts/submit_train_stage2.slurm --export=ALL,EXTRA_ARGS="--n_shards 5 --batch_size 512"

export MAMBA_ROOT_PREFIX='/scratch/gpfs/IOJALVO/gnn-tracking/object_condensation/micromamba'
export MAMBA_EXE='/home/lv7805/bin/micromamba'

cd /home/lv7805/genpu

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start: $(date)"
echo ""

$MAMBA_EXE run -n genpu python scripts/train_stage2.py ${EXTRA_ARGS:-}

echo ""
echo "End: $(date)"
