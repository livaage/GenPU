#!/bin/bash
#SBATCH --job-name=genpu-eval-s2
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/scratch/gpfs/IOJALVO/genpu_data/logs/eval_stage2_%j.out
#SBATCH --error=/scratch/gpfs/IOJALVO/genpu_data/logs/eval_stage2_%j.err

# Evaluate Stage 2 model.
#
# Usage:
#   sbatch scripts/submit_eval_stage2.slurm
#
# Override checkpoint path:
#   sbatch --export=ALL,CKPT=/path/to/checkpoint.pt scripts/submit_eval_stage2.slurm

export MAMBA_ROOT_PREFIX='/scratch/gpfs/IOJALVO/gnn-tracking/object_condensation/micromamba'
export MAMBA_EXE='/home/lv7805/bin/micromamba'

cd /home/lv7805/genpu

# Default to latest checkpoint
CKPT=${CKPT:-$(ls -t /scratch/gpfs/IOJALVO/genpu_data/checkpoints/stage2/*/checkpoint_final.pt 2>/dev/null | head -1)}

if [ -z "$CKPT" ]; then
    echo "No checkpoint found. Set CKPT env var."
    exit 1
fi

echo "Evaluating: $CKPT"
$MAMBA_EXE run -n genpu python scripts/eval_stage2.py \
    --checkpoint "$CKPT" \
    --n_samples 10000 \
    --n_steps 50 \
    --eval_shards 9
