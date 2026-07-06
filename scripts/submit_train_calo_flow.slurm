#!/bin/bash
#SBATCH --job-name=genpu-caloflow
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/scratch/gpfs/IOJALVO/lv7805/genpu_data/logs/caloflow_%j.out
#SBATCH --error=/scratch/gpfs/IOJALVO/lv7805/genpu_data/logs/caloflow_%j.err

# M2 calo-flow spike training.
#   sbatch scripts/submit_train_calo_flow.slurm
#   sbatch scripts/submit_train_calo_flow.slurm --export=ALL,EXTRA_ARGS="--steps 60000"

export MAMBA_ROOT_PREFIX='/scratch/gpfs/IOJALVO/gnn-tracking/object_condensation/micromamba'
export MAMBA_EXE='/home/lv7805/bin/micromamba'
export HF_HOME='/scratch/gpfs/IOJALVO/genpu_cache'

cd /home/lv7805/genpu
echo "Job ID: $SLURM_JOB_ID  Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"
echo "Start: $(date)"; echo ""

# genpu2 has torch 2.5.1 (the genpu env does not)
$MAMBA_EXE run -n genpu2 python scripts/train_calo_flow.py ${EXTRA_ARGS:-}

echo ""; echo "End: $(date)"
