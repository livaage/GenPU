#!/bin/bash
#SBATCH --job-name=genpu-params
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output=/scratch/gpfs/IOJALVO/genpu_data/logs/analyze_params.out
#SBATCH --error=/scratch/gpfs/IOJALVO/genpu_data/logs/analyze_params.err

# Quick CPU job to determine preprocessing parameters (500 events, ~15 min).

export MAMBA_ROOT_PREFIX='/scratch/gpfs/IOJALVO/gnn-tracking/object_condensation/micromamba'
export MAMBA_EXE='/home/lv7805/bin/micromamba'
export HF_HOME='/scratch/gpfs/IOJALVO/genpu_cache'

cd /home/lv7805/genpu
$MAMBA_EXE run -n genpu python scripts/analyze_preprocessing_params.py
