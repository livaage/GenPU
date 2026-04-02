#!/bin/bash
#SBATCH --job-name=genpu-preprocess
#SBATCH --array=0-999
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output=/scratch/gpfs/IOJALVO/genpu_data/logs/preprocess_%a.out
#SBATCH --error=/scratch/gpfs/IOJALVO/genpu_data/logs/preprocess_%a.err

# Preprocessing: one task per Arrow shard (1000 shards total).
# Each shard has ~7k common events. Stores flat arrays with offsets (no padding).
#
# Usage:
#   mkdir -p /scratch/gpfs/IOJALVO/genpu_data/{logs,preprocessed}
#   sbatch scripts/submit_preprocess.slurm
#
# To process a subset (e.g. first 10 shards for testing):
#   sbatch --array=0-9 scripts/submit_preprocess.slurm

export MAMBA_ROOT_PREFIX='/scratch/gpfs/IOJALVO/gnn-tracking/object_condensation/micromamba'
export MAMBA_EXE='/home/lv7805/bin/micromamba'
export HF_HOME='/scratch/gpfs/IOJALVO/genpu_cache'

cd /home/lv7805/genpu

$MAMBA_EXE run -n genpu python scripts/preprocess_shard.py ${SLURM_ARRAY_TASK_ID} \
    --output-dir /scratch/gpfs/IOJALVO/genpu_data/preprocessed
