#!/bin/bash
#SBATCH --job-name=calo_depth
#SBATCH --array=0,1,2,5
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --output=logs/%x-%A_%a.out
set -uo pipefail
export MAMBA_ROOT_PREFIX=/scratch/gpfs/IOJALVO/gnn-tracking/object_condensation/micromamba
export MAMBA_EXE=/home/lv7805/bin/micromamba
export GENPU_DATA_DIR=/scratch/gpfs/IOJALVO/lv7805/genpu_cache/datasets
export HF_HOME=/scratch/gpfs/IOJALVO/lv7805/genpu_cache
cd /home/lv7805/genpu
# Calo longitudinal coordinate sidecar. Shards 0,1,2,5 (3 and 4 were never preprocessed).
# Smoke test 12881363 validated 216 events after the value-matching fix.
$MAMBA_EXE run -n genpu2 python scripts/build_calo_depth.py --shard "$SLURM_ARRAY_TASK_ID"
