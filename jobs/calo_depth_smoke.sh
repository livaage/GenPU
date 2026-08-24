#!/bin/bash
#SBATCH --job-name=calo_depth_smoke
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
export MAMBA_ROOT_PREFIX=/scratch/gpfs/IOJALVO/gnn-tracking/object_condensation/micromamba
export MAMBA_EXE=/home/lv7805/bin/micromamba
export GENPU_DATA_DIR=/scratch/gpfs/IOJALVO/lv7805/genpu_cache/datasets
export HF_HOME=/scratch/gpfs/IOJALVO/lv7805/genpu_cache
cd /home/lv7805/genpu
# Smoke test the calo depth sidecar before the 4-shard array (a smoke test caught the sentinel bug
# in the cascade-graph builder that would have burned every task).
$MAMBA_EXE run -n genpu2 python scripts/build_calo_depth.py --shard 0 --max_events 300
