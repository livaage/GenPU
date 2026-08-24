#!/bin/bash
#SBATCH --job-name=casc_graph_smoke
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
export MAMBA_ROOT_PREFIX=/scratch/gpfs/IOJALVO/gnn-tracking/object_condensation/micromamba
export MAMBA_EXE=/home/lv7805/bin/micromamba
export HF_HOME=/scratch/gpfs/IOJALVO/lv7805/genpu_cache
export GENPU_DATA_DIR=/scratch/gpfs/IOJALVO/lv7805/genpu_cache/datasets
cd /home/lv7805/genpu
# J2 SMOKE TEST — 300 events of shard 0 only. Validates the row-order assert, the derived columns
# and the depth walk before committing a 6-shard array. Writes _smoke output; no join key.
$MAMBA_EXE run -n genpu2 python scripts/build_cascade_graph.py --shard 0 --max_events 300
