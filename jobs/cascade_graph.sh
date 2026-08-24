#!/bin/bash
#SBATCH --job-name=cascade_graph
#SBATCH --array=0-5
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=logs/%x-%A_%a.out
set -uo pipefail
export MAMBA_ROOT_PREFIX=/scratch/gpfs/IOJALVO/gnn-tracking/object_condensation/micromamba
export MAMBA_EXE=/home/lv7805/bin/micromamba
export HF_HOME=/scratch/gpfs/IOJALVO/lv7805/genpu_cache
export GENPU_DATA_DIR=/scratch/gpfs/IOJALVO/lv7805/genpu_cache/datasets
cd /home/lv7805/genpu

# J2 — complete per-shard particle graph from the RAW source (one task per shard).
#
# A stage2 sidecar was the original plan and is WRONG: visible_mask drops 24.2% of raw particles,
# and those are not leaves — 16.1% of visible secondaries have an invisible direct parent and 48.6%
# of chains to the primary cross an invisible node. A parent->child graph on stage2 is broken at
# 1 in 6 edges while looking complete. See scripts/build_cascade_graph.py for the full rationale.
#
# Smoke test (job 12873315, 300 events): row-order assert passed 216/216, depth distribution
# reproduced an independent measurement to 3 decimals, soft-QCD events (no hit tables) handled.

$MAMBA_EXE run -n genpu2 python scripts/build_cascade_graph.py --shard "$SLURM_ARRAY_TASK_ID"
