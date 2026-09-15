#!/bin/bash
#SBATCH --job-name=calo_v2_smoke2
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
export MAMBA_ROOT_PREFIX=/scratch/gpfs/IOJALVO/gnn-tracking/object_condensation/micromamba
export MAMBA_EXE=/home/lv7805/bin/micromamba
cd /home/lv7805/genpu
OUT=/scratch/gpfs/IOJALVO/lv7805/genpu_data/calo_slice
# TIMING re-check after the hoist fix (job 13035236 stalled: the det gather was inside the
# per-shower loop). Uncapped single shard, ALL classes -- the same loop size that stalled, so if
# this finishes quickly the full rebuild is safe. Scratch name; touches no production slice.
$MAMBA_EXE run -n genpu2 python scripts/build_calo_slice_v2.py \
    --shards 5 --pdg_class 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 \
    --out $OUT/_smoke2.npz
echo "=== smoke2 done ==="
