#!/bin/bash
#SBATCH --job-name=calo_slice_v2_smoke
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
export MAMBA_ROOT_PREFIX=/scratch/gpfs/IOJALVO/gnn-tracking/object_condensation/micromamba
export MAMBA_EXE=/home/lv7805/bin/micromamba
cd /home/lv7805/genpu
OUT=/scratch/gpfs/IOJALVO/lv7805/genpu_data/calo_slice
# Smoke: ONE shard, e± only. Checks the ancestor walk, the dedup, the depth/layer derivation and
# the anchor recomputed for the ancestor. Predictions to verify against (measured 2026-08-24):
#   showers/depositors ~0.51 for e±   |   cells/shower ~17.9 (NOT 20.7 -- that was the naive sum)
#   layer defined ~0.83 (endcap fraction)  |  turning-branch fraction should FALL sharply vs 0.73
$MAMBA_EXE run -n genpu2 python scripts/build_calo_slice_v2.py \
    --shards 0 --pdg_class 0 1 --out $OUT/electron_v2_smoke.npz
echo "=== v1-style control (no re-attribution), same shard ==="
$MAMBA_EXE run -n genpu2 python scripts/build_calo_slice_v2.py \
    --shards 0 --pdg_class 0 1 --no_reattribute --out $OUT/electron_v2_noreattr_smoke.npz
