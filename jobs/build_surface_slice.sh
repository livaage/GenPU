#!/bin/bash
#SBATCH --job-name=surface_slice
#SBATCH --time=00:50:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/%x-%j.out
# CPU-only preprocessing (no GPU). NO --partition line (errors on this grid).
set -euo pipefail

export MAMBA_ROOT_PREFIX=/scratch/gpfs/IOJALVO/gnn-tracking/object_condensation/micromamba
export MAMBA_EXE=/home/lv7805/bin/micromamba
export HF_HOME=/scratch/gpfs/IOJALVO/lv7805/genpu_cache

cd /home/lv7805/genpu
DATA=/scratch/gpfs/IOJALVO/lv7805/genpu_data

echo "=== [1/2] build module geometry (shards 0-5) ==="
$MAMBA_EXE run -n genpu2 python scripts/build_module_geometry.py \
    --shards 0 1 2 3 4 5 \
    --out $DATA/module_geometry.npz

echo "=== [2/2] build surface pion slice (shards 0-2, pdg 3,4) ==="
$MAMBA_EXE run -n genpu2 python scripts/build_tracker_surface_slice.py \
    --module_geometry $DATA/module_geometry.npz \
    --shards 0 1 2 --pdg_class 3 4 \
    --out $DATA/tracker_slice/surface_pion.npz

echo "=== DONE ==="
