#!/bin/bash
#SBATCH --job-name=ms_slice
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
export MAMBA_ROOT_PREFIX=/scratch/gpfs/IOJALVO/gnn-tracking/object_condensation/micromamba
export MAMBA_EXE=/home/lv7805/bin/micromamba
export HF_HOME=/scratch/gpfs/IOJALVO/lv7805/genpu_cache
cd /home/lv7805/genpu
DATA=/scratch/gpfs/IOJALVO/lv7805/genpu_data

# All-CHARGED-species surface slice, matching multispecies.npz (the 0.77 population):
# pdg classes e- e+ pi+ pi- p pbar mu- mu+ = [0,1,3,4,7,8,11,12], shards 0-2.
$MAMBA_EXE run -n genpu2 python scripts/build_tracker_surface_slice.py \
    --module_geometry $DATA/module_geometry.npz \
    --shards 0 1 2 --pdg_class 0 1 3 4 7 8 11 12 \
    --out $DATA/tracker_slice/surface_multispecies.npz
echo "=== done ==="
