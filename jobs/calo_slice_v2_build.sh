#!/bin/bash
#SBATCH --job-name=calo_slice_v2
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
export MAMBA_ROOT_PREFIX=/scratch/gpfs/IOJALVO/gnn-tracking/object_condensation/micromamba
export MAMBA_EXE=/home/lv7805/bin/micromamba
cd /home/lv7805/genpu
OUT=/scratch/gpfs/IOJALVO/lv7805/genpu_data/calo_slice
# Production v2 slices: e± train (shards 0,1,2) + held-out (shard 5). Two files only -- no control
# arm (its diagnostics are logged) and no pion until we actually retrain pion.
echo "########## e± TRAIN (shards 0 1 2) ##########"
$MAMBA_EXE run -n genpu2 python scripts/build_calo_slice_v2.py \
    --shards 0 1 2 --pdg_class 0 1 --out $OUT/electron_v2.npz
echo "########## e± HELD-OUT (shard 5) ##########"
$MAMBA_EXE run -n genpu2 python scripts/build_calo_slice_v2.py \
    --shards 5 --pdg_class 0 1 --out $OUT/electron_v2_h5.npz
echo "=== done ==="
