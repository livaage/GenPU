#!/bin/bash
#SBATCH --job-name=helix_pion_slice
#SBATCH --time=00:35:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
export MAMBA_ROOT_PREFIX=/scratch/gpfs/IOJALVO/gnn-tracking/object_condensation/micromamba
export MAMBA_EXE=/home/lv7805/bin/micromamba
export HF_HOME=/scratch/gpfs/IOJALVO/lv7805/genpu_cache
cd /home/lv7805/genpu
$MAMBA_EXE run -n genpu2 python scripts/build_tracker_helix_slice.py \
    --shards 0 1 2 --pdg_class 3 4 \
    --out /scratch/gpfs/IOJALVO/lv7805/genpu_data/tracker_slice/helix_pion.npz
echo "=== done ==="
