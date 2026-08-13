#!/bin/bash
#SBATCH --job-name=pion_calo_slice
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
export MAMBA_ROOT_PREFIX=/scratch/gpfs/IOJALVO/gnn-tracking/object_condensation/micromamba
export MAMBA_EXE=/home/lv7805/bin/micromamba
export HF_HOME=/scratch/gpfs/IOJALVO/lv7805/genpu_cache
cd /home/lv7805/genpu
$MAMBA_EXE run -n genpu2 python scripts/build_calo_slice.py \
    --pdg_class 3 4 --energy_mode abs --shards 0 1 2 \
    --out /scratch/gpfs/IOJALVO/lv7805/genpu_data/calo_slice/pion_ctr_v2.npz
echo "=== done ==="
