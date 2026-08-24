#!/bin/bash
#SBATCH --job-name=calo_3d_struct
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
export MAMBA_ROOT_PREFIX=/scratch/gpfs/IOJALVO/gnn-tracking/object_condensation/micromamba
export MAMBA_EXE=/home/lv7805/bin/micromamba
export GENPU_DATA_DIR=/scratch/gpfs/IOJALVO/lv7805/genpu_cache/datasets
export HF_HOME=/scratch/gpfs/IOJALVO/lv7805/genpu_cache
cd /home/lv7805/genpu
# Map the calo's 3D structure before designing the depth representation. Measure first (today's
# lesson). Question D is the one that could reinterpret past results.
$MAMBA_EXE run -n genpu2 python scripts/calo_3d_structure.py --shard 0 --n_events 250
