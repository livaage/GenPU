#!/bin/bash
#SBATCH --job-name=cascade_depth_evr
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
export MAMBA_ROOT_PREFIX=/scratch/gpfs/IOJALVO/gnn-tracking/object_condensation/micromamba
export MAMBA_EXE=/home/lv7805/bin/micromamba
cd /home/lv7805/genpu
# Does depth still predict daughter multiplicity once (E, vr) is controlled? Decides whether one
# recursive cascade model serves all levels, or whether depth needs an explicit input.
$MAMBA_EXE run -n genpu2 python scripts/cascade_depth_evr.py --shard 0
