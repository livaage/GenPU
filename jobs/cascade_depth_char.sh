#!/bin/bash
#SBATCH --job-name=cascade_depth_char
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
export MAMBA_ROOT_PREFIX=/scratch/gpfs/IOJALVO/gnn-tracking/object_condensation/micromamba
export MAMBA_EXE=/home/lv7805/bin/micromamba
cd /home/lv7805/genpu
# J4a — the two architecture questions for the cascade generator, answered from data before any
# model is built: (1) is depth-2 the same function as depth-1 (=> one recursive model), and
# (2) does the material map need phi (the July spikes condition on [log_E,eta,vr,vz], no phi)?
$MAMBA_EXE run -n genpu2 python scripts/cascade_depth_characterize.py --shard 0
