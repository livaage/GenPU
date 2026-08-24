#!/bin/bash
#SBATCH --job-name=calo_reattr_probe
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
export MAMBA_ROOT_PREFIX=/scratch/gpfs/IOJALVO/gnn-tracking/object_condensation/micromamba
export MAMBA_EXE=/home/lv7805/bin/micromamba
export HF_HOME=/scratch/gpfs/IOJALVO/lv7805/genpu_cache
cd /home/lv7805/genpu
# J3a — measure what re-attributing calo cells to the calo-incident ancestor would change,
# BEFORE spending any GPU. Kills the over-splitting hypothesis for free if the predicted shifts
# (e± shower count down, cells/shower up; photon ~unchanged as the control) do not appear.
# Also diagnoses whether born-inside depositors are getting physically meaningless helix anchors.
$MAMBA_EXE run -n genpu2 python scripts/calo_reattribution_probe.py --shard 0
