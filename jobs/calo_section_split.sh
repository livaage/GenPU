#!/bin/bash
#SBATCH --job-name=calo_section_split
#SBATCH --time=00:25:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
export MAMBA_ROOT_PREFIX=/scratch/gpfs/IOJALVO/gnn-tracking/object_condensation/micromamba
export MAMBA_EXE=/home/lv7805/bin/micromamba
cd /home/lv7805/genpu
RUN="$MAMBA_EXE run -n genpu2 python"
SL=/scratch/gpfs/IOJALVO/lv7805/genpu_data/calo_slice

# How much of rho(logE, depth) is shower PHYSICS vs the ECAL->HCAL SAMPLING STEP?
# No model, no GPU -- real data only.
echo "########## multispecies v2, held-out shard 5, all classes ##########"
$RUN scripts/calo_section_split.py --real_slice $SL/multispecies_v2_h5.npz --tag ms
echo "########## e+- dedicated slice (cross-check vs the 2026-08-27 numbers) ##########"
$RUN scripts/calo_section_split.py --real_slice $SL/electron_v2_h5.npz --tag ele
echo "=== done ==="
