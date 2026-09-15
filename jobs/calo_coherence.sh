#!/bin/bash
#SBATCH --job-name=calo_coherence
#SBATCH --time=00:55:00
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --qos=gpu-test
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
export MAMBA_ROOT_PREFIX=/scratch/gpfs/IOJALVO/gnn-tracking/object_condensation/micromamba
export MAMBA_EXE=/home/lv7805/bin/micromamba
cd /home/lv7805/genpu
RUN="$MAMBA_EXE run -n genpu2 python"
SL=/scratch/gpfs/IOJALVO/lv7805/genpu_data/calo_slice
CK=/scratch/gpfs/IOJALVO/lv7805/genpu_data/checkpoints/calo_flow
S=scripts/calo_shower_coherence.py
E=$CK/electron_v2_s0/checkpoint_060000.pt
M=$CK/multispecies_v2_s0/checkpoint_060000.pt

# Do a shower's cells see EACH OTHER, or is each drawn independently from a profile?
# 2026-08-27 settled the ONE-POINT function (real gradient up to rho=+0.54, model 0.000).
# This settles whether fixing that is ENOUGH.
#
#   xi_res  = two-point energy correlation vs separation, with each side's own one-point
#             profile removed.
#   1-pt null = values permuted among cells sharing a (radius-rank, depth-rank) bin ACROSS
#             showers: preserves E[z|own position] exactly, destroys within-shower structure.
#             This IS a simulation of the cheap fix.
#   Synthetic calibration: coherent blob +0.365, one-point-only +0.064, null +0.001.
#   So real xi_res >> 0.06 means per-cell conditioning is NOT enough.

echo "########## 1. e+- DEDICATED ##########"
$RUN $S --real_slice $SL/electron_v2_h5.npz --ckpt $E --pdg_class 0 1 --tag ele_dedicated
echo "########## 2. e+- POOLED ##########"
$RUN $S --real_slice $SL/multispecies_v2_h5.npz --ckpt $M --pdg_class 0 1 --tag ms_ele
echo "########## 3. pi+- POOLED ##########"
$RUN $S --real_slice $SL/multispecies_v2_h5.npz --ckpt $M --pdg_class 3 4 --tag ms_pion
echo "########## 4. gamma POOLED ##########"
$RUN $S --real_slice $SL/multispecies_v2_h5.npz --ckpt $M --pdg_class 2 --tag ms_photon
echo "########## 5. REAL-ONLY 17-class survey ##########"
$RUN $S --real_slice $SL/multispecies_v2_h5.npz --max_showers 200000 --tag survey
echo "=== done ==="
