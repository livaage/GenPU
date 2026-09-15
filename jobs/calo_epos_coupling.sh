#!/bin/bash
#SBATCH --job-name=calo_epos_coupling
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
S=scripts/calo_energy_position_coupling.py
E=$CK/electron_v2_s0/checkpoint_060000.pt
M=$CK/multispecies_v2_s0/checkpoint_060000.pt

# Does a cell's ENERGY depend on WHERE it sits in its shower? The v2 calo defect is majority
# JOINT (copula 0.70-0.72 vs marginals 0.60-0.62, 2026-08-25) and the copula half has never
# moved. `EnergyHead` never sees the cell's position (calo_flow.py:192), so within a shower
# E is independent of (d_eta, d_phi, depth) BY CONSTRUCTION. This measures whether real showers
# violate that, and by how much.
#
# Every arm reports three sides:
#   real  -- the held-out shard-5 slice
#   null  -- the SAME real cells with each shower's shape paired to another shower's energy
#            multiset, matched on cell count. This IS the architecture's conditional
#            independence, so real-vs-null is the coupling the current model cannot represent.
#   gen   -- the checkpoint, which must land on the null. If it does not, the reading of the
#            architecture is wrong and none of the real numbers should be trusted.

echo "########## 1. e+- , DEDICATED model (the copula-0.70 reference population) ##########"
$RUN $S --real_slice $SL/electron_v2_h5.npz --ckpt $E --pdg_class 0 1 \
        --max_showers 300000 --tag ele_dedicated

echo "########## 2. e+- , POOLED model (same population; pooling cost +0.09, all marginal) ##########"
$RUN $S --real_slice $SL/multispecies_v2_h5.npz --ckpt $M --pdg_class 0 1 \
        --max_showers 300000 --tag ms_ele

echo "########## 3. pi+- , POOLED (deepest big class: depth 470-483 mm vs e+- 301) ##########"
$RUN $S --real_slice $SL/multispecies_v2_h5.npz --ckpt $M --pdg_class 3 4 \
        --max_showers 300000 --tag ms_pion

echo "########## 4. gamma , POOLED (shallowest, 113 mm; frac_near_floor NEVER broken here) ##########"
$RUN $S --real_slice $SL/multispecies_v2_h5.npz --ckpt $M --pdg_class 2 \
        --max_showers 300000 --tag ms_photon

echo "########## 5. REAL-ONLY 17-class survey (no generation: is the gradient universal?) ##########"
$RUN $S --real_slice $SL/multispecies_v2_h5.npz --max_showers 200000 --tag survey
echo "=== done ==="
