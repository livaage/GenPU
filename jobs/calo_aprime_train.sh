#!/bin/bash
#SBATCH --job-name=calo_aprime_train
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --array=0-1
#SBATCH --output=logs/%x-%A_%a.out
set -uo pipefail
export MAMBA_ROOT_PREFIX=/scratch/gpfs/IOJALVO/gnn-tracking/object_condensation/micromamba
export MAMBA_EXE=/home/lv7805/bin/micromamba
cd /home/lv7805/genpu
RUN="$MAMBA_EXE run -n genpu2 python"
SL=/scratch/gpfs/IOJALVO/lv7805/genpu_data/calo_slice
CK=/scratch/gpfs/IOJALVO/lv7805/genpu_data/checkpoints/calo_flow

# ---------------------------------------------------------------------------
# OPTION A-PRIME: position AND energy from one joint flow whose cells see each other.
#
# Why: three independent measurements say a shower's cells behave as if drawn
# independently -- no two-point energy coherence (hadrons real 1.54-2.22, gen
# flat 1.00), cell collisions 0.036 vs a real 0.0003, and within-shower NN
# spacing 2.14x real at matched width. The generator draws each cell alone
# (PointCFM, calo_flow.py) and trains on random POINTS, so it has never seen
# two cells of one shower together.
#
# ARMS (identical recipe to multispecies_v2_s0 -- default trainer, 60k steps,
# seed 0 -- except the flags below):
#   0  cross      --joint_energy --point_attn 3
#   1  self-only  --joint_energy --point_attn 3 --attn_self_only
# The self-only arm is the same network with attention masked to the diagonal,
# so 0-vs-1 isolates "cells see each other" and 1-vs-baseline isolates the new
# architecture + joint energy. Baseline is the existing multispecies_v2_s0.
# Runs > 1h: logging every 500 steps, checkpoints every 10k.
# ---------------------------------------------------------------------------
ARMS=("cross:" "selfonly:--attn_self_only")
ARM=${ARMS[$SLURM_ARRAY_TASK_ID]}; NAME=${ARM%%:*}; FLAG=${ARM#*:}
echo "########## arm $NAME ##########"
$RUN scripts/train_calo_flow.py --slice $SL/multispecies_v2.npz \
     --out $CK/ms_aprime_${NAME}_s0 --steps 60000 --seed 0 --no_wandb \
     --log_every 500 --save_every 10000 \
     --joint_energy --point_attn 3 --attn_dim 128 --attn_heads 4 --set_budget 24576 $FLAG
echo "=== done ==="
