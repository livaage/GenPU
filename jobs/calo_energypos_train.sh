#!/bin/bash
#SBATCH --job-name=calo_epos_train
#SBATCH --time=00:45:00
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --qos=gpu-test
#SBATCH --array=0-3
#SBATCH --output=logs/%x-%A_%a.out
set -uo pipefail
export MAMBA_ROOT_PREFIX=/scratch/gpfs/IOJALVO/gnn-tracking/object_condensation/micromamba
export MAMBA_EXE=/home/lv7805/bin/micromamba
cd /home/lv7805/genpu
RUN="$MAMBA_EXE run -n genpu2 python"
SL=/scratch/gpfs/IOJALVO/lv7805/genpu_data/calo_slice
CK=/scratch/gpfs/IOJALVO/lv7805/genpu_data/checkpoints/calo_flow

# --energy_pos: condition the ENERGY head on each cell's OWN position (depth-within-section +
# transverse radius) and on the 4-way ECAL/HCAL x barrel/endcap region token.
#
# Until now every input to that head was a per-SHOWER quantity gathered by [src], so the cells of a
# shower received a bit-identical input vector and cell energy was independent of cell position BY
# CONSTRUCTION. Measured within-shower rho(logE, depth), real vs generated (jobs 13032162/13033802,
# geometry step removed): p +0.519/0.000, mu± +0.38/0.000, pi± +0.20/0.000, gamma -0.110/0.000.
#
# CONTROLS ARE THE EXISTING 60k CHECKPOINTS -- not retrained. `points_flat` is byte-identical after
# the 2026-08-27 slice rebuild (verified: max|depth_global - depth_local| = 0.000000 mm over every
# ECAL cell), the recipe is matched exactly, and ONLY --energy_pos differs:
#   electron_v2_s0/s1      gate8 0.8105 / 0.8073   full 0.8253/0.8329  marg 0.6007/0.6166  cop 0.7004/0.7242
#   multispecies_v2_s0/s1  gate8 0.9079 / 0.8852 (e±), 0.9354 / 0.8891 (all classes)
#
# PREDICTION, recorded before the run: the COPULA column moves, the MARGINAL column does not. The
# 2026-08-25 pooling result proved the two halves move independently, so a fix to one can be paid
# for out of the other -- if both move, or only the marginal does, the mechanism story is wrong.
#
# 2 seeds because measured seed spread is 0.023 (pooled e±) and 0.046 (all classes); one seed
# cannot read an effect below ~0.05 on the mixture.
case $SLURM_ARRAY_TASK_ID in
  0) SLICE=electron_v2;     TAG=ele_epos_s0;  SEED=0 ;;
  1) SLICE=electron_v2;     TAG=ele_epos_s1;  SEED=1 ;;
  2) SLICE=multispecies_v2; TAG=ms_epos_s0;   SEED=0 ;;
  3) SLICE=multispecies_v2; TAG=ms_epos_s1;   SEED=1 ;;
esac
echo "########## $TAG : slice $SLICE seed $SEED, 60k steps, --energy_pos ##########"
$RUN scripts/train_calo_flow.py --slice $SL/$SLICE.npz --steps 60000 --log_every 5000 \
     --energy_pos --seed $SEED --out $CK/$TAG
echo "=== done $TAG ==="
