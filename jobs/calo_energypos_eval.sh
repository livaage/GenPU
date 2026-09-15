#!/bin/bash
#SBATCH --job-name=calo_epos_eval
#SBATCH --time=00:55:00
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
FD=/scratch/gpfs/IOJALVO/lv7805/genpu_data/gate_features
mkdir -p $FD
ALL="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16"

# Evaluate the --energy_pos arms EXACTLY as their controls were evaluated (jobs 12930682 for the
# dedicated e±, 12953682/12953717 for the multispecies), so every number is like-for-like:
#   dedicated e±  vs electron_v2_s0/s1     gate8 0.8105 / 0.8073
#                                          full 0.8253/0.8329  marg 0.6007/0.6166  cop 0.7004/0.7242
#   pooled  e±    vs multispecies_v2_s0/s1 gate8 0.9079 / 0.8852
#                                          full 0.9107/0.8918  marg 0.7266/0.7232  cop 0.7053/0.6959
#   all classes   vs multispecies_v2_s0/s1 gate8 0.9354 / 0.8891
#
# PREDICTION recorded before the run: the COPULA column moves, the MARGINAL does not.
case $SLURM_ARRAY_TASK_ID in
  0) M=$CK/ele_epos_s0/checkpoint_060000.pt; RS=$SL/electron_v2_h5.npz;     MODE=ele; S=0 ;;
  1) M=$CK/ele_epos_s1/checkpoint_060000.pt; RS=$SL/electron_v2_h5.npz;     MODE=ele; S=1 ;;
  2) M=$CK/ms_epos_s0/checkpoint_060000.pt;  RS=$SL/multispecies_v2_h5.npz; MODE=ms;  S=0 ;;
  3) M=$CK/ms_epos_s1/checkpoint_060000.pt;  RS=$SL/multispecies_v2_h5.npz; MODE=ms;  S=1 ;;
esac

if [ "$MODE" = "ele" ]; then
  echo "########## DEDICATED e+- seed $S (A/B vs electron_v2_s$S) ##########"
  $RUN scripts/calo_metrics.py --ckpt $M --pdg_class 0 1 --real_slice $RS \
       --tag ele_epos_s$S --dump_features $FD/ele_epos_s$S.npz
  $RUN scripts/calo_gate_diagnose.py --features $FD/ele_epos_s$S.npz --seeds 3 --pairs --tag ele_epos_s$S
else
  echo "########## POOLED, e+- population seed $S (A/B vs multispecies_v2_s$S on classes 0,1) ##########"
  $RUN scripts/calo_metrics.py --ckpt $M --pdg_class 0 1 --real_slice $RS \
       --tag ms_epos_ele_s$S --dump_features $FD/ms_epos_ele_s$S.npz
  $RUN scripts/calo_gate_diagnose.py --features $FD/ms_epos_ele_s$S.npz --seeds 3 --pairs --tag ms_epos_ele_s$S
  echo "########## POOLED, ALL CLASSES seed $S ##########"
  $RUN scripts/calo_metrics.py --ckpt $M --pdg_class $ALL --real_slice $RS \
       --tag ms_epos_all_s$S --dump_features $FD/ms_epos_all_s$S.npz
  $RUN scripts/calo_gate_diagnose.py --features $FD/ms_epos_all_s$S.npz --seeds 3 --pairs --tag ms_epos_all_s$S
fi
echo "=== done ==="
