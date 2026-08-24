#!/bin/bash
#SBATCH --job-name=calo_v2_eval
#SBATCH --time=00:40:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --qos=gpu-test
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
export MAMBA_ROOT_PREFIX=/scratch/gpfs/IOJALVO/gnn-tracking/object_condensation/micromamba
export MAMBA_EXE=/home/lv7805/bin/micromamba
cd /home/lv7805/genpu
SL=/scratch/gpfs/IOJALVO/lv7805/genpu_data/calo_slice
CK=/scratch/gpfs/IOJALVO/lv7805/genpu_data/checkpoints/calo_flow
# Eval only -- both 60k checkpoints already exist from job 12899065.
for s in 0 1; do
  echo "########## EVAL seed $s ##########"
  $MAMBA_EXE run -n genpu2 python scripts/calo_metrics.py \
      --ckpt $CK/electron_v2_s$s/checkpoint_060000.pt --pdg_class 0 1 \
      --real_slice $SL/electron_v2_h5.npz --tag electron_v2_s$s
done
