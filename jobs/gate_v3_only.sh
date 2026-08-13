#!/bin/bash
#SBATCH --job-name=gate_v3_only
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --qos=gpu-test
#SBATCH --output=logs/%x-%j.out
set -uo pipefail

export MAMBA_ROOT_PREFIX=/scratch/gpfs/IOJALVO/gnn-tracking/object_condensation/micromamba
export MAMBA_EXE=/home/lv7805/bin/micromamba
export HF_HOME=/scratch/gpfs/IOJALVO/lv7805/genpu_cache
cd /home/lv7805/genpu
CK=/scratch/gpfs/IOJALVO/lv7805/genpu_data/checkpoints/tracker

for STEP in 040000 060000; do
  echo "############ v3 SURFACE surface_pion_v3 @ ${STEP} ############"
  $MAMBA_EXE run -n genpu2 python scripts/tracker_honest_gate_module.py \
      --ckpt $CK/surface_pion_v3/checkpoint_${STEP}.pt --count_ckpt $CK/count_head_d0.pt \
      --pdg_class 3 4 --shard 0
done
echo "=== done ==="
