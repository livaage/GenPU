#!/bin/bash
#SBATCH --job-name=gate_honest_fixed
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
COUNT=$CK/count_head_d0_selfnorm.pt

# HONEST gate (count head samples n_hits, no truth) with the FIXED self-normalizing count head.
echo "############ v3-SS HONEST (count-driven, fixed count) ############"
$MAMBA_EXE run -n genpu2 python scripts/tracker_honest_gate_module.py \
    --ckpt $CK/surface_pion_v3_ss/checkpoint_025000.pt --count_ckpt $COUNT \
    --pdg_class 3 4 --shard 0

echo "############ v3-TF HONEST (count-driven, fixed count) ############"
$MAMBA_EXE run -n genpu2 python scripts/tracker_honest_gate_module.py \
    --ckpt $CK/surface_pion_v3/checkpoint_060000.pt --count_ckpt $COUNT \
    --pdg_class 3 4 --shard 0
echo "=== done ==="
