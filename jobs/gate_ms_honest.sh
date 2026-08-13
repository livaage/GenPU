#!/bin/bash
#SBATCH --job-name=gate_ms_honest
#SBATCH --time=00:40:00
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --qos=gpu-test
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
export MAMBA_ROOT_PREFIX=/scratch/gpfs/IOJALVO/gnn-tracking/object_condensation/micromamba
export MAMBA_EXE=/home/lv7805/bin/micromamba
export HF_HOME=/scratch/gpfs/IOJALVO/lv7805/genpu_cache
cd /home/lv7805/genpu
CK=/scratch/gpfs/IOJALVO/lv7805/genpu_data/checkpoints/tracker
COUNT=$CK/count_head_d0_selfnorm.pt

# HONEST full-event gate over the same charged species as the 0.77 deliverable.
echo "############ v3-ms HONEST (all charged species, count-driven) ############"
$MAMBA_EXE run -n genpu2 python scripts/tracker_honest_gate_module.py \
    --ckpt $CK/surface_ms_v3_ss/checkpoint_050000.pt --count_ckpt $COUNT \
    --slice /scratch/gpfs/IOJALVO/lv7805/genpu_data/tracker_slice/surface_multispecies.npz \
    --pdg_class 0 1 3 4 7 8 11 12 --shard 0
echo "=== done ==="
