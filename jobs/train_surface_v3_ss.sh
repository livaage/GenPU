#!/bin/bash
#SBATCH --job-name=train_v3_ss
#SBATCH --time=00:55:00
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=gpu80
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
SL=/scratch/gpfs/IOJALVO/lv7805/genpu_data/tracker_slice

# Fine-tune the 60k teacher-forced v3 with SCHEDULED SAMPLING (the confirmed-exposure-bias fix).
echo "############ TRAIN v3 + scheduled sampling (warm-start from 60k TF) ############"
$MAMBA_EXE run -n genpu2 python scripts/train_tracker_module.py \
    --slice $SL/surface_pion.npz \
    --out $CK/surface_pion_v3_ss \
    --init_from $CK/surface_pion_v3/checkpoint_060000.pt \
    --steps 25000 --batch 512 --lr 2e-4 \
    --ss_prob 0.4 --ss_warmup 500 --ss_ramp 8000 \
    --save_every 25000 --log_every 500 --no_wandb --run_name v3_ss

echo "############ GATE (truth count) — compare to v3-TF 0.80 ############"
$MAMBA_EXE run -n genpu2 python scripts/tracker_honest_gate_module.py \
    --ckpt $CK/surface_pion_v3_ss/checkpoint_025000.pt --count_ckpt $CK/count_head_d0.pt \
    --pdg_class 3 4 --shard 0 --truth_count

echo "############ DRIFT TEST — did free-running flatten? ############"
$MAMBA_EXE run -n genpu2 python scripts/tracker_teacherforce_test.py \
    --ckpt $CK/surface_pion_v3_ss/checkpoint_025000.pt --n_tracks 20000 \
    --out /home/lv7805/genpu/plots/tracker/v3_ss_drift.png
echo "=== done ==="
