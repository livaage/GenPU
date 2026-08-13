#!/bin/bash
#SBATCH --job-name=baseline_v1_pion
#SBATCH --time=00:58:00
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

# v1-pion baseline with CURRENT code (512-bin + abspos), plain AR (no vertex/helix/mom) to
# match v3's plainness. Same pion set (pion.npz = build_tracker_slice shards 0-2 pdg 3,4),
# same 60k steps -> the ONLY difference vs v3 is the layer-residual vs surface-local representation.
echo "############ TRAIN v1-pion baseline (60k, plain, current code) ############"
$MAMBA_EXE run -n genpu2 python scripts/train_tracker.py \
    --slice $SL/pion.npz --out $CK/pion_baseline_current \
    --steps 60000 --batch 512 --lr 3e-4 --save_every 20000 --log_every 500 \
    --run_name pion_baseline_current --no_wandb

echo "############ GATE v1-pion baseline @ 60k ############"
$MAMBA_EXE run -n genpu2 python scripts/tracker_honest_gate.py \
    --ckpt $CK/pion_baseline_current/checkpoint_060000.pt --count_ckpt $CK/count_head_d0.pt \
    --slice $SL/pion.npz --pdg_class 3 4 --shard 0
echo "=== done ==="
