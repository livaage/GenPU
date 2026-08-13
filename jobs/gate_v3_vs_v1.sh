#!/bin/bash
#SBATCH --job-name=gate_v3_vs_v1
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --qos=gpu-test
#SBATCH --output=logs/%x-%j.out
set -euo pipefail

export MAMBA_ROOT_PREFIX=/scratch/gpfs/IOJALVO/gnn-tracking/object_condensation/micromamba
export MAMBA_EXE=/home/lv7805/bin/micromamba
export HF_HOME=/scratch/gpfs/IOJALVO/lv7805/genpu_cache
cd /home/lv7805/genpu
CK=/scratch/gpfs/IOJALVO/lv7805/genpu_data/checkpoints/tracker
COUNT=$CK/count_head_d0.pt
SL=/scratch/gpfs/IOJALVO/lv7805/genpu_data/tracker_slice

echo "############ v1 BASELINE: pion_v1 (40k, no vertex) ############"
$MAMBA_EXE run -n genpu2 python scripts/tracker_honest_gate.py \
    --ckpt $CK/pion_v1/checkpoint_040000.pt --count_ckpt $COUNT \
    --slice $SL/pion.npz --pdg_class 3 4 --shard 0

echo "############ v1 BASELINE: pion_v2_long (150k, best-trained) ############"
$MAMBA_EXE run -n genpu2 python scripts/tracker_honest_gate.py \
    --ckpt $CK/pion_v2_long/checkpoint_150000.pt --count_ckpt $COUNT \
    --slice $SL/pion.npz --pdg_class 3 4 --shard 0

echo "############ v3 SURFACE: surface_pion_v3 (60k) ############"
$MAMBA_EXE run -n genpu2 python scripts/tracker_honest_gate_module.py \
    --ckpt $CK/surface_pion_v3/checkpoint_060000.pt --count_ckpt $COUNT \
    --pdg_class 3 4 --shard 0

echo "=== gates done ==="
