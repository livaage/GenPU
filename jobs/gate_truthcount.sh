#!/bin/bash
#SBATCH --job-name=gate_truthcount
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
SL=/scratch/gpfs/IOJALVO/lv7805/genpu_data/tracker_slice

# TRUTH-count ablation: feed real n_hits so the gate measures ONLY spatial hit quality,
# removing the (multispecies) count-head-undercounts-pions confound. Isolates v1 vs v3 representation.
echo "############ v1 baseline (truth count) ############"
$MAMBA_EXE run -n genpu2 python scripts/tracker_honest_gate.py \
    --ckpt $CK/pion_baseline_current/checkpoint_060000.pt --count_ckpt $CK/count_head_d0.pt \
    --slice $SL/pion.npz --pdg_class 3 4 --shard 0 --truth_count

echo "############ v3 surface (truth count) ############"
$MAMBA_EXE run -n genpu2 python scripts/tracker_honest_gate_module.py \
    --ckpt $CK/surface_pion_v3/checkpoint_060000.pt --count_ckpt $CK/count_head_d0.pt \
    --pdg_class 3 4 --shard 0 --truth_count
echo "=== done ==="
