#!/bin/bash
#SBATCH --job-name=v4_coherence
#SBATCH --time=00:15:00
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
$MAMBA_EXE run -n genpu2 python scripts/tracker_track_plots_v4.py \
    --ckpt $CK/helix_pion_v4/checkpoint_050000.pt \
    --slice /scratch/gpfs/IOJALVO/lv7805/genpu_data/tracker_slice/helix_pion.npz \
    --outdir /home/lv7805/genpu/plots/tracker/v4_eval
echo "=== done ==="
