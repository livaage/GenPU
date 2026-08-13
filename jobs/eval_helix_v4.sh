#!/bin/bash
#SBATCH --job-name=eval_helix_v4
#SBATCH --time=00:30:00
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
SL=/scratch/gpfs/IOJALVO/lv7805/genpu_data/tracker_slice
echo "############ v4 HONEST gate (vs v3-pion 0.596) ############"
$MAMBA_EXE run -n genpu2 python scripts/tracker_honest_gate_helix.py \
    --ckpt $CK/helix_pion_v4/checkpoint_050000.pt --count_ckpt $CK/count_head_d0_selfnorm.pt \
    --slice $SL/helix_pion.npz --pdg_class 3 4 --shard 0
echo "############ v4 track_feats (coherence spikes) ############"
$MAMBA_EXE run -n genpu2 python scripts/tracker_track_plots_v4.py \
    --ckpt $CK/helix_pion_v4/checkpoint_050000.pt --slice $SL/helix_pion.npz \
    --outdir /home/lv7805/genpu/plots/tracker/v4_eval
echo "=== done ==="
