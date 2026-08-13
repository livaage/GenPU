#!/bin/bash
#SBATCH --job-name=probe_stdplots
#SBATCH --time=00:35:00
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
COUNT=$CK/count_head_d0_selfnorm.pt

# PROBE: gate the 25k checkpoint (ss=0.23) vs the 50k (ss=0.40) 0.61 — did lighter SS over-correct less?
echo "############ v3-ms @ 25k (ss=0.23) HONEST gate ############"
$MAMBA_EXE run -n genpu2 python scripts/tracker_honest_gate_module.py \
    --ckpt $CK/surface_ms_v3_ss/checkpoint_025000.pt --count_ckpt $COUNT \
    --slice $SL/surface_multispecies.npz --pdg_class 0 1 3 4 7 8 11 12 --shard 0

# STANDARD PLOTS for v3-ms (50k) — regenerate the canonical tracker diagnostics for v3.
echo "############ v3-ms standard plots (tracks / track_feats / hit_hists) ############"
$MAMBA_EXE run -n genpu2 python scripts/tracker_track_plots_v3.py \
    --ckpt $CK/surface_ms_v3_ss/checkpoint_050000.pt --slice $SL/surface_multispecies.npz \
    --outdir /home/lv7805/genpu/plots/tracker/v3_ms_eval
echo "=== done ==="
