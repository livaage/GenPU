#!/bin/bash
#SBATCH --job-name=calo_anchor_cond
#SBATCH --time=00:55:00
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --qos=gpu-test
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
export MAMBA_ROOT_PREFIX=/scratch/gpfs/IOJALVO/gnn-tracking/object_condensation/micromamba
export MAMBA_EXE=/home/lv7805/bin/micromamba
export HF_HOME=/scratch/gpfs/IOJALVO/lv7805/genpu_cache
cd /home/lv7805/genpu
CK=/scratch/gpfs/IOJALVO/lv7805/genpu_data/checkpoints/calo_flow
SL=/scratch/gpfs/IOJALVO/lv7805/genpu_data/calo_slice
RUN="$MAMBA_EXE run -n genpu2 python"

# LEVER 1 — condition the GlobalHead (and ONLY the GlobalHead) on the anchor:
# [a_eta, a_phi, |a|] standardised + a 4-way branch one-hot.
# Why: measured on the Phase 1 checkpoints, the real residual core scale differs 10x (pion:
# 0.057 face vs 0.598 curler) / 15x (e±: 0.010 vs 0.132) between anchor branches, and the branch is
# a hard threshold (2R vs r_calo, arc length vs pi*R) that the trunk's smooth features cannot
# express — so the mixture blended the two populations, generating face showers 1.13-1.48x too wide
# and curlers 0.85-0.92x too narrow.
# Injected AFTER the shared trunk on purpose: feeding it upstream would move what the point and
# energy heads see (the interference measured twice on 2026-08-13).
# Truth-derived and deterministic, so no exposure-bias risk.
#
# CONTROLLED vs the Phase 1 checkpoints — same slices, same steps, --anchor_cond is the only change:
#   pion_anchor      (40k): gate8 0.8055, gateW 0.9833, width_std 0.9425, residual spread 0.93
#   electron_anchor  (60k): gate8 0.7456 / 0.7488, gateW 0.9639 / 0.9578, residual spread 0.99

echo "=== train pion_anchor_cond (40k, shared trunk, +anchor conditioning) ==="
$RUN scripts/train_calo_flow.py --slice $SL/pion_anchor.npz --out $CK/pion_anchor_cond \
    --steps 40000 --pos_transform quantile --count_dither --no_energy_glob --anchor_cond \
    --run_name calo_pion_anchor_cond --no_wandb

echo "=== pion metrics ==="
$RUN scripts/calo_metrics.py --ckpt $CK/pion_anchor_cond/checkpoint_040000.pt \
    --pdg_class 3 4 --shard 5 --tag acond_pion --no_partition

echo "=== pion core diag (target: branch ratios 1.13/0.93/0.92 -> ~1.0) ==="
$RUN scripts/calo_core_diag.py --slice $SL/pion_anchor.npz \
    --ckpt $CK/pion_anchor_cond/checkpoint_040000.pt --tag pion_acond --max_showers 300000

echo "=== train electron_anchor_cond (60k, shared trunk, +anchor conditioning) ==="
$RUN scripts/train_calo_flow.py --slice $SL/electron_anchor.npz --out $CK/electron_anchor_cond \
    --steps 60000 --pos_transform quantile --count_dither --no_energy_glob --anchor_cond \
    --run_name calo_electron_anchor_cond --no_wandb

for spec in "0:electron" "1:positron"; do
    cls="${spec%%:*}"; name="${spec##*:}"
    echo "=== metrics: $name (pdg $cls) ==="
    $RUN scripts/calo_metrics.py --ckpt $CK/electron_anchor_cond/checkpoint_060000.pt \
        --pdg_class $cls --shard 5 --tag "acond_$name" --no_partition
done

echo "=== e± core diag (target: branch ratios 1.17/1.48/0.85 -> ~1.0) ==="
$RUN scripts/calo_core_diag.py --slice $SL/electron_anchor.npz \
    --ckpt $CK/electron_anchor_cond/checkpoint_060000.pt --tag electron_acond --max_showers 300000
echo "=== done ==="
