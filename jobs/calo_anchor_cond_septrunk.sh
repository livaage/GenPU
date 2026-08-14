#!/bin/bash
#SBATCH --job-name=calo_acond_sep
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

# Anchor conditioning RETESTED WITH SEPARATE TRUNKS — the diagnosed cause of its failure.
#
# On a shared trunk (job 12376191) anchor_cond did exactly what it was designed to do:
#   e± physical-frame core mechanism 1.275 -> 1.003, branch ratios 1.17/1.48/0.85 -> 0.87/0.74/0.93,
#   d_eta W/sigma 0.0269 -> 0.0067
# and paid for it in the untouched ENERGY head:
#   frac_near_floor AUC 0.601 -> 0.660, logE_mean +0.043, cell_logE 0.0153 -> 0.0186, gate8 0.746 -> 0.886
# — the identical fingerprint to width_norm and ctx_norm on 2026-08-13. The anchor features are
# injected AFTER the trunk, which blocks forward contamination but NOT the gradient the GlobalHead
# sends back into the shared trunk. Separate trunks make that impossible (+10.6k params, ~4%).
#
# This is also the config Phase 0c independently recommended for pion / e± (photon stays shared);
# Phase 1 deliberately deferred it to keep the frame comparison like-for-like.
#
# PREDICTION being tested: the 1.003 mechanism survives and the gate recovers to at least the
# Phase 1 helix numbers. Honest risk: Phase 0c measured separate trunks helping e± alone
# (gate8 0.773 -> 0.739), but the two effects need not compose.
#
# Reference grid (all shared-trunk unless noted):
#   pion   : baseline 0.8092 | helix 0.8055 | helix+cond 0.8155 | sep-trunk-only 0.7973
#   e-     : baseline 0.7729 | helix 0.7456 | helix+cond 0.8857 | sep-trunk-only 0.7385
#   e+     : baseline 0.7597 | helix 0.7488 | helix+cond 0.8712 | sep-trunk-only 0.7411

echo "=== train pion_anchor_cond_sep (40k, SEPARATE trunks + anchor conditioning) ==="
$RUN scripts/train_calo_flow.py --slice $SL/pion_anchor.npz --out $CK/pion_anchor_cond_sep \
    --steps 40000 --pos_transform quantile --count_dither --no_energy_glob \
    --anchor_cond --separate_trunks --run_name calo_pion_acond_sep --no_wandb

echo "=== pion metrics ==="
$RUN scripts/calo_metrics.py --ckpt $CK/pion_anchor_cond_sep/checkpoint_040000.pt \
    --pdg_class 3 4 --shard 5 --tag acondsep_pion --no_partition

echo "=== pion core diag ==="
$RUN scripts/calo_core_diag.py --slice $SL/pion_anchor.npz \
    --ckpt $CK/pion_anchor_cond_sep/checkpoint_040000.pt --tag pion_acondsep --max_showers 300000

echo "=== train electron_anchor_cond_sep (60k, SEPARATE trunks + anchor conditioning) ==="
$RUN scripts/train_calo_flow.py --slice $SL/electron_anchor.npz --out $CK/electron_anchor_cond_sep \
    --steps 60000 --pos_transform quantile --count_dither --no_energy_glob \
    --anchor_cond --separate_trunks --run_name calo_electron_acond_sep --no_wandb

for spec in "0:electron" "1:positron"; do
    cls="${spec%%:*}"; name="${spec##*:}"
    echo "=== metrics: $name (pdg $cls) ==="
    $RUN scripts/calo_metrics.py --ckpt $CK/electron_anchor_cond_sep/checkpoint_060000.pt \
        --pdg_class $cls --shard 5 --tag "acondsep_$name" --no_partition
done

echo "=== e± core diag (does the 1.003 mechanism survive?) ==="
$RUN scripts/calo_core_diag.py --slice $SL/electron_anchor.npz \
    --ckpt $CK/electron_anchor_cond_sep/checkpoint_060000.pt --tag electron_acondsep --max_showers 300000
echo "=== done ==="
