#!/bin/bash
#SBATCH --job-name=calo_auto_train
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

# LEVER 2 — the AUTO anchor for e± (helix below pT 0.035 GeV, straight line above), the
# bremsstrahlung fix. Probe says pooled phi tightening 2.69x vs 1.86x for the pure helix, and it
# removes the helix's pT collapse (4.52x -> 0.43x) that showed up as physical-frame over-dispersion
# rising monotonically to 1.93 in the Phase 1 e± checkpoint.
#
# Two runs so the two levers stay attributable:
#   electron_auto        = anchor kind swap ONLY  (vs electron_anchor: gate8 0.7456/0.7488)
#   electron_auto_cond   = kind swap + anchor conditioning (both levers, the candidate config)
# `--anchor_kind auto` at metric time MUST match the slice, or the generation frame silently
# disagrees with the training frame.

echo "=== train electron_auto (60k, anchor kind swap only) ==="
$RUN scripts/train_calo_flow.py --slice $SL/electron_auto.npz --out $CK/electron_auto \
    --steps 60000 --pos_transform quantile --count_dither --no_energy_glob \
    --run_name calo_electron_auto --no_wandb

for spec in "0:electron" "1:positron"; do
    cls="${spec%%:*}"; name="${spec##*:}"
    echo "=== metrics: auto_$name (pdg $cls) ==="
    $RUN scripts/calo_metrics.py --ckpt $CK/electron_auto/checkpoint_060000.pt \
        --pdg_class $cls --shard 5 --tag "auto_$name" --no_partition --anchor_kind auto
done
echo "=== e± core diag (auto) ==="
$RUN scripts/calo_core_diag.py --slice $SL/electron_auto.npz \
    --ckpt $CK/electron_auto/checkpoint_060000.pt --tag electron_auto --max_showers 300000

echo "=== train electron_auto_cond (60k, both levers) ==="
$RUN scripts/train_calo_flow.py --slice $SL/electron_auto.npz --out $CK/electron_auto_cond \
    --steps 60000 --pos_transform quantile --count_dither --no_energy_glob --anchor_cond \
    --run_name calo_electron_auto_cond --no_wandb

for spec in "0:electron" "1:positron"; do
    cls="${spec%%:*}"; name="${spec##*:}"
    echo "=== metrics: autocond_$name (pdg $cls) ==="
    $RUN scripts/calo_metrics.py --ckpt $CK/electron_auto_cond/checkpoint_060000.pt \
        --pdg_class $cls --shard 5 --tag "autocond_$name" --no_partition --anchor_kind auto
done
echo "=== e± core diag (auto + cond) ==="
$RUN scripts/calo_core_diag.py --slice $SL/electron_auto.npz \
    --ckpt $CK/electron_auto_cond/checkpoint_060000.pt --tag electron_autocond --max_showers 300000
echo "=== done ==="
