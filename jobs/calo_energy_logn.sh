#!/bin/bash
#SBATCH --job-name=calo_energy_logn
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

# ARM A — does giving the energy head MULTIPLICITY fix frac_near_floor at all?
# Single variable vs the Phase 1 helix checkpoints: `--energy_glob_idx 1` (log_n only), shared
# trunk, no anchor conditioning, same slices and steps.
#
# The 2026-08-14 diagnosis: frac_near_floor is a function of multiplicity, which is the
# GlobalHead's variable. With --no_energy_glob the energy head is never told it and must infer it
# implicitly through the shared trunk — and FIVE separate changes that disturbed that route all
# degraded frac_near_floor (width_norm 0.616->0.858, ctx_norm 0.616->0.858, anchor_cond
# 0.601->0.660, auto, cond+separate-trunks 0.601->0.828). Confirmed by likelihood, not just the
# gate: e± val_ehl 0.2655 (separate trunks) vs 0.2141 (shared) at 60k.
#
# NOT the ruled-out "energy head on the sampled global" (2026-08-13): that failure was dim 0,
# total_logE, making continuous cell energy a sharp function of a noisy energy SCALE, paired with
# partition. Here the head gets dim 1 only, a COUNT — and at generation it is handed the SAME
# sampled log_n that set the cell count, so the conditioning is self-consistent by construction.
#
# TARGETS (beat these on frac_near_floor without losing the gate):
#   pion  `pion_anchor`     : gate8 0.8055 gateW 0.9833 cell_logE 0.0223 frac_floor 0.6176
#   e-    `electron_anchor` : gate8 0.7456 gateW 0.9639 cell_logE 0.0153 frac_floor 0.6008
#   e+                      : gate8 0.7488 gateW 0.9578 cell_logE 0.0170 frac_floor 0.5924

echo "=== train pion_elogn (40k, shared trunk, energy head sees log_n) ==="
$RUN scripts/train_calo_flow.py --slice $SL/pion_anchor.npz --out $CK/pion_elogn \
    --steps 40000 --pos_transform quantile --count_dither --energy_glob_idx 1 \
    --run_name calo_pion_elogn --no_wandb

echo "=== pion metrics ==="
$RUN scripts/calo_metrics.py --ckpt $CK/pion_elogn/checkpoint_040000.pt \
    --pdg_class 3 4 --shard 5 --tag elogn_pion --no_partition

echo "=== train electron_elogn (60k, shared trunk, energy head sees log_n) ==="
$RUN scripts/train_calo_flow.py --slice $SL/electron_anchor.npz --out $CK/electron_elogn \
    --steps 60000 --pos_transform quantile --count_dither --energy_glob_idx 1 \
    --run_name calo_electron_elogn --no_wandb

for spec in "0:electron" "1:positron"; do
    cls="${spec%%:*}"; name="${spec##*:}"
    echo "=== metrics: $name (pdg $cls) ==="
    $RUN scripts/calo_metrics.py --ckpt $CK/electron_elogn/checkpoint_060000.pt \
        --pdg_class $cls --shard 5 --tag "elogn_$name" --no_partition
done
echo "=== done ==="
