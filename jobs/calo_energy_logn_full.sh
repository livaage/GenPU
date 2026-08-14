#!/bin/bash
#SBATCH --job-name=calo_elogn_full
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

# ARM B — the full candidate: separate trunks + anchor conditioning + energy head on log_n.
#
# anchor_cond is confirmed to FIX the core mechanism robustly (e± physical core spread 1.275 -> 1.00,
# pion 0.929 -> 1.02, branch ratios -> ~0.8-1.0, identical val_cfm/val_gnll) with shared OR separate
# trunks. The only thing blocking it is the energy head's floor fraction, and separating the trunks
# made that WORSE (0.601 -> 0.828), because it removed the implicit multiplicity route entirely.
# So supply multiplicity EXPLICITLY and the isolation should become free — that is the whole point
# of separate trunks, which Phase 0c independently recommended for pion / e±.
#
# If ARM A works and this does not, the extra damage is not about multiplicity after all.
# If both work, this is the shipping config: fixed mechanism AND no trunk coupling.
#
# TARGETS:
#   pion  best gate `pion_anchor` 0.8055 / gateW 0.9833; `pion_acondsep` mechanism 1.023, floor 0.5868
#   e-    best gate `electron_anchor` 0.7456 / gateW 0.9639; `electron_acondsep` mechanism 1.014,
#         floor 0.8282, cell_logE 0.0796  <- the damage this arm must remove

echo "=== train pion_full (40k, separate trunks + anchor_cond + energy log_n) ==="
$RUN scripts/train_calo_flow.py --slice $SL/pion_anchor.npz --out $CK/pion_full \
    --steps 40000 --pos_transform quantile --count_dither --energy_glob_idx 1 \
    --anchor_cond --separate_trunks --run_name calo_pion_full --no_wandb

echo "=== pion metrics ==="
$RUN scripts/calo_metrics.py --ckpt $CK/pion_full/checkpoint_040000.pt \
    --pdg_class 3 4 --shard 5 --tag full_pion --no_partition
echo "=== pion core diag ==="
$RUN scripts/calo_core_diag.py --slice $SL/pion_anchor.npz \
    --ckpt $CK/pion_full/checkpoint_040000.pt --tag pion_full --max_showers 300000

echo "=== train electron_full (60k, separate trunks + anchor_cond + energy log_n) ==="
$RUN scripts/train_calo_flow.py --slice $SL/electron_anchor.npz --out $CK/electron_full \
    --steps 60000 --pos_transform quantile --count_dither --energy_glob_idx 1 \
    --anchor_cond --separate_trunks --run_name calo_electron_full --no_wandb

for spec in "0:electron" "1:positron"; do
    cls="${spec%%:*}"; name="${spec##*:}"
    echo "=== metrics: $name (pdg $cls) ==="
    $RUN scripts/calo_metrics.py --ckpt $CK/electron_full/checkpoint_060000.pt \
        --pdg_class $cls --shard 5 --tag "full_$name" --no_partition
done
echo "=== e± core diag (does the ~1.0 mechanism survive?) ==="
$RUN scripts/calo_core_diag.py --slice $SL/electron_anchor.npz \
    --ckpt $CK/electron_full/checkpoint_060000.pt --tag electron_full --max_showers 300000
echo "=== done ==="
