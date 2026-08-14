#!/bin/bash
#SBATCH --job-name=calo_anchor_train
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

# PHASE 1 part B: train on the helix-anchored slices. SHARED TRUNK (no --separate_trunks) — Phase 0c
# showed the split helps where there is data volume but regressed the photon, so the shipping config
# stays shared and the trunk split is kept as a diagnostic instrument.
#
# CONTROLLED vs the references — same steps, same recipe, ONLY the core frame changes:
#   pion   `pion_qtd_v2`  (40k, slice pion_ctr_v2) : gate8 0.809, gate10 0.993, width_std AUC 0.963
#   e±     `electron_v1`  (60k, slice electron_v1) : gate8 0.773 / 0.760, gate10 0.971 / 0.968
# The new checkpoints carry their own logE_max / logE_max_pdg from the slice, so no metric override
# is needed (the references needed --logE_max / --logE_max_slice only because they predate it).
#
# MECHANISM CHECK is the headline, not the gate: calo_core_diag reports the gen/real per-(charge x
# pT)-bin core spread in the PHYSICAL frame (residual + anchor), directly comparable with the
# pre-anchor 0.58x (e±) / 0.94x (pion). Target ~1.0.

echo "=== train pion_anchor (40k, shared trunk) ==="
$RUN scripts/train_calo_flow.py --slice $SL/pion_anchor.npz --out $CK/pion_anchor \
    --steps 40000 --pos_transform quantile --count_dither --no_energy_glob \
    --run_name calo_pion_anchor --no_wandb

echo "=== pion metrics (held-out shard 5) ==="
$RUN scripts/calo_metrics.py --ckpt $CK/pion_anchor/checkpoint_040000.pt \
    --pdg_class 3 4 --shard 5 --tag anchor_pion --no_partition

echo "=== pion core diag (mechanism: gen/real per-bin spread 0.94 -> ~1.0?) ==="
$RUN scripts/calo_core_diag.py --slice $SL/pion_anchor.npz \
    --ckpt $CK/pion_anchor/checkpoint_040000.pt --tag pion_anchor

echo "=== train electron_anchor (60k, shared trunk) ==="
$RUN scripts/train_calo_flow.py --slice $SL/electron_anchor.npz --out $CK/electron_anchor \
    --steps 60000 --pos_transform quantile --count_dither --no_energy_glob \
    --run_name calo_electron_anchor --no_wandb

for spec in "0:electron" "1:positron"; do
    cls="${spec%%:*}"; name="${spec##*:}"
    echo "=== metrics: $name (pdg $cls) ==="
    $RUN scripts/calo_metrics.py --ckpt $CK/electron_anchor/checkpoint_060000.pt \
        --pdg_class $cls --shard 5 --tag "anchor_$name" --no_partition
done

echo "=== e± core diag (mechanism: gen/real per-bin spread 0.58 -> ~1.0?) ==="
$RUN scripts/calo_core_diag.py --slice $SL/electron_anchor.npz \
    --ckpt $CK/electron_anchor/checkpoint_060000.pt --tag electron_anchor

echo "=== done ==="
