#!/bin/bash
#SBATCH --job-name=calo_ctx_norm
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

# CONTEXT NORMALISATION of the shower core (glob dims 2,3), per (charge sign x pT bin), fit from
# TRUTH conditioning so the context is exactly known at generation — the structural difference from
# the sampled-global conditioning that failed earlier.
# Why: the core carries 83-93% of the "shower width" variance the gate reacts to. Its MARGINAL is
# already matched (pooled quantile transform), but per (charge x pT) bin the model generates only
# 0.58x (e±) / 0.94x (pion) of the real core spread — the pooled transform leaves each conditional
# slice heavy-tailed and differently scaled (2.4x / 4.0x across bins), so the mixture regresses to
# the mean. Normalising each context to zero-median / unit-IQR first makes the pooled transform act
# on a representative shape.
# CONTROLLED: same slices, same steps, same recipe as the references — ctx_norm is the only change.
#   e± reference `electron_v1`: gate8 0.773/0.760, gate10 0.971/0.968
#   pion reference `pion_qtd_v2`: gate8 0.809, gate10 0.993

echo "=== train electron_ctx (60k steps, ctx_norm) ==="
$RUN scripts/train_calo_flow.py \
    --slice $SL/electron_v1.npz --out $CK/electron_ctx \
    --steps 60000 --pos_transform quantile --count_dither --no_energy_glob \
    --ctx_norm --ctx_pt_bins 8 --run_name calo_electron_ctx --no_wandb

echo "=== e- metrics ==="
$RUN scripts/calo_metrics.py --ckpt $CK/electron_ctx/checkpoint_060000.pt \
    --pdg_class 0 --shard 5 --tag ctx_electron --no_partition --logE_max_slice $SL/electron_v1.npz
echo "=== e+ metrics ==="
$RUN scripts/calo_metrics.py --ckpt $CK/electron_ctx/checkpoint_060000.pt \
    --pdg_class 1 --shard 5 --tag ctx_positron --no_partition --logE_max_slice $SL/electron_v1.npz
echo "=== e± core diag (mechanism check: gen/real per-bin scale should go 0.58 -> ~1.0) ==="
$RUN scripts/calo_core_diag.py --slice $SL/electron_wn.npz \
    --ckpt $CK/electron_ctx/checkpoint_060000.pt --tag electron_ctx

echo "=== train pion_ctx (40k steps, ctx_norm) ==="
$RUN scripts/train_calo_flow.py \
    --slice $SL/pion_ctr_v2.npz --out $CK/pion_ctx \
    --steps 40000 --pos_transform quantile --count_dither --no_energy_glob \
    --ctx_norm --ctx_pt_bins 8 --run_name calo_pion_ctx --no_wandb

echo "=== pion metrics ==="
$RUN scripts/calo_metrics.py --ckpt $CK/pion_ctx/checkpoint_040000.pt \
    --pdg_class 3 4 --shard 5 --tag ctx_pion --no_partition --logE_max -1.1328
echo "=== pion core diag ==="
$RUN scripts/calo_core_diag.py --slice $SL/pion_wn.npz \
    --ckpt $CK/pion_ctx/checkpoint_040000.pt --tag pion_ctx
echo "=== done ==="
