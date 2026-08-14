#!/bin/bash
#SBATCH --job-name=calo_efix_pion
#SBATCH --time=00:55:00
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
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

# Pion calo energy fix (2026-08-13 audit). Two changes, measured as a 2x2 against the
# pion_qtd_v2 baseline (held-out gate 0.809, energy resolution 1.3-1.8 vs real 0.9-1.2):
#   (1) energy head conditions on the global (total_logE, log_n) — the shower energy scale
#       is R^2 0.05 from particle features but 0.85 with the global;
#   (2) cells are renormalised to the SAMPLED total at generation (partition), instead of
#       the total being a sum of independently drawn cells (W/sigma 0.073 vs 0.017).
# Plus --qt_total: bounded quantile inverse on total_logE (the unbounded mixture tail put
# the photon total at 1e9 x E_true).
echo "=== train pion_efix_v3 ==="
$RUN scripts/train_calo_flow.py \
    --slice $SL/pion_ctr_v2.npz --out $CK/pion_efix_v3 \
    --steps 40000 --pos_transform quantile --count_dither --qt_total \
    --run_name calo_pion_efix_v3 --no_wandb

# 2x2: {baseline ckpt, fixed ckpt} x {sum-of-cells, partition}. Isolates which change did it.
echo "=== A: baseline ckpt, sum-of-cells (reproduces 0.809) ==="
$RUN scripts/calo_metrics.py --ckpt $CK/pion_qtd_v2/checkpoint_040000.pt \
    --pdg_class 3 4 --shard 5 --tag pion_A_base_sum --no_partition
echo "=== B: baseline ckpt, partition only ==="
$RUN scripts/calo_metrics.py --ckpt $CK/pion_qtd_v2/checkpoint_040000.pt \
    --pdg_class 3 4 --shard 5 --tag pion_B_base_part
echo "=== C: fixed ckpt, partition (the candidate) ==="
$RUN scripts/calo_metrics.py --ckpt $CK/pion_efix_v3/checkpoint_040000.pt \
    --pdg_class 3 4 --shard 5 --tag pion_C_efix_part
echo "=== D: fixed ckpt, sum-of-cells (glob conditioning alone) ==="
$RUN scripts/calo_metrics.py --ckpt $CK/pion_efix_v3/checkpoint_040000.pt \
    --pdg_class 3 4 --shard 5 --tag pion_D_efix_sum --no_partition
echo "=== done ==="
