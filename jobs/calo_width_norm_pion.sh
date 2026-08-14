#!/bin/bash
#SBATCH --job-name=calo_wn_pion
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

# Same width-in-global fix on the pion — the species where width is the most extreme discriminator
# (width_std single-feature AUC 0.963, width_mean 0.945, gate10 0.993). CONTROLLED vs
# `pion_qtd_v2` (gate8 0.809, gate10 0.993): same shards 0-2, no per-class cap, same 40k steps.
echo "=== build pion width-normalised slice ==="
$RUN scripts/build_calo_slice.py --shards 0 1 2 --pdg_class 3 4 --width_norm \
    --out $SL/pion_wn.npz

echo "=== train pion_wn (40k steps) ==="
$RUN scripts/train_calo_flow.py \
    --slice $SL/pion_wn.npz --out $CK/pion_wn \
    --steps 40000 --pos_transform quantile --count_dither \
    --no_energy_glob --run_name calo_pion_wn --no_wandb

CKPT=$CK/pion_wn/checkpoint_040000.pt
echo "=== pion metrics, width_renorm ON ==="
$RUN scripts/calo_metrics.py --ckpt $CKPT --pdg_class 3 4 --shard 5 --tag wn_pion \
    --no_partition --width_renorm --logE_max_slice $SL/pion_wn.npz
echo "=== pion metrics, width_renorm OFF ==="
$RUN scripts/calo_metrics.py --ckpt $CKPT --pdg_class 3 4 --shard 5 --tag wn_pion_norenorm \
    --no_partition --logE_max_slice $SL/pion_wn.npz
echo "=== done ==="
