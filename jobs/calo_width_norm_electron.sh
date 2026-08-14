#!/bin/bash
#SBATCH --job-name=calo_wn_electron
#SBATCH --time=00:55:00
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
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

# WIDTH-IN-GLOBAL fix, on the biggest species (e± = 48% of calo energy, 57% of cells).
# Points are stored in units of each shower's own RMS width and log_width rides in the global
# (dim 4, quantile-normalised); generation scales the sampled cloud by the sampled width. This is
# the analogue of the per-shower core fix: the core removed per-shower LOCATION variance from the
# point flow, this removes per-shower SCALE variance. Motivation: width_std is a 0.90-0.91
# single-feature discriminator for e±, and generated widths were compressed to 0.79x the real
# spread against a real q10-q90 range of ~50x.
# CONTROLLED vs `electron_v1` (gate8 0.773/0.760, gate10 0.971/0.968): same 400k/class, same 60k
# steps, same recipe — width_norm is the only change.
echo "=== build e± width-normalised slice ==="
$RUN scripts/build_calo_slice.py --shards 0 1 2 --pdg_class 0 1 --max_per_class 400000 \
    --width_norm --out $SL/electron_wn.npz

echo "=== train electron_wn (60k steps) ==="
$RUN scripts/train_calo_flow.py \
    --slice $SL/electron_wn.npz --out $CK/electron_wn \
    --steps 60000 --pos_transform quantile --count_dither \
    --no_energy_glob --run_name calo_electron_wn --no_wandb

CKPT=$CK/electron_wn/checkpoint_060000.pt
echo "=== e- metrics, width_renorm ON ==="
$RUN scripts/calo_metrics.py --ckpt $CKPT --pdg_class 0 --shard 5 --tag wn_electron \
    --no_partition --width_renorm --logE_max_slice $SL/electron_wn.npz
echo "=== e- metrics, width_renorm OFF (A/B on the unit-RMS projection) ==="
$RUN scripts/calo_metrics.py --ckpt $CKPT --pdg_class 0 --shard 5 --tag wn_electron_norenorm \
    --no_partition --logE_max_slice $SL/electron_wn.npz
echo "=== e+ metrics, width_renorm ON ==="
$RUN scripts/calo_metrics.py --ckpt $CKPT --pdg_class 1 --shard 5 --tag wn_positron \
    --no_partition --width_renorm --logE_max_slice $SL/electron_wn.npz
echo "=== done ==="
