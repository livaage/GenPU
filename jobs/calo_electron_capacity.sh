#!/bin/bash
#SBATCH --job-name=calo_e_capacity
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

# CAPACITY TEST for the biggest species. e± are 48% of deposited calo energy and 57% of cells; the
# shared 17-class head gives e- 0.888 / e+ 0.887 while the single-species photon head reaches 0.557
# on the same architecture. Is the gap shared capacity, or are e± simply harder than photons?
# Controlled: SAME per-class data (400k/class, as in multispecies_v1) and SAME 60k steps — the only
# difference is that this model serves 2 classes instead of 17 (and fits its norms on e± alone).
#   -> lands near 0.56-0.62 : the multi-species gap is capacity -> widen / per-class output heads
#   -> stays near 0.85      : e± are genuinely harder; the shared head is exonerated
echo "=== build e± slice ==="
$RUN scripts/build_calo_slice.py --shards 0 1 2 --pdg_class 0 1 --max_per_class 400000 \
    --out $SL/electron_v1.npz

echo "=== train electron_v1 (2 classes, 60k steps) ==="
$RUN scripts/train_calo_flow.py \
    --slice $SL/electron_v1.npz --out $CK/electron_v1 \
    --steps 60000 --pos_transform quantile --count_dither \
    --no_energy_glob --run_name calo_electron_v1 --no_wandb

for spec in "0:electron" "1:positron"; do
    cls="${spec%%:*}"; name="${spec##*:}"
    echo "=== single-species metrics: $name (pdg $cls) ==="
    $RUN scripts/calo_metrics.py --ckpt $CK/electron_v1/checkpoint_060000.pt \
        --pdg_class $cls --shard 5 --tag "ss_$name" --no_partition \
        --logE_max_slice $SL/electron_v1.npz
done
echo "=== done ==="
