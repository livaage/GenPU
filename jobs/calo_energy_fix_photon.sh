#!/bin/bash
#SBATCH --job-name=calo_efix_photon
#SBATCH --time=00:40:00
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

# Same energy fix on the photon (gate 0.557 baseline). The photon's per-cell energy was
# already good, but its GlobalHead total is unbounded above — a rare draw reached 1e9 x
# E_true, and the tail cells put the lowest-energy bin's resolution at 2.2 vs real 1.07.
# So the photon tests the fix's DOWNSIDE risk: it must not regress a working species.
echo "=== train photon_efix_v2 ==="
$RUN scripts/train_calo_flow.py \
    --slice $SL/photon_core.npz --out $CK/photon_efix_v2 \
    --steps 40000 --pos_transform quantile --count_dither --qt_total \
    --run_name calo_photon_efix_v2 --no_wandb

echo "=== photon: baseline ckpt, sum-of-cells (reproduces 0.557) ==="
$RUN scripts/calo_metrics.py --ckpt $CK/photon_qtd_v1/checkpoint_040000.pt \
    --pdg_class 2 --shard 5 --tag photon_A_base_sum --no_partition
echo "=== photon: fixed ckpt, partition ==="
$RUN scripts/calo_metrics.py --ckpt $CK/photon_efix_v2/checkpoint_040000.pt \
    --pdg_class 2 --shard 5 --tag photon_C_efix_part
echo "=== done ==="
