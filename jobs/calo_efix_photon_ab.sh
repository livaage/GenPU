#!/bin/bash
#SBATCH --job-name=calo_efix_photon_ab
#SBATCH --time=00:30:00
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
RUN="$MAMBA_EXE run -n genpu2 python"

# Complete the photon 2x2 (A and C already ran): the C regression (gate 0.557 -> 0.809, with
# cell_logE W/sigma 0.016 -> 0.084) changed both the checkpoint and the generation path, so
# attribute it. B = partition on the OLD ckpt, D = new ckpt WITHOUT partition.
echo "=== photon B: baseline ckpt, partition only ==="
$RUN scripts/calo_metrics.py --ckpt $CK/photon_qtd_v1/checkpoint_040000.pt \
    --pdg_class 2 --shard 5 --tag photon_B_base_part
echo "=== photon D: fixed ckpt, sum-of-cells (glob conditioning + qt_total alone) ==="
$RUN scripts/calo_metrics.py --ckpt $CK/photon_efix_v2/checkpoint_040000.pt \
    --pdg_class 2 --shard 5 --tag photon_D_efix_sum --no_partition
echo "=== done ==="
