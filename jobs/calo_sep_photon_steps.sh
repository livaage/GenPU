#!/bin/bash
#SBATCH --job-name=calo_sep_ph_steps
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

# Is the photon's separate-trunk regression (gate8 0.558 -> 0.679, cell_logE 0.016 -> 0.027) a WORSE
# MODEL or just UNDERTRAINED? Evidence for undertrained: at 40k the separate-trunk val_ehl was still
# descending (0.1730 -> 0.1654 -> 0.1631 -> 0.1547), val_cfm was identical to shared (1.3933 vs
# 1.3936) and val_gnll was BETTER (-1.226 vs -1.081) — the damage is confined to the energy head,
# whose trunk now gets gradient from one loss instead of a representation built by three.
# Test at MATCHED 120k steps so neither configuration is favoured by the step budget.
# (Photon is the small-data species: ~4.2M cells vs the pion's 39M — where a shared trunk's
#  statistical sharing should matter most, and where the pion showed no such regression.)

echo "=== photon SEPARATE trunks, 120k ==="
$RUN scripts/train_calo_flow.py --slice $SL/photon_core.npz --out $CK/photon_sep_120k \
    --steps 120000 --pos_transform quantile --count_dither --no_energy_glob \
    --separate_trunks --run_name calo_photon_sep_120k --no_wandb
echo "=== metrics ==="
$RUN scripts/calo_metrics.py --ckpt $CK/photon_sep_120k/checkpoint_120000.pt \
    --pdg_class 2 --shard 5 --tag sep120_photon --no_partition --logE_max -2.5456

echo "=== photon SHARED trunk, 120k (matched-step control) ==="
$RUN scripts/train_calo_flow.py --slice $SL/photon_core.npz --out $CK/photon_shared_120k \
    --steps 120000 --pos_transform quantile --count_dither --no_energy_glob \
    --run_name calo_photon_shared_120k --no_wandb
echo "=== metrics ==="
$RUN scripts/calo_metrics.py --ckpt $CK/photon_shared_120k/checkpoint_120000.pt \
    --pdg_class 2 --shard 5 --tag shared120_photon --no_partition --logE_max -2.5456
echo "=== done ==="
