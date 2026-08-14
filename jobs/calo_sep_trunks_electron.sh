#!/bin/bash
#SBATCH --job-name=calo_sep_e
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

# Phase 0c on e± — the species that SHOWED the trunk interference twice, so the most direct test that
# separating the trunks removes it. Same slice and 60k steps as the `electron_v1` reference
# (gate8 0.773 / 0.760, gate10 0.971 / 0.968, cell_logE 0.017, frac_floor AUC 0.616).
echo "=== train electron_sep (60k, separate trunks) ==="
$RUN scripts/train_calo_flow.py --slice $SL/electron_v1.npz --out $CK/electron_sep \
    --steps 60000 --pos_transform quantile --count_dither --no_energy_glob \
    --separate_trunks --run_name calo_electron_sep --no_wandb

for spec in "0:electron" "1:positron"; do
    cls="${spec%%:*}"; name="${spec##*:}"
    echo "=== metrics: $name (pdg $cls) ==="
    $RUN scripts/calo_metrics.py --ckpt $CK/electron_sep/checkpoint_060000.pt \
        --pdg_class $cls --shard 5 --tag "sep_$name" --no_partition \
        --logE_max_slice $SL/electron_v1.npz
done
echo "=== done ==="
