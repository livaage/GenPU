#!/bin/bash
#SBATCH --job-name=calo_core_born
#SBATCH --time=00:50:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --qos=gpu-test
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
export MAMBA_ROOT_PREFIX=/scratch/gpfs/IOJALVO/gnn-tracking/object_condensation/micromamba
export MAMBA_EXE=/home/lv7805/bin/micromamba
export HF_HOME=/scratch/gpfs/IOJALVO/lv7805/genpu_cache
cd /home/lv7805/genpu
CK=/scratch/gpfs/IOJALVO/lv7805/genpu_data/checkpoints/calo_flow/electron_anchor/checkpoint_060000.pt
SL=/scratch/gpfs/IOJALVO/lv7805/genpu_data/calo_slice/electron_anchor.npz

# AUDIT 1(a) — is the Phase 1 headline ("e± core spread 0.585 -> 0.99") a statement about the HELIX
# anchor, or about the turning-point FALLBACK that 91% of born-inside fragments receive?
# The slice is 73.4% turning branch (anchor_mode counts 91,624 / 121,136 / 587,238), and 64.8% of
# depositors are born inside the calo. If the 0.99 holds only on the pooled/inside set and degrades
# on born-outside showers, then the current best calo config is not doing what the entry claims.
for b in all outside inside; do
    echo "########## born = $b ##########"
    $MAMBA_EXE run -n genpu2 python scripts/calo_core_diag.py \
        --slice "$SL" --ckpt "$CK" --tag "electron_born_$b" --born "$b"
done
echo "=== done ==="
