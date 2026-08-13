#!/bin/bash
#SBATCH --job-name=train_ms_v3_ss
#SBATCH --time=00:58:00
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --qos=gpu-test
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
export MAMBA_ROOT_PREFIX=/scratch/gpfs/IOJALVO/gnn-tracking/object_condensation/micromamba
export MAMBA_EXE=/home/lv7805/bin/micromamba
export HF_HOME=/scratch/gpfs/IOJALVO/lv7805/genpu_cache
cd /home/lv7805/genpu
CK=/scratch/gpfs/IOJALVO/lv7805/genpu_data/checkpoints/tracker
SL=/scratch/gpfs/IOJALVO/lv7805/genpu_data/tracker_slice

# All-charged-species v3 from scratch with a single TF->SS schedule:
# teacher-forced for the first 18k (learn the conditional), ramp scheduled sampling to 0.4 by 30k.
$MAMBA_EXE run -n genpu2 python scripts/train_tracker_module.py \
    --slice $SL/surface_multispecies.npz \
    --out $CK/surface_ms_v3_ss \
    --steps 50000 --batch 512 --lr 3e-4 \
    --ss_prob 0.4 --ss_warmup 18000 --ss_ramp 12000 \
    --save_every 25000 --log_every 500 --no_wandb --run_name ms_v3_ss
echo "=== training done ==="
