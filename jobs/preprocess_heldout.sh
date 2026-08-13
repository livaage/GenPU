#!/bin/bash
#SBATCH --job-name=preprocess_s5
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
export MAMBA_ROOT_PREFIX=/scratch/gpfs/IOJALVO/gnn-tracking/object_condensation/micromamba
export MAMBA_EXE=/home/lv7805/bin/micromamba
export HF_HOME=/scratch/gpfs/IOJALVO/lv7805/genpu_cache
cd /home/lv7805/genpu
# held-out shard 5 (training used 0-2) -> stage2, for out-of-sample evaluation
$MAMBA_EXE run -n genpu2 python scripts/preprocess_shard.py 5 \
    --output-dir /scratch/gpfs/IOJALVO/lv7805/genpu_data/preprocessed
echo "=== done ==="
