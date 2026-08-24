#!/bin/bash
#SBATCH --job-name=incidence_head
#SBATCH --time=00:50:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --qos=gpu-test
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
export MAMBA_ROOT_PREFIX=/scratch/gpfs/IOJALVO/gnn-tracking/object_condensation/micromamba
export MAMBA_EXE=/home/lv7805/bin/micromamba
cd /home/lv7805/genpu
# Incidence head — P(tracker trace) and P(calo trace) per particle, the piece BOTH subsystems lack.
# Trains on shards 0-2 of the complete particle graph (every raw particle, so the negatives exist;
# stage2 cannot supply them, being the filtered set). Held out on shard 5.
# Judged on CALIBRATION and the 3-way joint, not accuracy: the irreducible residual is the
# conversion coin flip, which a generator must sample rather than predict.
$MAMBA_EXE run -n genpu2 python scripts/train_incidence_head.py --shards 0 1 2 --val_shard 5 --seed 0
