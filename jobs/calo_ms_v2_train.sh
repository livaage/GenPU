#!/bin/bash
#SBATCH --job-name=calo_ms_train
#SBATCH --time=00:55:00
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=4
#SBATCH --mem=196G
#SBATCH --qos=gpu-test
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
export MAMBA_ROOT_PREFIX=/scratch/gpfs/IOJALVO/gnn-tracking/object_condensation/micromamba
export MAMBA_EXE=/home/lv7805/bin/micromamba
cd /home/lv7805/genpu
RUN="$MAMBA_EXE run -n genpu2 python"
SL=/scratch/gpfs/IOJALVO/lv7805/genpu_data/calo_slice
CK=/scratch/gpfs/IOJALVO/lv7805/genpu_data/checkpoints/calo_flow

# ONE PDG-conditioned head over all 17 classes (decision of 2026-08-13), now on re-attributed 3D
# showers. 60k steps and 2 seeds MATCH the electron_v2 recipe exactly, so the only difference from
# the 0.8105 / 0.8074 baseline is the data mixture -- evaluating this model on classes 0,1 against
# the same held-out population measures the cost of pooling and nothing else.
for s in 0 1; do
  echo "########## TRAIN multispecies seed $s ##########"
  $RUN scripts/train_calo_flow.py --slice $SL/multispecies_v2.npz \
       --out $CK/multispecies_v2_s$s --steps 60000 --seed $s --no_wandb
done
echo "=== done ==="
