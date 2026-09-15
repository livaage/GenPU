#!/bin/bash
#SBATCH --job-name=calo_ms_eval_sp
#SBATCH --time=00:55:00
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --qos=gpu-test
#SBATCH --array=0-1
#SBATCH --output=logs/%x-%A_%a.out
set -uo pipefail
export MAMBA_ROOT_PREFIX=/scratch/gpfs/IOJALVO/gnn-tracking/object_condensation/micromamba
export MAMBA_EXE=/home/lv7805/bin/micromamba
cd /home/lv7805/genpu
RUN="$MAMBA_EXE run -n genpu2 python"
SL=/scratch/gpfs/IOJALVO/lv7805/genpu_data/calo_slice
CK=/scratch/gpfs/IOJALVO/lv7805/genpu_data/checkpoints/calo_flow
FD=/scratch/gpfs/IOJALVO/lv7805/genpu_data/gate_features
mkdir -p $FD
s=$SLURM_ARRAY_TASK_ID
M=$CK/multispecies_v2_s$s/checkpoint_060000.pt

# PER-SPECIES breakdown. Under v2 these populations are not the v1 ones: re-attribution makes gamma
# the LARGEST class (410k showers/shard vs e- 339k) because the e± fragments that used to be their
# own "showers" now book to their photon ancestor. No v1 per-species number is comparable.
# gamma is the control -- it split 0.1% under v1, so re-attribution should barely move it, while
# pi± and p are the deep hadronic showers the 3D model has never been trained on.
for spec in "gam 2" "pi 3 4" "prot 7" "neut 9"; do
  set -- $spec; tag=$1; shift
  echo "########## seed $s: $tag (classes $*) ##########"
  $RUN scripts/calo_metrics.py --ckpt $M --pdg_class "$@" \
       --real_slice $SL/multispecies_v2_h5.npz --tag ms_${tag}_s$s \
       --dump_features $FD/ms_${tag}_s$s.npz
done
echo "=== done ==="
