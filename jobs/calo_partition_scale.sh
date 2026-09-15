#!/bin/bash
#SBATCH --job-name=calo_part_scale
#SBATCH --time=00:25:00
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --qos=gpu-test
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
export MAMBA_ROOT_PREFIX=/scratch/gpfs/IOJALVO/gnn-tracking/object_condensation/micromamba
export MAMBA_EXE=/home/lv7805/bin/micromamba
cd /home/lv7805/genpu
RUN="$MAMBA_EXE run -n genpu2 python"
SL=/scratch/gpfs/IOJALVO/lv7805/genpu_data/calo_slice
CK=/scratch/gpfs/IOJALVO/lv7805/genpu_data/checkpoints/calo_flow

# Would `partition` SMEAR a truncated energy edge? Decides whether the floor truncation can be its
# own arm or has to wait for the simplex-fraction reparameterisation. See script docstring.
echo "########## e+- dedicated ##########"
$RUN scripts/calo_partition_scale.py --ckpt $CK/electron_v2_s0/checkpoint_060000.pt \
     --real_slice $SL/electron_v2_h5.npz --pdg_class 0 1 --tag ele
echo "########## pi+- pooled (hadronic: largest HCAL fraction of the big classes) ##########"
$RUN scripts/calo_partition_scale.py --ckpt $CK/multispecies_v2_s0/checkpoint_060000.pt \
     --real_slice $SL/multispecies_v2_h5.npz --pdg_class 3 4 --tag ms_pion
echo "########## all classes pooled ##########"
$RUN scripts/calo_partition_scale.py --ckpt $CK/multispecies_v2_s0/checkpoint_060000.pt \
     --real_slice $SL/multispecies_v2_h5.npz --tag ms_all
echo "=== done ==="
