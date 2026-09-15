#!/bin/bash
#SBATCH --job-name=calo_ms_eval_core
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
ALL="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16"

# The two evals that carry the argument, plus the joint-structure diagnostic on each.
#   ms_all  -- every class pooled: the number a full-event calo generator is actually judged on,
#              and the first time it has been measurable on re-attributed 3D showers.
#   ms_ele  -- classes 0,1 against the SAME held-out population the dedicated e± model was scored
#              on, so the difference from 0.8105 / 0.8074 is the cost of pooling and nothing else.
echo "########## seed $s: ALL CLASSES POOLED ##########"
$RUN scripts/calo_metrics.py --ckpt $M --pdg_class $ALL \
     --real_slice $SL/multispecies_v2_h5.npz --tag ms_all_s$s \
     --dump_features $FD/ms_all_s$s.npz
$RUN scripts/calo_gate_diagnose.py --features $FD/ms_all_s$s.npz --seeds 3 --pairs --tag ms_all_s$s

echo "########## seed $s: e+- (A/B vs the dedicated electron_v2 model) ##########"
$RUN scripts/calo_metrics.py --ckpt $M --pdg_class 0 1 \
     --real_slice $SL/multispecies_v2_h5.npz --tag ms_ele_s$s \
     --dump_features $FD/ms_ele_s$s.npz
$RUN scripts/calo_gate_diagnose.py --features $FD/ms_ele_s$s.npz --seeds 3 --pairs --tag ms_ele_s$s
echo "=== done ==="
