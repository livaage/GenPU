#!/bin/bash
#SBATCH --job-name=calo_aprime_nosnap
#SBATCH --time=00:55:00
#SBATCH --qos=gpu-test
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --array=0-2
#SBATCH --output=logs/%x-%A_%a.out
set -uo pipefail
export MAMBA_ROOT_PREFIX=/scratch/gpfs/IOJALVO/gnn-tracking/object_condensation/micromamba
export MAMBA_EXE=/home/lv7805/bin/micromamba
export GENPU_DATA_DIR=/scratch/gpfs/IOJALVO/lv7805/genpu_cache/datasets
cd /home/lv7805/genpu
RUN="$MAMBA_EXE run -n genpu2 python"
SL=/scratch/gpfs/IOJALVO/lv7805/genpu_data/calo_slice
CK=/scratch/gpfs/IOJALVO/lv7805/genpu_data/checkpoints/calo_flow
FD=/scratch/gpfs/IOJALVO/lv7805/genpu_data/gate_features
mkdir -p $FD
ALL="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16"

# ---------------------------------------------------------------------------
# A-PRIME COLLISION DIAGNOSTIC -- the calo_aprime_eval.sh gate with --snap_cells
# REMOVED, nothing else changed. Existing 60k checkpoints, no training.
#
# Question: cross collides 0.130 (baseline 0.036, self-only 0.027) and merges
# 13.1% of its points under the snap. Its gate losses are all marginal and all
# in the direction merging pushes (logE_p90 0.844, logE_mean 0.838,
# cells_per_src 0.695: fewer cells, more energy each). Is the merge the cause?
#
# PRE-REGISTERED (written before the run):
#   merge IS the cause  -> without snap, cross's logE_p90 / logE_mean /
#                          cells_per_src AUCs fall to roughly self-only's
#                          (<= ~0.72 / ~0.71 / ~0.55), and cross's composite
#                          gate falls BELOW self-only's no-snap gate.
#   merge is NOT        -> those three stay >= 0.8 / 0.8 / 0.65 unsnapped: the
#                          energy marginal is wrong upstream of the projection.
#   Self-only and baseline merge 2.6% / 3.6%, so they should move little.
# Compare arms only with EACH OTHER: without the snap the gate carries depth
# features (12, not 10), so these composites are not comparable to the
# --snap_cells numbers. Per-feature AUCs are comparable.
# This is a DIAGNOSTIC. Merging is the correct response to two deposits in one
# channel (STATUS 2026-09-15); if confirmed, the fix is exclusion upstream.
# ---------------------------------------------------------------------------
ARMS=("baseline:multispecies_v2_s0" "cross:ms_aprime_cross_s0" "selfonly:ms_aprime_selfonly_s0")
ARM=${ARMS[$SLURM_ARRAY_TASK_ID]}; NAME=${ARM%%:*}; M=$CK/${ARM#*:}/checkpoint_060000.pt
echo "########## arm $NAME  ($M)  NO SNAP ##########"

$RUN scripts/calo_metrics.py --ckpt $M --pdg_class $ALL --real_slice $SL/multispecies_v2_h5.npz \
     --max_cells 4096 --tag aprime_nosnap_$NAME --dump_features $FD/aprime_nosnap_$NAME.npz
$RUN scripts/calo_gate_diagnose.py --features $FD/aprime_nosnap_$NAME.npz --seeds 3 --tag aprime_nosnap_$NAME
echo "=== done ==="
