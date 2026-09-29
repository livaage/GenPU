#!/bin/bash
#SBATCH --job-name=calo_aprime_resolve
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
ALL="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16"

# ---------------------------------------------------------------------------
# A-PRIME COLLISION RESOLVER -- is the unmerged gate REACHABLE once collisions are gone?
# calo_aprime_eval.sh's merged gate + --resolve_collisions (calo_cells.resolve_collisions):
# in each (cell, shower) the highest-energy point keeps the cell; the others move to the
# nearest FREE in-plane neighbour (same layer, <= 2 pitches); no free neighbour -> merge as
# before. Existing 60k checkpoints, no training. A DIAGNOSTIC, not a generator component.
#
# Also scores the 2026-09-29 unmerged features with depth DROPPED (aprime_nosnap10_*), so
# resolved-merged and unmerged are compared on the SAME 10 features.
#
# PRE-REGISTERED (written before the run):
#   reachable     -> cross resolved: logE_p90 / logE_mean / cells_per_src AUCs <= ~0.60
#                    (unmerged 0.525 / 0.576 / 0.507), and its gate within ~0.05 of its own
#                    10-feature unmerged number and BELOW self-only resolved.
#   not reachable -> cross resolved stays >= ~0.90: moving cells costs as much as merging them,
#                    i.e. the unmerged 0.787 was partly an artifact of scoring points as cells.
#   Side effect to watch: moving cells outward can WIDEN showers -> width_mean / width_std.
#   Self-only and baseline collide 2.7% / 3.6% and should move little.
# ---------------------------------------------------------------------------
ARMS=("baseline:multispecies_v2_s0" "cross:ms_aprime_cross_s0" "selfonly:ms_aprime_selfonly_s0")
ARM=${ARMS[$SLURM_ARRAY_TASK_ID]}; NAME=${ARM%%:*}; M=$CK/${ARM#*:}/checkpoint_060000.pt
echo "########## arm $NAME  ($M)  SNAP + RESOLVE ##########"

$RUN scripts/calo_metrics.py --ckpt $M --pdg_class $ALL --real_slice $SL/multispecies_v2_h5.npz \
     --max_cells 4096 --snap_cells --resolve_collisions --tag aprime_resolve_$NAME \
     --dump_features $FD/aprime_resolve_$NAME.npz
$RUN scripts/calo_gate_diagnose.py --features $FD/aprime_resolve_$NAME.npz --seeds 3 --tag aprime_resolve_$NAME

echo "---------- reference: 2026-09-29 unmerged features, depth dropped (10 features) ----------"
$RUN scripts/calo_gate_diagnose.py --features $FD/aprime_nosnap10_$NAME.npz --seeds 3 --tag aprime_nosnap10_$NAME
echo "=== done ==="
