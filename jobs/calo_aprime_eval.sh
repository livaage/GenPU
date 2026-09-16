#!/bin/bash
#SBATCH --job-name=calo_aprime_eval
#SBATCH --time=01:30:00
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
GD=/scratch/gpfs/IOJALVO/lv7805/genpu_data/calo_geom
mkdir -p $FD $GD
ALL="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16"

# ---------------------------------------------------------------------------
# A-PRIME EVALUATION -- three arms through IDENTICAL settings (the baseline is
# re-run here rather than read from old logs, so nothing differs but the model).
#   0 baseline   multispecies_v2_s0          (per-cell MLP + EnergyHead)
#   1 cross      ms_aprime_cross_s0          (joint flow, cells see each other)
#   2 self-only  ms_aprime_selfonly_s0       (joint flow, attention on the diagonal)
#
# PRE-REGISTERED, written before any A-prime model exists:
#   collision rate      self-only ~ baseline (0.036); cross -> toward 0.0003
#   NN spacing ratio    self-only ~ 2.1;              cross -> toward 1
#   coherence (hadrons) self-only flat ~1.00;         cross -> toward real 1.5-2.2
#   copula half         cross < self-only
#   zero-suppressed     small (well under 1% of cells) on BOTH new arms. If it is
#                       large, the joint energy is broken and nothing else here
#                       can be read, whatever attention does.
# If cross ~ self-only on collisions AND spacing, attention is not what fixes
# them and the premise of this line is wrong.
#
# Also the INFERENCE-TIME measurement promised on 2026-09-16: calo_metrics
# prints gen_seconds for 1.68M showers (baseline 57.5 s); estimate for the set
# model was 2-5x.
# Coherence runs through the script's default cell cap (128) on every arm, so it
# is comparable across arms but not to the cap-free numbers elsewhere.
# ---------------------------------------------------------------------------
ARMS=("baseline:multispecies_v2_s0" "cross:ms_aprime_cross_s0" "selfonly:ms_aprime_selfonly_s0")
ARM=${ARMS[$SLURM_ARRAY_TASK_ID]}; NAME=${ARM%%:*}; M=$CK/${ARM#*:}/checkpoint_060000.pt
echo "########## arm $NAME  ($M) ##########"

echo "---------- co-occupancy ----------"
$RUN scripts/calo_cell_cooccupancy.py --ckpt $M --real_slice $SL/multispecies_v2_h5.npz \
     --n_showers 200000 --nn_showers 4000 --max_cells 4096 --out $GD/aprime_cooccup_$NAME.json

echo "---------- gate, projected cells (timed) ----------"
$RUN scripts/calo_metrics.py --ckpt $M --pdg_class $ALL --real_slice $SL/multispecies_v2_h5.npz \
     --max_cells 4096 --snap_cells --tag aprime_$NAME --dump_features $FD/aprime_$NAME.npz
$RUN scripts/calo_gate_diagnose.py --features $FD/aprime_$NAME.npz --seeds 3 --tag aprime_$NAME

echo "---------- coherence: pions and e+- (pooled model) ----------"
$RUN scripts/calo_shower_coherence.py --real_slice $SL/multispecies_v2_h5.npz --ckpt $M \
     --pdg_class 3 4 --tag aprime_${NAME}_pion
$RUN scripts/calo_shower_coherence.py --real_slice $SL/multispecies_v2_h5.npz --ckpt $M \
     --pdg_class 0 1 --tag aprime_${NAME}_ele
echo "=== done ==="
