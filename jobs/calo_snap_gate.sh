#!/bin/bash
#SBATCH --job-name=calo_snap_gate
#SBATCH --time=00:35:00
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --qos=gpu-test
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
export MAMBA_ROOT_PREFIX=/scratch/gpfs/IOJALVO/gnn-tracking/object_condensation/micromamba
export MAMBA_EXE=/home/lv7805/bin/micromamba
export GENPU_DATA_DIR=/scratch/gpfs/IOJALVO/lv7805/genpu_cache/datasets
export HF_HOME=/scratch/gpfs/IOJALVO/lv7805/genpu_cache
cd /home/lv7805/genpu
RUN="$MAMBA_EXE run -n genpu2 python"
SL=/scratch/gpfs/IOJALVO/lv7805/genpu_data/calo_slice
CK=/scratch/gpfs/IOJALVO/lv7805/genpu_data/checkpoints/calo_flow
FD=/scratch/gpfs/IOJALVO/lv7805/genpu_data/gate_features
mkdir -p $FD
M=$CK/multispecies_v2_s0/checkpoint_060000.pt
ALL="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16"

# ---------------------------------------------------------------------------
# FIRST GATE ON CELLS THAT ACTUALLY EXIST (PIPELINE gap #1).
#
# Until now the calo model emitted a continuous point cloud that nothing snapped,
# so generated "cells" sat at positions the detector cannot produce.
# `genpu.calo_cells` projects onto real ODD cells using constants read from the
# detector description (github.com/OpenDataDetector/OpenDataDetector):
# 16 faces, phase 11.25 deg, cell 5.1 / 30 mm, 48 / 36 layers at 5.050 / 51.0 mm.
#
# Validated on REAL cells before ever being applied to generated ones:
#   snap displacement          0.99698 of 4.29M cell-hits within 10 microns
#   round trip eta,phi,depth   position error 0.00000 mm, det recovered 0.9950,
#                              SAME cell id recovered 0.9925
#   merge within one event     collision rate 0.0004 (real cells are already
#                              distinct, so this is the id map's own error floor)
#   energy conservation        1.000000
#
# TWO ARMS, one flag apart, same checkpoint and same seed:
#   --max_cells 4096                 continuous cells (the 2026-09-15 cap-fixed reference)
#   --max_cells 4096 --snap_cells    projected + merged onto real cells
#
# WHAT TO EXPECT, recorded before the run. Merging REMOVES cells, so `n_cells`
# and `cells_per_src` must fall and per-cell energies must RISE (a merged cell
# carries the sum). Real cells are already merged by construction, so if the
# generator's co-occupancy is realistic the gate should IMPROVE; if the
# generator piles too many points into one cell it will get WORSE, and
# `multi_contrib_frac` against the real PU0 reference of ~3.5% says which.
# Depth observables are dropped under --snap_cells (the snapped depth is a layer
# index, not the continuous coordinate the real reference carries), so compare
# `event_gate_auc` and `event_gate_auc_width`, NOT the depth gate.
# ---------------------------------------------------------------------------
for ARM in plain snap; do
  EXTRA=""; [ "$ARM" = "snap" ] && EXTRA="--snap_cells"
  echo "########## arm=$ARM ##########"
  $RUN scripts/calo_metrics.py --ckpt $M --pdg_class $ALL \
       --real_slice $SL/multispecies_v2_h5.npz \
       --max_cells 4096 $EXTRA --tag ms_all_s0_$ARM \
       --dump_features $FD/ms_all_s0_$ARM.npz
done

echo "########## gate decomposition, both arms ##########"
for ARM in plain snap; do
  $RUN scripts/calo_gate_diagnose.py --features $FD/ms_all_s0_$ARM.npz \
       --seeds 3 --tag ms_all_s0_$ARM
done

echo "=== done ==="
