#!/bin/bash
#SBATCH --job-name=calo_cap_grid
#SBATCH --time=00:40:00
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
GD=/scratch/gpfs/IOJALVO/lv7805/genpu_data/calo_geom
mkdir -p $FD $GD
ALL="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16"

# ---------------------------------------------------------------------------
# PART 1 -- THE CELL-COUNT CAP A/B.
#
# `sample_showers` clamped generated cells per shower at a bare max_cells=128
# literal that no measurement ever justified. Measured 2026-09-14 on
# multispecies_v2_h5 (1,676,985 showers / 41,653,183 cells): 128 truncates 2.10%
# of showers carrying 17.7% of cells and destroys 6.87% of ALL cells, and it is
# hadron-selective -- pbar 25.3% of showers / 21.3% of cells, mu± and pi0 exactly
# 0.00%. Because `partition=True` renormalises survivors to the SAMPLED total,
# truncation does not drop that energy, it PACKS it into fewer cells, so this
# contaminates every per-cell energy feature and not just n_cells.
#
# CPU-only check already done (3000 showers, seed-matched, no retrain):
#   cap 128 : n_trunc 2.37%, mean n 24.54, total cells 73,629
#   cap 4096: n_trunc 0.00%, mean n 26.66, total cells 79,966
#   REAL    :   >128 2.43%, mean n 26.09, total cells 78,280
# So the log_n mixture's TAIL is sound and the cap was costing 5.9% of cells.
# What is unmeasured is the GATE, which is what this part is for.
#
# ARMS: ONE seed x 2 caps, all 17 classes pooled.
# One training seed is not a corner cut here -- this is a PAIRED A/B on the SAME
# checkpoint with only a sampling flag changed, so the training-seed spread
# (0.046 on the pooled mixture, the number that makes 2 seeds mandatory when
# comparing different checkpoints) cancels identically. What does NOT cancel is
# generation stochasticity: both arms draw independently, so a difference much
# smaller than the expected effect should not be over-read.
# e± is deliberately NOT run -- the cap binds only 1.0% of e± showers, so it is a
# null arm, and the pooled all-species gate is the number a full-event calo
# generator is judged on.
# 128 is the CONTROL and must reproduce the logged 0.9354 (seed 0, job 12953682);
# if it does not, the code has drifted and the treatment arm means nothing.
# ---------------------------------------------------------------------------
s=0
M=$CK/multispecies_v2_s$s/checkpoint_060000.pt
for CAP in 128 4096; do
  echo "########## seed $s  max_cells=$CAP ##########"
  $RUN scripts/calo_metrics.py --ckpt $M --pdg_class $ALL \
       --real_slice $SL/multispecies_v2_h5.npz \
       --max_cells $CAP --tag ms_all_s${s}_cap${CAP} \
       --dump_features $FD/ms_all_s${s}_cap${CAP}.npz
done

# `calo_gate_diagnose.py` is deliberately NOT run here. It reads the dumped
# feature npz, needs no GPU and no checkpoint, and runs in seconds -- keeping it
# out is what holds this job under the 1h gpu-test cap. Run afterwards:
#   for c in 128 4096; do
#     python scripts/calo_gate_diagnose.py --features $FD/ms_all_s0_cap${c}.npz \
#            --seeds 3 --pairs --tag ms_all_s0_cap${c}; done

# ---------------------------------------------------------------------------
# PART 2 -- CALO CELL GRID DERIVATION (PIPELINE.md gap #1).
#
# Verified 2026-09-14 that the cell identifier is genuinely absent, not dropped
# by us: our arrow shard, the dataset README's documented schema, the source
# parquet as served by HF datasets-server, and two other calo configs all carry
# the same 9 columns with no cell id; the HF repo holds 47,295 parquet files
# plus README and .gitattributes and no geometry file at all. EDM4HEP's
# SimCalorimeterHit has exactly four members -- cellID, energy, position,
# contributions -- and ColliderML published three of them. cellID is the one
# that was dropped in the EDM4HEP->Parquet conversion, and the converter is not
# public. The public ODD repo ships tracker XML only, no calorimeter.
#
# So the lattice must be FITTED. Established on the login node (shard 0, 300
# events, det 9 layer 0): square lattice, pitch 5.09 mm, 32 phi sectors, with
# post-fit R jumping 0.19 (24 sectors) -> 0.9937 (32) and 99.37% of real cells
# within 0.5 mm, median residual 0.12 mm. This job extends that to both ECAL
# endcaps and both HCAL endcaps at real statistics -- endcap HCAL is the open
# one, where 300 events found pitch ~29.99 mm but left the sector scan starved.
# ---------------------------------------------------------------------------
echo "########## cell grid derivation ##########"
$RUN scripts/calo_cell_grid_derive.py \
     --shards 0 1 2 --events 7000 \
     --dets 9 11 12 14 --layers 0 12 30 \
     --max_pts 30000 --r_band 200 \
     --out $GD/calo_cell_grid.json

echo "=== done ==="
