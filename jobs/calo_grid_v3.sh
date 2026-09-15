#!/bin/bash
#SBATCH --job-name=calo_grid_v3
#SBATCH --time=00:45:00
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
GD=/scratch/gpfs/IOJALVO/lv7805/genpu_data/calo_geom
mkdir -p $GD

# ---------------------------------------------------------------------------
# CELL GRID v3 -- refined pitch, radial banding OFF, angle pattern reported.
#
# v2 (job 13921623) reported ECAL on-grid 0.999 at median residual 0.08 mm and
# needed `--r_band 200` to get there. Both facts were symptoms of ONE bug:
# `pitch_from_nn` returns a histogram MODE and so quantises to the bin width,
# giving 5.0900 where the truth is 5.1 mm EXACTLY. A 0.0100 mm error slips a
# full pitch every ~510 cells, so the radial bands were not describing structure,
# they were keeping the accumulated slip under half a pitch.
#
# `refine_pitch` now takes the mode as a SEED and maximises circular
# concentration over local patches. Verified on the login node, det 9 layer 0,
# with `--r_band 1e9` (banding OFF):
#   pitch 5.0900 -> 5.1000;  sectors=32 -> on-grid 1.0000, median residual
#   0.0014 mm (v2: 0.999 / 0.080 mm WITH banding).
# The transverse axis is now as exact as the longitudinal one (which cleared
# 0.0000), and `--r_band` should be irrelevant. It is set OFF here on purpose:
# if 1.0000 does NOT survive without it, the pitch story is incomplete.
#
# NEW DIAGNOSTIC, and it may already have answered the HCAL question. `refine_pitch` now reports
# the INTER-WEDGE SPREAD of the pitch estimate. Login-node check, det 9 vs det 12, layer 0:
#   ECAL  5.09998 mm, IQR 0.00000 mm over 24 wedges  -> STABLE   -> on-grid 1.0000
#   HCAL 30.01894 mm, IQR 0.23407 mm over 24 wedges  -> UNSTABLE -> on-grid 0.7545
# Every ECAL wedge independently finds the same constant to 5 dp. HCAL's wedges disagree by 0.8%,
# and the HCAL estimate MOVES with the partition (29.993 per-sector on 12k events, 30.019 over 24
# wedges, 30.131 with a radial subdivision), with on-grid never above ~0.8 for any of them.
# A pitch that depends on how the data is sliced means there is NO SINGLE PITCH -- a result about
# the detector, not a tuning failure. Refinement is therefore kept ON for both: it is decisive for
# ECAL, and for HCAL its instability is the measurement.
#
# THE HCAL QUESTION, which is what this run is really for.
# HCAL's pitch was never wrong (mode 29.9900 vs refined 29.9930) and projective
# towers are FALSIFIED (pitch identical in 6 radial bands, r 362-3047 mm). What
# is wrong is ORIENTATION: fitting each sector's angle gave 0 / 22.5 / 45 deg,
# with 45 on sectors 4,12,20 (= 4 mod 8) and 0 on 8,16 (= 0 mod 8) -- suggesting
# 8-fold with ALTERNATING orientations, of which the accepted 32 is an alias.
# The script now prints angle clusters and angle-by-(sector mod 8) so that is
# decidable straight from the log.
#
# NOTE the same printout on ECAL is already informative and was NOT expected:
# ECAL angles are not uniform either -- clusters at -45x4, -22.5x8, 0x8,
# +22.5x8, +45x4, i.e. a period-8 palindrome in sector index. ECAL still reaches
# on-grid 1.0000, so whatever that pattern is, the model absorbs it; on HCAL it
# does not. Comparing the two patterns is the point.
# ---------------------------------------------------------------------------
$RUN scripts/calo_cell_grid_derive.py \
     --shards 0 1 2 3 4 --events 7000 \
     --dets 9 11 12 14 --layers 0 4 12 20 30 40 \
     --max_pts 30000 --n_annuli 3 --r_band 1e9 \
     --out $GD/calo_cell_grid_v3.json

echo "=== done ==="
