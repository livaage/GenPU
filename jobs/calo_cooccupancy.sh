#!/bin/bash
#SBATCH --job-name=calo_cooccup
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
GD=/scratch/gpfs/IOJALVO/lv7805/genpu_data/calo_geom
mkdir -p $GD

# ---------------------------------------------------------------------------
# WHERE does the generator put two deposits in one readout channel?
#
# Job 13929818: generated collision rate 0.0361 against a real floor of 0.0003
# measured through the identical snap path on 200k showers / 5.0M rows. 120x.
# The generator OVER-CONCENTRATES -- an earlier note in this repo said
# "under-merges", which was backwards and rested on the 7.2% within-shower
# CONTRIBUTION sharing, a population the v2 slice already merges away.
#
# HYPOTHESIS: the excess sits in the shower CORE, which is where cells are
# densest, and which `PointCFM` controls directly -- making it testable by
# resampling rather than retraining.
#
# PRE-REGISTERED READING of the output profile:
#   rising sharply toward r/width = 0   -> the CORE is too dense. Next step is a
#                                          resampling test, no retrain.
#   flat in r/width                     -> a GLOBAL density error, i.e. the whole
#                                          cloud is too tight. Would be surprising:
#                                          shower_width W/sigma is only 0.0505, so
#                                          a pure width error is not indicated.
#   rising at LARGE r/width             -> not anticipated; would point at the
#                                          fringe, where cells are sparsest and
#                                          collisions should be rarest.
#
# The script also prints a PHYSICAL SANITY CHECK first (endcap cells inside
# r = 315 mm must be ~0). That check exists because a probe that re-standardised
# an already-raw `cont` inflated |eta| to a median of 4.5 and put 83% of cells
# inside the beam pipe before anything caught it. If that line is not ~0, stop
# and ignore every number below it.
# ---------------------------------------------------------------------------
$RUN scripts/calo_cell_cooccupancy.py \
     --ckpt $CK/multispecies_v2_s0/checkpoint_060000.pt \
     --real_slice $SL/multispecies_v2_h5.npz \
     --n_showers 200000 --nn_showers 4000 --max_cells 4096 \
     --out $GD/calo_cooccupancy.json

echo "=== done ==="
