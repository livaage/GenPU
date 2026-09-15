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
# RESULT (job 13931553): collisions ARE a core effect -- 0.1168 at r/width < 0.25 falling
# monotonically to 0.0001 beyond 3 widths, a 1000x gradient with no flat component and nothing at
# large radius. Both alternative branches are excluded.
#
# BUT the pre-registered reading ("rising toward r=0 -> the core is too dense") was TOO STRONG, and
# missed a third possibility. `PointCFM.sample` draws every cell from its own randn and integrates a
# field that sees only that cell, so a shower's cells are i.i.d. given the conditioning -- and i.i.d.
# draws collide EVEN AT EXACTLY THE RIGHT DENSITY (birthday argument). A real shower is a SET of
# distinct channels, a draw without replacement, and cannot collide at all. Both "core too dense"
# and "core density right, sampler wrong" predict a monotone rise toward the core.
#
# THIS RUN ADDS THE DISCRIMINATOR: within-shower nearest-neighbour SPACING, generated vs real.
#   ratio ~1   -> density is RIGHT; the collisions are the i.i.d. sampler, and this becomes a second
#                 and MECHANICAL argument for the set/attention head, independent of the coherence
#                 result. Tuning core density would then be fitting a symptom.
#   ratio << 1 -> the cloud really is too tight; a density/shape problem after all.
# Real reference already measured: median 0.00314, p10 0.00128, p90 0.03707.
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
