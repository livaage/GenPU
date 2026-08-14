#!/bin/bash
#SBATCH --job-name=calo_auto_slice
#SBATCH --time=00:50:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
export MAMBA_ROOT_PREFIX=/scratch/gpfs/IOJALVO/gnn-tracking/object_condensation/micromamba
export MAMBA_EXE=/home/lv7805/bin/micromamba
export HF_HOME=/scratch/gpfs/IOJALVO/lv7805/genpu_cache
cd /home/lv7805/genpu
SL=/scratch/gpfs/IOJALVO/lv7805/genpu_data/calo_slice
RUN="$MAMBA_EXE run -n genpu2 python"

# e± slice with the AUTO anchor: truth helix below pT 0.035 GeV, straight LINE above it.
# Measured on shard 0 (120k showers) BEFORE spending any training, phi tightening per pT bin
# (median 0.015 .. 0.254 GeV):
#   helix  4.52  2.14  1.36  1.08  1.04  0.43   <- collapses, then INVERTS (anchor hurts)
#   line   1.54  1.94  2.55  2.63  2.55  2.38   <- flat, wins everywhere above ~0.04 GeV
#   auto   4.49  2.17  2.55  2.64  2.55  2.38   <- takes the better branch in every bin
# pooled 1.86x / 2.10x / 2.69x; median |core| 0.302 -> 0.098 / 0.134 / 0.069.
# That crossover IS the bremsstrahlung signature: a soft electron curls and deposits where it
# stops, a stiffer one radiates photons that fly straight from the radiation point, so its own
# curved path is the wrong predictor and gets wronger the further it bends.
# Threshold is an e± number — hadrons do not brem, so pions stay on kind=helix.
#
# Same arguments as the `electron_v1` / `electron_anchor` builds so the anchor kind is the only change.
echo "=== build electron_auto.npz ==="
$RUN scripts/build_calo_slice.py --shards 0 1 2 --pdg_class 0 1 --max_per_class 400000 \
    --core_anchor auto --out $SL/electron_auto.npz
echo "=== done ==="
