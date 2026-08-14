#!/bin/bash
#SBATCH --job-name=calo_anchor_pion_slice
#SBATCH --time=01:40:00
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

# PHASE 1 part A (CPU only): rebuild the pion training slice with the shower core stored as a
# RESIDUAL from the truth-helix extrapolation to the calo front face.
#
# BUG FIXED 2026-08-14 (helix.py:helix_at_z): the endcap sweep direction was -sign*sign(q), i.e.
# backwards vs helix_at_r / layer_references. Measured against 80k real endcap tracker hits
# (|z|>1200mm, outermost hit per track): median |dphi| to the true hit 0.633 rad with the old sign,
# 0.016 rad with the fixed one (|dr| 33mm -> 8.9mm). The endcaps carry 83% of deposited calo energy,
# so every Phase 0b number was measured through this. Pion phi tightening on a 50k-shower re-run:
# 4.6x -> 16.5x.
#
# THIRD ANCHOR BRANCH: 63% of pion / 72% of e± showers come from soft particles (median pT 0.27 GeV,
# born at vr ~ 420mm) that curl back before r_calo and reach NO face at all. Those now anchor at the
# helix TURNING POINT (outermost radius the circle reaches) instead of at the particle direction:
# measured sigma(core_phi) 2.4x tighter (pion, proton) / 2.0x (e±), sigma(core_eta) 4.6x.
#
# Arguments are IDENTICAL to the `pion_ctr_v2` reference build (2.52M showers) so the only change
# is the core frame.
echo "=== build pion_anchor.npz ==="
$RUN scripts/build_calo_slice.py --shards 0 1 2 --pdg_class 3 4 \
    --core_anchor helix --out $SL/pion_anchor.npz
echo "=== done ==="
