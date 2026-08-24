#!/bin/bash
#SBATCH --job-name=calo_curler_probe
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
export MAMBA_ROOT_PREFIX=/scratch/gpfs/IOJALVO/gnn-tracking/object_condensation/micromamba
export MAMBA_EXE=/home/lv7805/bin/micromamba
export HF_HOME=/scratch/gpfs/IOJALVO/lv7805/genpu_cache
cd /home/lv7805/genpu
RUN="$MAMBA_EXE run -n genpu2 python"

# CPU-only measurement on REAL data — no model, no training, no GPU.
#
# 63% of pion / 72% of e± calo showers are anchored at the vacuum-helix TURNING POINT: soft
# secondaries (median pT 0.27 GeV, born at vr ~ 420 mm) that curl back before the calo face, with
# their cells booked to the parent. That anchor is worth ~2.5x vs 15-24x on the face branch, and
# it is a proxy — the vacuum helix knows nothing about dE/dx, scattering or nuclear interaction.
#
# Before committing to a track-endpoint anchor (Phase 4), settle three things on real data:
#   Q1 do these particles even leave tracker hits? (0 hits -> no track edge at all)
#   Q2 how far is the vacuum turning point from where the track actually got?
#   Q3 THE DECIDER: does the real track endpoint predict the shower core better than the helix
#      anchor does, per branch? If not, the turning-point proxy is as good as it gets and the
#      track edge is not worth building.
#
# NOTE: preprocessed tracker hits are sorted r-ASCENDING within a particle, so the last stored hit
# is the OUTERMOST, not the chronologically final one. For the curler branch that is the right
# quantity (the empirical analogue of the turning point) but the two must not be confused.

for spec in "3 4:pion" "0 1:electron"; do
    cls="${spec%%:*}"; name="${spec##*:}"
    echo "=== curler/track probe: $name (pdg $cls) ==="
    $RUN scripts/calo_curler_track_probe.py --pdg_class $cls --shard 5 --tag "$name"
done
echo "=== done ==="
