#!/bin/bash
#SBATCH --job-name=calo_curler_fixr
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
export MAMBA_ROOT_PREFIX=/scratch/gpfs/IOJALVO/gnn-tracking/object_condensation/micromamba
export MAMBA_EXE=/home/lv7805/bin/micromamba
export HF_HOME=/scratch/gpfs/IOJALVO/lv7805/genpu_cache
cd /home/lv7805/genpu
RUN="$MAMBA_EXE run -n genpu2 python"

# J1 (2026-08-24) — RE-RUN of the Phase 4 curler/track probe with the ORDERING BUG FIXED.
#
# The 2026-08-16 run took the LAST STORED hit as "the outermost hit", on the CLAUDE.md claim that
# stage2 tracker hits are r-ascending. They are NOT: measured Spearman(index, r) = 0.01, ~50% of
# steps decrease in r, median max r-drop 158 mm. The r-sort lives in build_tracker_slice.py:85,
# NOT in preprocessing. So Q2/Q3 were computed on an essentially RANDOM hit.
#   Q1 (coverage = hit COUNTS) is unaffected -> the Phase 4 headline (81% pion / 93% e± curlers
#      with zero tracker hits) STANDS and is not being re-litigated here.
#   Q2/Q3 are invalid and are what this re-run replaces.
#
# SPECIFIC THING TO WATCH: STATUS records "the vacuum turning point's phi is UNCORRELATED with the
# real outermost hit's phi -- |dphi| median 1.567 rad ~ pi/2, the uniform value". pi/2 is exactly
# what picking a RANDOM hit would produce, so that finding is the prime suspect for being a bug
# artifact. If the corrected |dphi| comes in well below pi/2, the turning-point anchor carries more
# information than we credited it with, and the Phase 4 "surviving scope" note needs rewriting.
#
# New tags so the old JSONs are preserved side by side for comparison.

for spec in "3 4:pion_fixr" "0 1:electron_fixr"; do
    cls="${spec%%:*}"; name="${spec##*:}"
    echo "=== curler/track probe (FIXED argmax-r): $name (pdg $cls) ==="
    $RUN scripts/calo_curler_track_probe.py --pdg_class $cls --shard 5 --tag "$name"
done
echo "=== done ==="
