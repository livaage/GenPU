#!/bin/bash
#SBATCH --job-name=calo_q3_clean
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
export MAMBA_ROOT_PREFIX=/scratch/gpfs/IOJALVO/gnn-tracking/object_condensation/micromamba
export MAMBA_EXE=/home/lv7805/bin/micromamba
export HF_HOME=/scratch/gpfs/IOJALVO/lv7805/genpu_cache
cd /home/lv7805/genpu
# Q3 CLEAN — does the real track endpoint beat the helix anchor for the population that actually
# HAS tracks? Q3 has never been measured on the clean set: 2026-08-16 pooled the contaminated
# turning branch, and this morning's corrected re-run (12871830) fixed the r-ordering bug but still
# pooled. This restricts to depositors born OUTSIDE the calo front face.
for spec in "3 4:pion_clean" "0 1:electron_clean"; do
    cls="${spec%%:*}"; name="${spec##*:}"
    echo "=== Q3 CLEAN (born-outside only): $name (pdg $cls) ==="
    $MAMBA_EXE run -n genpu2 python scripts/calo_curler_track_probe.py \
        --pdg_class $cls --shard 5 --tag "$name" --born_outside_only
done
echo "=== done ==="
