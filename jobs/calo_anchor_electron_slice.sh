#!/bin/bash
#SBATCH --job-name=calo_anchor_e_slice
#SBATCH --time=01:30:00
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

# PHASE 1 part A (CPU only), second half: the e± slice, plus the Phase 0b probe re-run on all four
# species now that the endcap sweep sign is fixed (see jobs/calo_anchor_pion_slice.sh for the
# measurement that found the bug). The probe numbers in STATUS.md (2.9x photon / 11.2x e± / 4.6x
# pion / 25.5x proton) were all measured through the wrong sign and need replacing.

echo "=== Phase 0b re-run with the corrected endcap sign (all four species) ==="
for spec in "2:photon" "0 1:electron" "3 4:pion" "7 8:proton"; do
    cls="${spec%%:*}"; name="${spec##*:}"
    $RUN scripts/calo_helix_core_probe.py --pdg_class $cls --tag "${name}_fixed" --max_showers 400000
done

# IDENTICAL arguments to the `electron_v1` reference build (800k showers, 400k per class)
echo "=== build electron_anchor.npz ==="
$RUN scripts/build_calo_slice.py --shards 0 1 2 --pdg_class 0 1 --max_per_class 400000 \
    --core_anchor helix --out $SL/electron_anchor.npz
echo "=== done ==="
