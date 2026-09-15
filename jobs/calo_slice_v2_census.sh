#!/bin/bash
#SBATCH --job-name=calo_v2_census
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
export MAMBA_ROOT_PREFIX=/scratch/gpfs/IOJALVO/gnn-tracking/object_condensation/micromamba
export MAMBA_EXE=/home/lv7805/bin/micromamba
cd /home/lv7805/genpu
OUT=/scratch/gpfs/IOJALVO/lv7805/genpu_data/calo_slice

# ALL-SPECIES CENSUS, one shard, no cap. The 2026-08-13 species census counted DEPOSITORS; under v2
# every cell is booked to its calo-INCIDENT ancestor, so the population is a different one and the
# per-class shower counts and cells/shower are unknown. This measures them before the production
# build, because they set --max_per_class: the training path materialises a per-POINT copy of the
# conditioning on the GPU, so total cells, not showers, is the binding constraint.
$MAMBA_EXE run -n genpu2 python scripts/build_calo_slice_v2.py \
    --shards 0 --pdg_class 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 \
    --out $OUT/multispecies_v2_census.npz
echo "=== done ==="
