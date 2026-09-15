#!/bin/bash
#SBATCH --job-name=calo_ms_build
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
export MAMBA_ROOT_PREFIX=/scratch/gpfs/IOJALVO/gnn-tracking/object_condensation/micromamba
export MAMBA_EXE=/home/lv7805/bin/micromamba
cd /home/lv7805/genpu
OUT=/scratch/gpfs/IOJALVO/lv7805/genpu_data/calo_slice
ALL="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16"

# ALL-SPECIES v2 slices. UNCAPPED: the census (job 12930683) put three shards at ~5.0M showers /
# 124M cells, and the training path's per-POINT conditioning copy is ~84 bytes/point -> ~11 GB
# resident, which an A100 takes without trouble. v1's --max_per_class existed for a bloat that is
# now measured and affordable, and the uncapped mixture IS the target distribution a full-event
# generator has to reproduce. The per-class quota is fixed and available if this proves wrong.
echo "########## TRAIN (shards 0 1 2, all classes) ##########"
$MAMBA_EXE run -n genpu2 python scripts/build_calo_slice_v2.py \
    --shards 0 1 2 --pdg_class $ALL --out $OUT/multispecies_v2.npz
echo "########## HELD-OUT (shard 5, all classes) ##########"
$MAMBA_EXE run -n genpu2 python scripts/build_calo_slice_v2.py \
    --shards 5 --pdg_class $ALL --out $OUT/multispecies_v2_h5.npz
echo "=== done ==="
