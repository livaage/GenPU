#!/bin/bash
#SBATCH --job-name=calo_v2_sect_build
#SBATCH --time=01:15:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
export MAMBA_ROOT_PREFIX=/scratch/gpfs/IOJALVO/gnn-tracking/object_condensation/micromamba
export MAMBA_EXE=/home/lv7805/bin/micromamba
cd /home/lv7805/genpu
RUN="$MAMBA_EXE run -n genpu2 python"
OUT=/scratch/gpfs/IOJALVO/lv7805/genpu_data/calo_slice
ALL="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16"

# Rebuild the v2 slices with the SECTION fields (job 13033802): point_depth_local, point_region,
# point_det, and a globally-unique point_layer. IN PLACE, and that is safe because the change is
# purely ADDITIVE -- verified on the smoke build (job 13033949):
#   ECAL cells: max|depth_global - depth_local| = 0.000000 mm over 775,818 cells, i.e. points_flat
#   column 2 is untouched and the ECAL front faces ARE the global ones;
#   HCAL: depth_global - depth_local is a single exact constant per detector (389.396 barrel,
#   435.000 endcap), so no cell is mis-assigned.
# Every existing checkpoint, metric and recorded number therefore stays valid against these files.
#
# Timing (job 13036294, after the hoist fix): shard 5 all classes = 1.67M showers in 61 s. The
# first attempt (13035236) stalled because the det gather sat INSIDE the per-shower loop, O(n_cells)
# per shower; the smoke build was only 40k showers so it could not see a scaling bug. Re-checked at
# full loop size before this rebuild was allowed to touch a production file.
#
# Order matters: held-out shard 5 FIRST.
# so if wall time runs short the evaluation reference is already correct.
echo "########## HELD-OUT (shard 5, all classes) ##########"
$RUN scripts/build_calo_slice_v2.py --shards 5 --pdg_class $ALL --out $OUT/multispecies_v2_h5.npz
echo "########## e+- held-out (shard 5) ##########"
$RUN scripts/build_calo_slice_v2.py --shards 5 --pdg_class 0 1 --out $OUT/electron_v2_h5.npz
echo "########## TRAIN (shards 0 1 2, all classes) ##########"
$RUN scripts/build_calo_slice_v2.py --shards 0 1 2 --pdg_class $ALL --out $OUT/multispecies_v2.npz
echo "########## e+- train (shards 0 1 2) ##########"
$RUN scripts/build_calo_slice_v2.py --shards 0 1 2 --pdg_class 0 1 --out $OUT/electron_v2.npz
echo "=== done ==="
