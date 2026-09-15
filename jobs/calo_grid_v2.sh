#!/bin/bash
#SBATCH --job-name=calo_grid_diag
#SBATCH --time=00:50:00
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
GD=/scratch/gpfs/IOJALVO/lv7805/genpu_data/calo_geom
mkdir -p $GD

# ---------------------------------------------------------------------------
# CELL GRID DERIVATION, re-run with the pitch estimator FIXED.
#
# Job 13920087 resolved 32 phi sectors on all four endcap detectors -- ECAL
# (det 9/11) at pitch 5.09 mm, on-grid 0.9945/0.9937; HCAL (det 12/14) at pitch
# 29.99 mm, on-grid 0.90-0.94 -- but LOST both ECAL layers 0 and 12, which
# returned pitch 7.21 mm (= 5.09*sqrt2, the lattice DIAGONAL) and then failed
# every sector fit built on that wrong constant.
#
# Cause was `--max_pts` random subsampling, not the data. Deleting lattice points
# at random deletes each cell's true nearest neighbours, so the NN mode climbs
# the harmonics. Measured on det 9 layer 0 (12,485 distinct, true pitch 5.09):
#
#   retained        1.00    0.50    0.25    0.17    0.10
#   random (old)    5.09    7.21   11.41   11.41   11.41
#   annuli (new)    5.09    5.09    5.09    5.09    5.09
#
# The job kept 17% at layer 0. Layer 30 survived at 21% by luck of density, so
# its 0.9945 is right but was not safely obtained -- this re-run re-derives it.
#
# `annulus_subsample` keeps whole RADIAL annuli: neighbourhoods stay intact
# except at two edges. Annuli and not phi wedges on purpose -- wedge boundaries
# would impose an angular period, and the angular period is exactly what the
# sector scan is supposed to MEASURE, so wedges would be circular.
#
# Changes from 13920087: annulus subsampling; 5 shards not 3 (det 12/14 layer 30
# failed on genuine sparsity -- 4,234 distinct positions); more layers, to test
# whether pitch and sector count are constant with depth or whether ECAL changes
# granularity (still open -- 13920087 only ever fitted one ECAL layer cleanly).
# Layers beyond a detector's count are skipped by the script (HCAL has 36).
# ---------------------------------------------------------------------------
$RUN scripts/calo_cell_grid_derive.py \
     --shards 0 1 2 3 4 --events 7000 \
     --dets 9 11 12 14 --layers 0 4 12 20 30 40 \
     --max_pts 30000 --n_annuli 3 --r_band 200 \
     --out $GD/calo_cell_grid_v2.json

# ---------------------------------------------------------------------------
# GATE DECOMPOSITION for the cap A/B (job 13920087, features already on disk).
#
# Belongs here rather than on the login node: two attempts there died, the second
# OOM-killed (exit 137) by the per-user cgroup under a load average of 25 with 43
# users. The INPUT is 7,384 x 12 floats, so this is not a real memory need -- but
# guessing at a shared node's limits is exactly the thing not to do, and in a job
# it also sees the GPU (`dev = cuda if available`) and runs in seconds.
#
# THE TEST. Part 1 found five marginals collapsing toward chance -- logE_mean
# 0.629->0.524, logE_p90 0.710->0.610, cells_per_src 0.570->0.512,
# frac_near_floor 0.561->0.508, logE_std 0.755->0.707 -- while the composite gate
# did NOT move (0.9328 -> 0.9356). The reading is that the marginal half was
# never the binding constraint. That predicts, between the two arms:
#   marginals_only  DROPS sharply
#   copula_only     stays FLAT
# If instead copula_only moves too, the reading is wrong and the cap was
# entangled with the joint structure after all.
# ---------------------------------------------------------------------------
FD=/scratch/gpfs/IOJALVO/lv7805/genpu_data/gate_features
for c in 128 4096; do
  echo "########## gate decomposition, cap $c ##########"
  $RUN scripts/calo_gate_diagnose.py --features $FD/ms_all_s0_cap${c}.npz \
       --seeds 3 --pairs --tag ms_all_s0_cap${c}
done

echo "=== done ==="
