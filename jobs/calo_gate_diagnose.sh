#!/bin/bash
#SBATCH --job-name=calo_gate_diag
#SBATCH --time=00:50:00
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --qos=gpu-test
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
export MAMBA_ROOT_PREFIX=/scratch/gpfs/IOJALVO/gnn-tracking/object_condensation/micromamba
export MAMBA_EXE=/home/lv7805/bin/micromamba
cd /home/lv7805/genpu
RUN="$MAMBA_EXE run -n genpu2 python"
SL=/scratch/gpfs/IOJALVO/lv7805/genpu_data/calo_slice
CK=/scratch/gpfs/IOJALVO/lv7805/genpu_data/checkpoints/calo_flow
FD=/scratch/gpfs/IOJALVO/lv7805/genpu_data/gate_features
mkdir -p $FD

# WHAT THE MULTIFEATURE GATE SEES. The v2 e± model sits at composite 0.81 while no single event
# feature exceeds 0.58, so the per-feature table cannot say what to fix. Dump the PAIRED per-event
# feature matrices and decompose the composite into (a) aggregation of many small marginal offsets
# and (b) a wrong JOINT distribution -- see scripts/calo_gate_diagnose.py.
#
# BOTH SEEDS. The v2 gate seed spread is 0.003-0.016, small, but the floor_n_buckets episode
# (2026-08-24) was a one-seed conclusion that did not replicate. A structural claim about what the
# classifier uses has to hold on both checkpoints or it is not a claim.
for s in 0 1; do
  echo "########## seed $s: regenerate + dump gate features ##########"
  $RUN scripts/calo_metrics.py --ckpt $CK/electron_v2_s$s/checkpoint_060000.pt \
       --pdg_class 0 1 --real_slice $SL/electron_v2_h5.npz --tag electron_v2_s$s \
       --dump_features $FD/electron_v2_s$s.npz
  echo "########## seed $s: diagnose ##########"
  $RUN scripts/calo_gate_diagnose.py --features $FD/electron_v2_s$s.npz \
       --seeds 3 --pairs --tag electron_v2_s$s
done
echo "=== done ==="
