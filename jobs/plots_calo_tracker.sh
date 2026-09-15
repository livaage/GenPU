#!/bin/bash
#SBATCH --job-name=plots_calo_tracker
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
cd /home/lv7805/genpu
RUN="$MAMBA_EXE run -n genpu2 python"
CK=/scratch/gpfs/IOJALVO/lv7805/genpu_data/checkpoints
SL=/scratch/gpfs/IOJALVO/lv7805/genpu_data/calo_slice
SLT=/scratch/gpfs/IOJALVO/lv7805/genpu_data/tracker_slice

# PRESENTATION PLOTS. No model or metric change -- both scripts gained a --plots flag only.
#   calo:    real-vs-generated marginals (cells: logE, d_eta, d_phi, DEPTH; showers: N, logE,
#            width, depth) + the two-sample gate ROC. Written from calo_metrics.py because that
#            is the path that passes the helix anchor to sample_showers; eval_calo_flow.py never
#            learned about the anchor and raises on a core-anchored checkpoint, which is why the
#            existing calo histograms are stale (2026-07-09, v1 2-D era).
#   tracker: the honest full-event gate ROC for the current best v3 checkpoint (AUC 0.61).
# Metrics are re-printed as a side effect -- they should REPRODUCE the recorded numbers
# (calo gate8 0.8169 / cell_logE 0.0244; tracker 0.61). A mismatch means something drifted.

echo "########## calo: marginals + gate ROC, ele_noatom_s0 on the held-out shard ##########"
$RUN scripts/calo_metrics.py --ckpt $CK/calo_flow/ele_noatom_s0/checkpoint_060000.pt \
     --pdg_class 0 1 --real_slice $SL/electron_v2_h5.npz \
     --tag ele_noatom_s0 --plots

echo "########## tracker: honest full-event gate ROC, v3-ms all charged species ##########"
$RUN scripts/tracker_honest_gate_module.py \
     --ckpt $CK/tracker/surface_ms_v3_ss/checkpoint_050000.pt \
     --count_ckpt $CK/tracker/count_head_d0_selfnorm.pt \
     --slice $SLT/surface_multispecies.npz \
     --pdg_class 0 1 3 4 7 8 11 12 --shard 0 --plots --tag v3_ms
echo "=== done ==="
