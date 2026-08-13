#!/bin/bash
#SBATCH --job-name=verify_plots
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --qos=gpu-test
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
export MAMBA_ROOT_PREFIX=/scratch/gpfs/IOJALVO/gnn-tracking/object_condensation/micromamba
export MAMBA_EXE=/home/lv7805/bin/micromamba
export HF_HOME=/scratch/gpfs/IOJALVO/lv7805/genpu_cache
cd /home/lv7805/genpu
CK=/scratch/gpfs/IOJALVO/lv7805/genpu_data/checkpoints/tracker
SL=/scratch/gpfs/IOJALVO/lv7805/genpu_data/tracker_slice

# (1) VERIFY: run the v1 multispecies deliverable through the honest gate, same species, to confirm
#     it reproduces ~0.77 under the current setup -> makes the v3-ms 0.61 a clean like-for-like beat.
echo "############ v1 multispecies (deliverable) HONEST gate — expect ~0.77 ############"
$MAMBA_EXE run -n genpu2 python scripts/tracker_honest_gate.py \
    --ckpt $CK/multispecies_vertex_512bin/checkpoint_070000.pt --count_ckpt $CK/count_head_d0.pt \
    --slice $SL/multispecies.npz --pdg_class 0 1 3 4 7 8 11 12 --shard 0 --use_vertex

# (2) PLOTS: all-species per-step drift for v3-ms (regenerate diagnostics after the retrain).
echo "############ v3-ms per-step drift plot ############"
$MAMBA_EXE run -n genpu2 python scripts/tracker_teacherforce_test.py \
    --ckpt $CK/surface_ms_v3_ss/checkpoint_050000.pt --slice $SL/surface_multispecies.npz \
    --n_tracks 20000 --out /home/lv7805/genpu/plots/tracker/v3_ms_drift.png
echo "=== done ==="
