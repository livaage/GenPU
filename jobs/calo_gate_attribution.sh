#!/bin/bash
#SBATCH --job-name=calo_gate_attr
#SBATCH --time=00:40:00
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --qos=gpu-test
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
export MAMBA_ROOT_PREFIX=/scratch/gpfs/IOJALVO/gnn-tracking/object_condensation/micromamba
export MAMBA_EXE=/home/lv7805/bin/micromamba
export HF_HOME=/scratch/gpfs/IOJALVO/lv7805/genpu_cache
cd /home/lv7805/genpu
CK=/scratch/gpfs/IOJALVO/lv7805/genpu_data/checkpoints/calo_flow
RUN="$MAMBA_EXE run -n genpu2 python"

# WHICH of the 8 event-gate features carries the pion's 0.809? Clamping the energy tail fixed
# resp_max by 5 orders of magnitude and moved the gate by 0.000, so the gate is NOT the energy
# tail — and the per-observable Wasserstein table has every pion marginal looking healthy
# (cell_logE 0.020, cells_per_shower 0.013) except logEreco 0.089. Single-variable AUC per
# feature localises it before the next architecture attempt.
echo "=== pion, per-feature gate attribution ==="
$RUN scripts/calo_metrics.py --ckpt $CK/pion_qtd_v2/checkpoint_040000.pt \
    --pdg_class 3 4 --shard 5 --tag pion_G_attrib --no_partition --logE_max -1.1328
echo "=== photon, per-feature gate attribution (the 0.557 reference) ==="
$RUN scripts/calo_metrics.py --ckpt $CK/photon_qtd_v1/checkpoint_040000.pt \
    --pdg_class 2 --shard 5 --tag photon_G_attrib --no_partition --logE_max -2.5456
echo "=== done ==="
