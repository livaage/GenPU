#!/bin/bash
#SBATCH --job-name=calo_clamp_only
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

# The 2x2 showed BOTH architecture changes degrade the cell-log-E marginal (which the gate is
# mostly made of), because the pre-audit energy head is unconditional and therefore fits that
# marginal almost exactly. The genuine defect it leaves is the unbounded mixture tail (photon
# resp_max 240x E_true, pion 6.6e5x). So: keep the baseline checkpoints and only BOUND the tail
# at the training slice's hardest cell — no retraining, and it touches ~0.004% of cells.
#   photon slice max log-E -2.5456 (0.0784 GeV);  pion slice max -1.1328 (0.3221 GeV)
echo "=== photon E: baseline ckpt, sum-of-cells, clamped ==="
$RUN scripts/calo_metrics.py --ckpt $CK/photon_qtd_v1/checkpoint_040000.pt \
    --pdg_class 2 --shard 5 --tag photon_E_base_clamp --no_partition --logE_max -2.5456
echo "=== pion E: baseline ckpt, sum-of-cells, clamped ==="
$RUN scripts/calo_metrics.py --ckpt $CK/pion_qtd_v2/checkpoint_040000.pt \
    --pdg_class 3 4 --shard 5 --tag pion_E_base_clamp --no_partition --logE_max -1.1328
echo "=== done ==="
