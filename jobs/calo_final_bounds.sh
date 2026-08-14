#!/bin/bash
#SBATCH --job-name=calo_final_bounds
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

# Candidate shippable config: the BASELINE architecture (unconditional energy head — the 2x2
# showed conditioning it on the sampled global costs more marginal fidelity than it buys) with
# both physical bounds at generation:
#   --logE_max  : per-cell clamp at the training slice's hardest cell
#   E_reco<=E_true : energy conservation (default ON), using truth conditioning
echo "=== photon F: baseline + cell clamp + energy conservation ==="
$RUN scripts/calo_metrics.py --ckpt $CK/photon_qtd_v1/checkpoint_040000.pt \
    --pdg_class 2 --shard 5 --tag photon_F_bounded --no_partition --logE_max -2.5456
echo "=== pion F: baseline + cell clamp + energy conservation ==="
$RUN scripts/calo_metrics.py --ckpt $CK/pion_qtd_v2/checkpoint_040000.pt \
    --pdg_class 3 4 --shard 5 --tag pion_F_bounded --no_partition --logE_max -1.1328
echo "=== done ==="
