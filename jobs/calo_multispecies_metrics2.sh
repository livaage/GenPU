#!/bin/bash
#SBATCH --job-name=calo_ms_metrics2
#SBATCH --time=00:55:00
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --qos=gpu-test
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
export MAMBA_ROOT_PREFIX=/scratch/gpfs/IOJALVO/gnn-tracking/object_condensation/micromamba
export MAMBA_EXE=/home/lv7805/bin/micromamba
export HF_HOME=/scratch/gpfs/IOJALVO/lv7805/genpu_cache
cd /home/lv7805/genpu
CK=/scratch/gpfs/IOJALVO/lv7805/genpu_data/checkpoints/calo_flow
RUN="$MAMBA_EXE run -n genpu2 python"
CKPT=$CK/multispecies_v1/checkpoint_060000.pt

# The species that actually carry the calorimeter: e- and e+ are 48% of deposited energy and
# 57% of cells, p another 13% of energy. Plus the pooled tail (K, n, mu, pi0, ... ~8%) so the
# "one head covers everything" claim is measured, not assumed.
for spec in "0:electron" "1:positron" "7:proton" "5 6 8 9 10 11 12 13 14 15 16:rest"; do
    cls="${spec%%:*}"; name="${spec##*:}"
    echo "=== ms metrics: $name (pdg $cls) ==="
    $RUN scripts/calo_metrics.py --ckpt $CKPT --pdg_class $cls --shard 5 \
        --tag "ms_$name" --no_partition --logE_max -1.1328
done
echo "=== done ==="
