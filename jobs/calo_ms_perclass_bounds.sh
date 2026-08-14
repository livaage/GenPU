#!/bin/bash
#SBATCH --job-name=calo_ms_perclass
#SBATCH --time=00:50:00
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
SL=/scratch/gpfs/IOJALVO/lv7805/genpu_data/calo_slice
RUN="$MAMBA_EXE run -n genpu2 python"
CKPT=$CK/multispecies_v1/checkpoint_060000.pt

# The first multi-species eval clamped every species at the HADRONIC cell max (-1.13), which is
# 4x too loose for EM showers (photon slice max -2.55). Re-evaluate with per-species clamps fit
# from the multispecies slice, to separate "one shared head costs fidelity" from "the bound was
# wrong for this species". gamma is the sensitive case (0.681 here vs 0.557 single-species).
for spec in "2:gamma" "3 4:pi" "0:electron" "7:proton"; do
    cls="${spec%%:*}"; name="${spec##*:}"
    echo "=== ms per-class bound: $name (pdg $cls) ==="
    $RUN scripts/calo_metrics.py --ckpt $CKPT --pdg_class $cls --shard 5 \
        --tag "msb_$name" --no_partition --logE_max_slice $SL/multispecies_v1.npz
done
echo "=== done ==="
