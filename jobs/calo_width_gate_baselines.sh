#!/bin/bash
#SBATCH --job-name=calo_width_gate
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
SL=/scratch/gpfs/IOJALVO/lv7805/genpu_data/calo_slice
RUN="$MAMBA_EXE run -n genpu2 python"

# Re-baseline the reference points under the WIDTH-AUGMENTED gate (10 features: the 8 historical
# energy/multiplicity ones + per-shower width mean/std). Both AUCs are reported per run, so the
# 8-feature series stays comparable while we learn how much the blind spot was hiding. The
# multi-species e± runs here are the control for the capacity test (jobs/calo_electron_capacity.sh).
echo "=== photon single-species (the 0.557 reference) ==="
$RUN scripts/calo_metrics.py --ckpt $CK/photon_qtd_v1/checkpoint_040000.pt \
    --pdg_class 2 --shard 5 --tag w_photon --no_partition --logE_max -2.5456
echo "=== pion single-species (the 0.809 reference) ==="
$RUN scripts/calo_metrics.py --ckpt $CK/pion_qtd_v2/checkpoint_040000.pt \
    --pdg_class 3 4 --shard 5 --tag w_pion --no_partition --logE_max -1.1328
for spec in "0:electron" "1:positron"; do
    cls="${spec%%:*}"; name="${spec##*:}"
    echo "=== multispecies control: $name (pdg $cls) ==="
    $RUN scripts/calo_metrics.py --ckpt $CK/multispecies_v1/checkpoint_060000.pt \
        --pdg_class $cls --shard 5 --tag "w_ms_$name" --no_partition \
        --logE_max_slice $SL/multispecies_v1.npz
done
echo "=== done ==="
