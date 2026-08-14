#!/bin/bash
#SBATCH --job-name=calo_ms_train
#SBATCH --time=00:55:00
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=gpu80
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

# ONE PDG-conditioned calo head over all 17 classes (multispecies_v1: 2.65M showers, 400k/class
# cap). Baseline architecture — the energy head stays unconditional (the 2x2 on 2026-08-13 showed
# conditioning it on the sampled global degrades the cell-log-E marginal that the gate is built
# from). 60k steps rather than 40k: the shared heads now serve 17 classes.
echo "=== train multispecies_v1 ==="
$RUN scripts/train_calo_flow.py \
    --slice $SL/multispecies_v1.npz --out $CK/multispecies_v1 \
    --steps 60000 --pos_transform quantile --count_dither \
    --no_energy_glob --run_name calo_multispecies_v1 --no_wandb

# per-species gates on the held-out shard, with both physical bounds on.
# --logE_max is the multispecies slice max, printed by the trainer above; -1.13 is the pion
# slice value and the mixture's max is at least that (dominated by the same hadronic cells).
CKPT=$CK/multispecies_v1/checkpoint_060000.pt
for spec in "2:gamma" "3 4:pi"; do
    cls="${spec%%:*}"; name="${spec##*:}"
    echo "=== ms metrics: $name (pdg $cls) ==="
    $RUN scripts/calo_metrics.py --ckpt $CKPT --pdg_class $cls --shard 5 \
        --tag "ms_$name" --no_partition --logE_max -1.1328
done
echo "=== done ==="
