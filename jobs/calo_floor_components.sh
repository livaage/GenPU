#!/bin/bash
#SBATCH --job-name=calo_floor_comp
#SBATCH --time=01:10:00
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
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

# TWO independent questions from job 12473556 (--floor_n_buckets 32).
#
# (1) DIAGNOSTIC — which generative component owns the near-floor band?
#     Conditioning the Bernoulli on n moved p(band | n=1) by 0.003 (e- 0.061 -> 0.058, real 0.011),
#     i.e. essentially nothing. The likely reason: the band the gate scores is
#     `logE < log_floor + 0.5`, but the Bernoulli only models the NARROW pile
#     (|logE - log_floor| < 0.05). Everything else in the band is a MIXTURE draw, including the
#     physical sub-floor tail that EnergyHead.loss deliberately leaves to the mixture. If the e±
#     n=1 excess is mixture-tail rather than pile, no Bernoulli change can ever fix it — and that
#     also explains why `partition` could (it rescales cell energies, moving mixture draws out of
#     the band). Run on BOTH arms so the comparison is clean.
#
# (2) REPLICATE — is the e± gate gain real or seed noise?
#     Arm B moved e- gate8 0.7456 -> 0.7204 and e+ 0.7488 -> 0.7015 with cell_logE improving on all
#     three species (no collateral) — the first favourable e± gate move since Phase 1 itself. But it
#     is a single run per arm, it did NOT move frac_near_floor, and this project has repeatedly been
#     misled by single-run gate deltas. `--seed 1` on the identical arm-B recipe settles it.
#     (Every logged run before today used seed 0; --seed is new, default 0, so nothing else shifts.)

echo "##### (1) component decomposition — ARM A (Phase 1 helix, no floor head) #####"
$RUN scripts/calo_floor_dispersion.py --ckpt $CK/pion_anchor/checkpoint_040000.pt \
    --pdg_class 3 4 --shard 5 --tag compA_pion --no_partition --no_plot
for spec in "0:electron" "1:positron"; do
    cls="${spec%%:*}"; name="${spec##*:}"
    $RUN scripts/calo_floor_dispersion.py --ckpt $CK/electron_anchor/checkpoint_060000.pt \
        --pdg_class $cls --shard 5 --tag "compA_$name" --no_partition --no_plot
done

echo "##### (1) component decomposition — ARM B (--floor_n_buckets 32) #####"
$RUN scripts/calo_floor_dispersion.py --ckpt $CK/pion_floorn/checkpoint_040000.pt \
    --pdg_class 3 4 --shard 5 --tag compB_pion --no_partition --no_plot
for spec in "0:electron" "1:positron"; do
    cls="${spec%%:*}"; name="${spec##*:}"
    $RUN scripts/calo_floor_dispersion.py --ckpt $CK/electron_floorn/checkpoint_060000.pt \
        --pdg_class $cls --shard 5 --tag "compB_$name" --no_partition --no_plot
done

echo "##### (2) seed replicate of ARM B on e± (seed 1, otherwise identical) #####"
$RUN scripts/train_calo_flow.py --slice $SL/electron_anchor.npz --out $CK/electron_floorn_s1 \
    --steps 60000 --pos_transform quantile --count_dither --no_energy_glob \
    --floor_n_buckets 32 --seed 1 --run_name calo_electron_floorn_s1 --no_wandb

for spec in "0:electron" "1:positron"; do
    cls="${spec%%:*}"; name="${spec##*:}"
    echo "=== replicate metrics: $name (pdg $cls) ==="
    $RUN scripts/calo_metrics.py --ckpt $CK/electron_floorn_s1/checkpoint_060000.pt \
        --pdg_class $cls --shard 5 --tag "floorn_s1_$name" --no_partition
done
echo "=== done ==="
