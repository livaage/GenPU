#!/bin/bash
#SBATCH --job-name=calo_floor_nbuckets
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

# ARM B — the at-floor Bernoulli sees an EMBEDDING of the shower's cell count.
# Single variable vs the Phase 1 helix checkpoints: `--floor_n_buckets 32`. Same slices, same step
# counts, shared trunk, --no_energy_glob, helix anchor — byte-identical recipes otherwise, so the
# floor head's view of n is the only thing that changed. +9,985 params (~4%).
#
# WHY, measured (job 12470841, scripts/calo_floor_dispersion.py): p(floor | n) per bin.
#   e±   the ENTIRE error is at n=1 — real 0.011 (e-) / 0.009 (e+) vs generated 0.061 / 0.062,
#        a 6x excess over ~8% of showers each; real and gen agree to ~0.001 by n>=9.
#   pion no n-dependence AT ALL — real p climbs 0.018 -> 0.032 with n, generated is flat ~0.018.
# A single log_n scalar cannot express either shape, which is why --energy_glob_idx 1 was worth only
# frac_near_floor 0.618 -> 0.574. Physically: a one-cell shower's cell carries the ENTIRE shower
# energy, so it cannot be a faint fringe cell.
#
# WHY THIS ROUTE AND NOT `partition` (job 12471293): partition forces the same n=1 fix at generation
# time and it WORKS on target (e- n=1 0.061 -> 0.011, exactly real) but rescales every cell's
# continuous energy — cell_logE 0.0153 -> 0.1194, gate8 0.7456 -> 0.9882, and it made the generated
# floor count OVER-dispersed (D 1.02 -> 2.09) via the shared multiplicative scale. This head changes
# only a DISCRETE membership decision, so neither failure mode is available to it.
#
# ARM A (logged reference, --no_partition):
#   pion `pion_anchor`     gate8 0.8055 gateW 0.9833 cell_logE 0.0223 frac_floor 0.6176
#   e-   `electron_anchor` gate8 0.7456 gateW 0.9639 cell_logE 0.0153 frac_floor 0.6008
#   e+                     gate8 0.7488 gateW 0.9578 cell_logE 0.0170 frac_floor 0.5924
#
# PRIMARY read = the MECHANISM (p(floor|n) from calo_floor_dispersion.py), not the gate. Every one of
# the last seven changes improved its own target and lost on a gate whose 8 of 10 features are
# energy/multiplicity; judge this one on whether the per-n curve moves onto truth, then on the gate.

echo "=== train pion_floorn (40k, + floor_n_buckets 32) ==="
$RUN scripts/train_calo_flow.py --slice $SL/pion_anchor.npz --out $CK/pion_floorn \
    --steps 40000 --pos_transform quantile --count_dither --no_energy_glob \
    --floor_n_buckets 32 --run_name calo_pion_floorn --no_wandb

echo "=== pion: MECHANISM p(floor|n) ==="
$RUN scripts/calo_floor_dispersion.py --ckpt $CK/pion_floorn/checkpoint_040000.pt \
    --pdg_class 3 4 --shard 5 --tag floorn_pion --no_partition
echo "=== pion: full metrics ==="
$RUN scripts/calo_metrics.py --ckpt $CK/pion_floorn/checkpoint_040000.pt \
    --pdg_class 3 4 --shard 5 --tag floorn_pion --no_partition

echo "=== train electron_floorn (60k, + floor_n_buckets 32) ==="
$RUN scripts/train_calo_flow.py --slice $SL/electron_anchor.npz --out $CK/electron_floorn \
    --steps 60000 --pos_transform quantile --count_dither --no_energy_glob \
    --floor_n_buckets 32 --run_name calo_electron_floorn --no_wandb

for spec in "0:electron" "1:positron"; do
    cls="${spec%%:*}"; name="${spec##*:}"
    echo "=== $name: MECHANISM p(floor|n) ==="
    $RUN scripts/calo_floor_dispersion.py --ckpt $CK/electron_floorn/checkpoint_060000.pt \
        --pdg_class $cls --shard 5 --tag "floorn_$name" --no_partition
    echo "=== $name: full metrics ==="
    $RUN scripts/calo_metrics.py --ckpt $CK/electron_floorn/checkpoint_060000.pt \
        --pdg_class $cls --shard 5 --tag "floorn_$name" --no_partition
done
echo "=== done ==="
