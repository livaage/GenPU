#!/bin/bash
#SBATCH --job-name=calo_partition_ab
#SBATCH --time=00:50:00
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
RUN="$MAMBA_EXE run -n genpu2 python"

# NO TRAINING — one flag on the existing Phase 1 checkpoints. Arm B = partition ON;
# arm A is every logged number, which was measured with --no_partition.
#
# 2026-08-16 (job 12470841) localised the frac_near_floor defect precisely: the e± error is ENTIRELY
# at n = 1, where real p(floor) = 0.011 (e-) / 0.009 (e+) and the model emits 0.061 / 0.062 — a 6x
# excess over ~8% of showers. Physically a one-cell shower's cell carries the ENTIRE shower energy,
# so it cannot be a faint fringe cell.
#
# `partition` should fix exactly that by construction. In sample_showers, an n=1 all-floor shower has
# sum_rest = 0, so `ok` is FALSE and the DEGENERATE path fires: s_all = total / e_floor is applied to
# the floor cell itself, lifting it to carry the whole total. (The docstring's "at-floor cells are
# pinned so the floor pile survives" holds only in the `ok` branch.) n >= 2 mixed showers keep the
# floor cell pinned and are expected to be unchanged.
#
# Partition was ruled out 2026-08-13 (photon gate 0.557 -> 0.944, pion 0.809 -> 0.998) because it
# degrades the POOLED cell_logE marginal the gate is built from. That was never checked at small n,
# and never on the Phase 1 anchored checkpoints. So this measures BOTH sides: does it fix
# p(floor|n=1), and what does it cost pooled?
#
# ARM A (logged, --no_partition):
#   pion `pion_anchor`     gate8 0.8055 gateW 0.9833 cell_logE 0.0223 frac_floor 0.6176
#   e-   `electron_anchor` gate8 0.7456 gateW 0.9639 cell_logE 0.0153 frac_floor 0.6008
#   e+                     gate8 0.7488 gateW 0.9578 cell_logE 0.0170 frac_floor 0.5924

for spec in "3 4:pion:pion_anchor/checkpoint_040000.pt" \
            "0:electron:electron_anchor/checkpoint_060000.pt" \
            "1:positron:electron_anchor/checkpoint_060000.pt"; do
    cls="${spec%%:*}"; rest="${spec#*:}"; name="${rest%%:*}"; ck="${rest##*:}"
    echo "=== MECHANISM: p(floor | n) with partition ON — $name (pdg $cls) ==="
    $RUN scripts/calo_floor_dispersion.py --ckpt "$CK/$ck" \
        --pdg_class $cls --shard 5 --tag "part_$name"
    echo "=== COST: full metrics with partition ON — $name (pdg $cls) ==="
    $RUN scripts/calo_metrics.py --ckpt "$CK/$ck" \
        --pdg_class $cls --shard 5 --tag "part_$name"
done
echo "=== done ==="
