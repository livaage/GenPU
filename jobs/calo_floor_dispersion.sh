#!/bin/bash
#SBATCH --job-name=calo_floor_disp
#SBATCH --time=00:35:00
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

# DIAGNOSTIC ONLY — no training. Question: is the generated per-shower floor fraction
# UNDER-DISPERSED, i.e. is `frac_near_floor` a structural consequence of cells being sampled
# conditionally i.i.d. rather than an energy-head conditioning problem?
#
# Both heads are i.i.d. across cells given (particle embedding, sampled globals), so the at-floor
# count in a shower of n cells is Binomial(n, p) with variance EXACTLY n*p*(1-p). Real showers
# should be over-dispersed (a shower is compact-and-bright or diffuse-and-fringy as a whole).
# D = Var_obs(k|n) / (n*p*(1-p)) measured per n-bin, real vs generated.
#
#   D_real >> 1, D_gen ~ 1 -> structural: no amount of extra conditioning fixes it; the fix is a
#                             richer SAMPLED shower-level latent (floor fraction as a GlobalHead
#                             dim) or an actual set model. Explains why --energy_glob_idx 1 moved
#                             frac_near_floor only 0.618 -> 0.574 (the mean, not the variance).
#   D_real ~ D_gen         -> not structural; stays an energy-head conditioning question.
#
# `shower_width` runs alongside as a POSITIVE CONTROL: it is already believed i.i.d.-limited
# (calo_metrics.py:39-40, calo_flow.py:490-492), so a sound diagnostic must flag it too.
#
# Settings MATCH how these checkpoints were scored in jobs/calo_anchor_train.sh (--no_partition,
# helix anchor), so the numbers are comparable with the logged frac_near_floor AUCs:
#   pion  `pion_anchor`     frac_floor 0.6176   e- `electron_anchor` 0.6008   e+ 0.5924

for spec in "3 4:pion:pion_anchor/checkpoint_040000.pt" \
            "0:electron:electron_anchor/checkpoint_060000.pt" \
            "1:positron:electron_anchor/checkpoint_060000.pt"; do
    cls="${spec%%:*}"; rest="${spec#*:}"; name="${rest%%:*}"; ck="${rest##*:}"
    echo "=== floor dispersion: $name (pdg $cls) ==="
    $RUN scripts/calo_floor_dispersion.py --ckpt "$CK/$ck" \
        --pdg_class $cls --shard 5 --tag "$name" --no_partition
done
echo "=== done ==="
