#!/bin/bash
#SBATCH --job-name=calo_noatom_smoke
#SBATCH --time=00:55:00
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --qos=gpu-test
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
export MAMBA_ROOT_PREFIX=/scratch/gpfs/IOJALVO/gnn-tracking/object_condensation/micromamba
export MAMBA_EXE=/home/lv7805/bin/micromamba
cd /home/lv7805/genpu
RUN="$MAMBA_EXE run -n genpu2 python"
SL=/scratch/gpfs/IOJALVO/lv7805/genpu_data/calo_slice
CK=/scratch/gpfs/IOJALVO/lv7805/genpu_data/checkpoints/calo_flow
FD=/scratch/gpfs/IOJALVO/lv7805/genpu_data/gate_features

# DOES THE ENERGY MIXTURE NEED THE FLOOR ATOM AT ALL?
#
# Measured 2026-09-07 on 13,054,567 held-out e± cells (`electron_v2_h5.npz`):
#   cells EXACTLY at log(5e-5) : 0.000000   <- the atom models a value the data never takes
#   cells BELOW  log(5e-5)     : 0.00387    <- real, physical: a shared cell's ATTRIBUTED share
#   density step at threshold  : x19        (bin below 4,336 -> bin above 82,294)
# and the mixture beside the atom is trained with a HOLE cut out (`cont` mask) that the atom then
# fills back in. Circular -- the likeliest reason ten experiments since 2026-08-13 could not move
# `frac_near_floor`.
#
# THE ONE QUESTION THIS RUN ANSWERS: with the atom gone and the mixture trained on ALL cells, does
# a sum of Gaussians hold the x19 edge, or does it SMEAR across it?
#   Read `sub_floor_gen` against `sub_floor_real` = 0.00387 (new `floor_edge` block in
#   calo_metrics.py). One seed is enough for THIS: it is a fraction over ~13M cells, so seed noise
#   is negligible. The GATE question needs 2 seeds and is a separate, queued decision.
#
# PREDICTION recorded before the run: `at_floor_gen` -> 0.000000 (mechanical, guaranteed by the
# code path). `sub_floor_gen` is the real unknown -- if it lands near 0.004 the mixture holds the
# edge and the head just got simpler; if it lands at 0.01-0.03 it is smearing and the fix is to
# TRUNCATE the mixture at the threshold, NOT to restore the atom.
#
# CONTROL: electron_v2_s0 (60k, same slice, same recipe, atom ON) -- gate8 0.8105,
# frac_near_floor per-feature AUC 0.5045, full/marg/copula 0.8253/0.6007/0.7004.

echo "########## train: dedicated e+-, seed 0, --no_floor_atom (control electron_v2_s0) ##########"
$RUN scripts/train_calo_flow.py --slice $SL/electron_v2.npz --steps 60000 --log_every 5000 \
     --no_floor_atom --seed 0 --out $CK/ele_noatom_s0

echo "########## eval: identical to job 12930682's dedicated-e+- arm ##########"
$RUN scripts/calo_metrics.py --ckpt $CK/ele_noatom_s0/checkpoint_060000.pt \
     --pdg_class 0 1 --real_slice $SL/electron_v2_h5.npz \
     --tag ele_noatom_s0 --dump_features $FD/ele_noatom_s0.npz
$RUN scripts/calo_gate_diagnose.py --features $FD/ele_noatom_s0.npz --seeds 3 --pairs \
     --tag ele_noatom_s0
echo "=== done ==="
