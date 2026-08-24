#!/bin/bash
#SBATCH --job-name=calo_v2_train
#SBATCH --time=00:55:00
#SBATCH --gres=gpu:1
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

# FIRST 3D calo model, on re-attributed showers. 60k steps to match electron_anchor's recipe so the
# only differences are the data corrections. TWO SEEDS: this establishes the v2 baseline AND its
# seed spread, which every future comparison needs (the v1 e± gate8 spread was ~0.11).
#
# NOT comparable to any logged v1 number: v1 gated 2D fragment-showers against a v1 reference; this
# gates 3D re-attributed showers against a v2 reference. Different task, different target.
# Quote event_gate_auc_depth -- the 8-feature gate is blind to lateral shape, and both older
# variants are blind to depth.
for s in 0 1; do
  echo "########## TRAIN seed $s ##########"
  $RUN scripts/train_calo_flow.py --slice $SL/electron_v2.npz \
       --out $CK/electron_v2_s$s --steps 60000 --seed $s --no_wandb
  echo "########## EVAL seed $s (held-out shard 5, v2 reference) ##########"
  $RUN scripts/calo_metrics.py --ckpt $CK/electron_v2_s$s/checkpoint_060000.pt \
       --pdg_class 0 1 --real_slice $SL/electron_v2_h5.npz --tag electron_v2_s$s
done
echo "=== done ==="
