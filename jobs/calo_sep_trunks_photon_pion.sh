#!/bin/bash
#SBATCH --job-name=calo_sep_ph_pi
#SBATCH --time=00:55:00
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

# Phase 0c: SEPARATE PER-HEAD CONDITIONING TRUNKS. Experimental hygiene, not a quality change --
# with one shared ParticleConditioning MLP, altering one head's task moved what the others saw
# (measured twice: e± cell_logE 0.017 -> 0.081, frac_floor AUC 0.616 -> 0.858, from GlobalHead-only
# changes). Gradient isolation verified: the glob loss now yields exactly 0 gradient in the point and
# energy trunks. Cost +11k params on 240k (~4%).
# This run RE-BASELINES: the requirement is NO REGRESSION vs the shared-trunk references, because
# shared trunks can also act as a regulariser. If a head regresses, fall back to loss-scale
# balancing on a shared trunk instead.
#   photon reference `photon_qtd_v1`: gate8 0.558, gate10 0.805
#   pion   reference `pion_qtd_v2`  : gate8 0.809, gate10 0.993

echo "=== train photon_sep (40k, separate trunks) ==="
$RUN scripts/train_calo_flow.py --slice $SL/photon_core.npz --out $CK/photon_sep \
    --steps 40000 --pos_transform quantile --count_dither --no_energy_glob \
    --separate_trunks --run_name calo_photon_sep --no_wandb
echo "=== photon metrics ==="
$RUN scripts/calo_metrics.py --ckpt $CK/photon_sep/checkpoint_040000.pt \
    --pdg_class 2 --shard 5 --tag sep_photon --no_partition --logE_max -2.5456

echo "=== train pion_sep (40k, separate trunks) ==="
$RUN scripts/train_calo_flow.py --slice $SL/pion_ctr_v2.npz --out $CK/pion_sep \
    --steps 40000 --pos_transform quantile --count_dither --no_energy_glob \
    --separate_trunks --run_name calo_pion_sep --no_wandb
echo "=== pion metrics ==="
$RUN scripts/calo_metrics.py --ckpt $CK/pion_sep/checkpoint_040000.pt \
    --pdg_class 3 4 --shard 5 --tag sep_pion --no_partition --logE_max -1.1328
echo "=== done ==="
