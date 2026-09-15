#!/bin/bash
#SBATCH --job-name=calo_pos_smoke
#SBATCH --time=00:25:00
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

# END-TO-END smoke for --energy_pos. BOTH arms, short, on the real e± slice: the control must be
# bit-comparable to the existing recipe (position-blind) and the new arm must train and CHECKPOINT.
# The 2026-08-27 slice-rebuild lesson: a smoke sized only to prove correctness cannot see a scaling
# bug, so this one runs the real slice at real per-point batch size, just for few steps.
echo "########## CONTROL: position-blind (existing recipe) ##########"
$RUN scripts/train_calo_flow.py --slice $SL/electron_v2.npz --steps 300 --log_every 100 \
     --out $CK/_smoke_posblind --seed 0
echo "########## ARM: --energy_pos ##########"
$RUN scripts/train_calo_flow.py --slice $SL/electron_v2.npz --steps 300 --log_every 100 \
     --energy_pos --out $CK/_smoke_energypos --seed 0
echo "########## RELOAD + GENERATE from the --energy_pos checkpoint ##########"
$RUN scripts/calo_metrics.py --ckpt $CK/_smoke_energypos/checkpoint_000300.pt \
     --pdg_class 0 1 --real_slice $SL/electron_v2_h5.npz --tag _smoke_energypos
echo "=== done ==="
