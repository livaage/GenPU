#!/bin/bash
#SBATCH --job-name=calo_epos_coupling_ab
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
S=scripts/calo_energy_position_coupling.py

# THE MISSING HALF OF THE --energy_pos EXPERIMENT (jobs 13038484 train / 13045670 eval).
#
# That experiment put each cell's own (depth-within-section, transverse radius) + a 4-way
# ECAL/HCAL x barrel/endcap token into `EnergyHead`, whose every input had until then been a
# per-SHOWER quantity gathered by [src] -- so cell energy was independent of cell position BY
# CONSTRUCTION. The GATE was measured and moved (dedicated e+- WORSE 0.811/0.807 -> 0.941/0.876;
# pooled e+- BETTER 0.908/0.885 -> 0.848/0.871; all classes ~flat 0.935/0.889 -> 0.902/0.904),
# and the pre-registered prediction ("the copula moves, the marginal does not") FAILED in both
# arms, in opposite directions.
#
# But THE MECHANISM TARGET WAS NEVER RE-MEASURED. The whole point was that generated
# rho(logE, depth) was exactly 0.000 where real is +0.52 (p) .. -0.11 (gamma). Without this we
# cannot tell "the head learned the coupling and we paid for it elsewhere" from "the head never
# learned it at all" -- and those imply opposite next steps.
#
# CONTROLS (job 13032162, same script, same arms, seed-0 checkpoints), generated side:
#   e+- -0.01 real / +0.000 gen ; pi+- +0.27 / +0.000 ; gamma -0.101 / +0.000 ; p +0.542 / +0.000
#   p(floor) core->fringe: real x2.4-4.5, generated x1.00 flat
#
# PREDICTION, recorded before the run: gen rho becomes NON-ZERO and same-signed as real on every
# species. If it is still ~0.000 the position input is not reaching the output distribution and
# the gate movement is something else entirely -- which would make the dedicated-e+- regression a
# reason to look for a bug, not a trade-off.
#
# Both seeds on the two e+- arms because that is where the gate arms disagreed most (0.941 vs
# 0.876); pion and photon on seed 0 only, matching the control.

E0=$CK/ele_epos_s0/checkpoint_060000.pt
E1=$CK/ele_epos_s1/checkpoint_060000.pt
M0=$CK/ms_epos_s0/checkpoint_060000.pt
M1=$CK/ms_epos_s1/checkpoint_060000.pt

echo "########## 1. e+- DEDICATED seed 0  (control: ele_dedicated, gen rho +0.000) ##########"
$RUN $S --real_slice $SL/electron_v2_h5.npz --ckpt $E0 --pdg_class 0 1 \
        --max_showers 300000 --tag ele_epos_s0
echo "########## 2. e+- DEDICATED seed 1 ##########"
$RUN $S --real_slice $SL/electron_v2_h5.npz --ckpt $E1 --pdg_class 0 1 \
        --max_showers 300000 --tag ele_epos_s1
echo "########## 3. e+- POOLED seed 0  (control: ms_ele) ##########"
$RUN $S --real_slice $SL/multispecies_v2_h5.npz --ckpt $M0 --pdg_class 0 1 \
        --max_showers 300000 --tag ms_epos_ele_s0
echo "########## 4. e+- POOLED seed 1 ##########"
$RUN $S --real_slice $SL/multispecies_v2_h5.npz --ckpt $M1 --pdg_class 0 1 \
        --max_showers 300000 --tag ms_epos_ele_s1
echo "########## 5. pi+- POOLED seed 0  (control: ms_pion, real +0.27 / gen +0.000) ##########"
$RUN $S --real_slice $SL/multispecies_v2_h5.npz --ckpt $M0 --pdg_class 3 4 \
        --max_showers 300000 --tag ms_epos_pion_s0
echo "########## 6. gamma POOLED seed 0  (control: ms_photon, real -0.101 / gen +0.000) ##########"
$RUN $S --real_slice $SL/multispecies_v2_h5.npz --ckpt $M0 --pdg_class 2 \
        --max_showers 300000 --tag ms_epos_photon_s0
echo "=== done ==="
