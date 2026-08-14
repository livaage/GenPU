#!/bin/bash
#SBATCH --job-name=calo_ms_slice
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
export MAMBA_ROOT_PREFIX=/scratch/gpfs/IOJALVO/gnn-tracking/object_condensation/micromamba
export MAMBA_EXE=/home/lv7805/bin/micromamba
export HF_HOME=/scratch/gpfs/IOJALVO/lv7805/genpu_cache
cd /home/lv7805/genpu

# All-species calo slice for the single PDG-conditioned head. Census on held-out shard 5:
# e-+e+ = 48% of deposited energy / 57% of cells, pi± 29%/27%, p 13%/7%, gamma 2%/3%.
# --max_per_class 400000 caps the three dominant classes so the shared heads aren't set by
# e± alone, while classes below the cap (gamma, K, n, mu, pi0, ...) keep their full statistics.
$MAMBA_EXE run -n genpu2 python scripts/build_calo_slice.py \
    --shards 0 1 2 \
    --pdg_class 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 \
    --max_per_class 400000 \
    --out /scratch/gpfs/IOJALVO/lv7805/genpu_data/calo_slice/multispecies_v1.npz
echo "=== done ==="
