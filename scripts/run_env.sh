#!/bin/bash
# Convenience wrapper to run commands in the genpu micromamba environment.
# Usage: ./scripts/run_env.sh python my_script.py
#        ./scripts/run_env.sh jupyter notebook

export MAMBA_ROOT_PREFIX='/scratch/gpfs/IOJALVO/gnn-tracking/object_condensation/micromamba'
export MAMBA_EXE='/home/lv7805/bin/micromamba'
export HF_HOME='/scratch/gpfs/IOJALVO/genpu_cache'

exec "$MAMBA_EXE" run -n genpu "$@"
