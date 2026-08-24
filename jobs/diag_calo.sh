#!/bin/bash
#SBATCH --job-name=diag_calo
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
export MAMBA_ROOT_PREFIX=/scratch/gpfs/IOJALVO/gnn-tracking/object_condensation/micromamba
export MAMBA_EXE=/home/lv7805/bin/micromamba
export GENPU_DATA_DIR=/scratch/gpfs/IOJALVO/lv7805/genpu_cache/datasets
cd /home/lv7805/genpu
# NOTE: script inlined here on purpose — /tmp is NODE-LOCAL on this cluster, so a script written to
# the login node's /tmp is invisible to the compute node.
$MAMBA_EXE run -n genpu2 python - <<'PYEOF'
import sys, numpy as np
sys.path.insert(0,'/home/lv7805/genpu/src')
from genpu.data import load_shard, build_event_index, explode_list_columns
from genpu.preprocessing import process_event_vectorized, _empty_tracker, _empty_calo
t,tr,ca = load_shard("particles",0), load_shard("tracker_hits",0), load_shard("calo_hits",0)
pi,ti,ci = build_event_index(t), build_event_index(tr), build_event_index(ca)
s2=np.load('/scratch/gpfs/IOJALVO/lv7805/genpu_data/preprocessed/shard_0000_stage2.npz')
ch,coff,ev=s2['calo_hits_flat'],s2['calo_offsets'],s2['event_ids']
eid=3; rows=np.where(ev==eid)[0]
res=process_event_vectorized(explode_list_columns(t,pi[eid]),
    explode_list_columns(tr,ti[eid]) if eid in ti else _empty_tracker(),
    explode_list_columns(ca,ci[eid]) if eid in ci else _empty_calo())
vis=np.where(res['visible_mask'])[0]
print(f'event {eid}: visible {len(vis)}  stage2 rows {len(rows)}')
shown=0
for j in range(len(vis)):
    p_local, p_row = vis[j], rows[j]
    lo,hi=int(coff[p_row]),int(coff[p_row+1]); mine=res['calo_hits'][p_local]
    if len(mine)==0 and hi==lo: continue
    print(f'\n-- vis idx {j}: stored {hi-lo} rows, recomputed {len(mine)} --')
    if len(mine)==0 or hi==lo:
        shown+=1
        if shown>=4: break
        continue
    n=min(2,len(mine),hi-lo)
    print('  stored    :', np.round(ch[lo:lo+n],5).tolist())
    print('  recomputed:', np.round(mine[:n,:5],5).tolist())
    if (hi-lo)==len(mine):
        d=np.abs(mine[:,:5]-ch[lo:hi])
        print('  max abs diff per col:', np.round(d.max(0),6).tolist())
        a=np.sort(ch[lo:hi],axis=0); b=np.sort(mine[:,:5],axis=0)
        print('  identical as SETS (sorted per column)?', bool(np.allclose(a,b,atol=1e-4)))
    shown+=1
    if shown>=4: break
PYEOF
