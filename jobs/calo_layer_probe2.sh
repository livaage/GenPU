#!/bin/bash
#SBATCH --job-name=calo_layer2
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
export MAMBA_ROOT_PREFIX=/scratch/gpfs/IOJALVO/gnn-tracking/object_condensation/micromamba
export MAMBA_EXE=/home/lv7805/bin/micromamba
export GENPU_DATA_DIR=/scratch/gpfs/IOJALVO/lv7805/genpu_cache/datasets
cd /home/lv7805/genpu
$MAMBA_EXE run -n genpu2 python - <<'PYEOF'
"""Layer structure, without a hand-picked threshold. If the calo is layered, the gaps between
consecutive DISTINCT depth values are bimodal: many tiny (within-layer, from plate geometry) and a
few large (layer boundaries). Print the gap distribution and let it speak."""
import sys, numpy as np
sys.path.insert(0,'/home/lv7805/genpu/src')
from genpu.data import load_shard, build_event_index, explode_list_columns
BARREL={10,13}
ca=load_shard("calo_hits",0); ci=build_event_index(ca)
X,Y,Z,D=[],[],[],[]
for eid in sorted(ci)[:150]:
    c=explode_list_columns(ca,ci[eid])
    X.append(np.asarray(c["x"],np.float64)); Y.append(np.asarray(c["y"],np.float64))
    Z.append(np.asarray(c["z"],np.float64)); D.append(np.asarray(c["detector"],np.int64))
x,y,z,d=map(np.concatenate,(X,Y,Z,D))
r=np.hypot(x,y); az=np.abs(z)
for det in np.unique(d):
    m=d==det; v=(r if det in BARREL else az)[m]
    lab='r' if det in BARREL else '|z|'
    u=np.unique(np.round(v,3)); g=np.diff(u)
    print(f"\n=== det {det} ({'barrel' if det in BARREL else 'endcap'}, {lab}) — "
          f"{m.sum():,} cells, {len(u):,} distinct ===")
    if len(g)==0: continue
    print(f"  gap percentiles: " + "  ".join(f"p{q}={np.percentile(g,q):.3f}" for q in (10,50,75,90,95,99)))
    print(f"  gap max {g.max():.3f}   n gaps > 1mm: {(g>1).sum()}   > 3mm: {(g>3).sum()}   "
          f"> 10mm: {(g>10).sum()}")
    # count distinct values that carry real occupancy (>=0.1% of cells) -- plate spread is diffuse,
    # real layers are populated
    vals,cnt=np.unique(np.round(v,1),return_counts=True)
    keep=cnt>=max(len(v)*0.001,5)
    print(f"  distinct depths at 0.1mm holding >=0.1% of cells: {keep.sum()}  "
          f"(they hold {cnt[keep].sum()/len(v):.3f} of cells)")
    if keep.sum()<=60:
        print(f"    {np.round(vals[keep],1).tolist()}")
    else:
        kv=vals[keep]; print(f"    span {kv.min():.1f}..{kv.max():.1f}, median spacing {np.median(np.diff(kv)):.3f}")
PYEOF
