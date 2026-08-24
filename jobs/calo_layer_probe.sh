#!/bin/bash
#SBATCH --job-name=calo_layer_probe
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
"""Is the calo LAYERED everywhere -- i.e. is 'depth' a CATEGORICAL layer index, exactly like the
tracker's layer_class (0..47) + residual? The 3D map read det 10 as 2,526 distinct radii at 0.08 mm
and called it continuous, but flat plates tiling a cylinder give cells whose CENTRE radius varies
across each plate (r = R/cos(theta)). So those may be within-layer spread around few real layers."""
import sys, numpy as np
sys.path.insert(0,'/home/lv7805/genpu/src')
from genpu.data import load_shard, build_event_index, explode_list_columns
BARREL={10,13}
ca=load_shard("calo_hits",0); ci=build_event_index(ca)
X,Y,Z,D,E=[],[],[],[],[]
for eid in sorted(ci)[:150]:
    c=explode_list_columns(ca,ci[eid])
    X.append(np.asarray(c["x"],np.float64)); Y.append(np.asarray(c["y"],np.float64))
    Z.append(np.asarray(c["z"],np.float64)); D.append(np.asarray(c["detector"],np.int64))
    E.append(np.asarray(c["total_energy"],np.float64))
x,y,z,d,e=map(np.concatenate,(X,Y,Z,D,E))
r=np.hypot(x,y); az=np.abs(z)
print("Cluster the depth coordinate per detector: are there DISCRETE layers?\n")
for det in np.unique(d):
    m=d==det; v=(r if det in BARREL else az)[m]
    lab='r' if det in BARREL else '|z|'
    vs=np.sort(v); gaps=np.diff(vs)
    # a layer boundary = a gap much larger than the typical within-layer step
    typ=np.percentile(gaps[gaps>0],50) if (gaps>0).any() else 0
    thr=max(typ*20, 0.5)
    nlayer=int((gaps>thr).sum())+1
    edges=np.concatenate([[vs[0]], vs[1:][gaps>thr]])
    # width of each cluster
    idx=np.searchsorted(edges, v, side='right')-1
    widths=[v[idx==k].max()-v[idx==k].min() for k in range(min(nlayer,60))]
    print(f"det {det:>2} ({'barrel' if det in BARREL else 'endcap'}, {lab}): "
          f"{m.sum():>8,} cells  median step {typ:.4f}  gap thr {thr:.3f}")
    print(f"        -> {nlayer:>4} CLUSTERS   median cluster width {np.median(widths):.3f} mm   "
          f"centres p5/50/95: {np.percentile(edges,5):.1f} / {np.percentile(edges,50):.1f} / "
          f"{np.percentile(edges,95):.1f}")
    if nlayer<=60:
        print(f"        centres: {np.round(edges[:12],1).tolist()}{' ...' if nlayer>12 else ''}")
print("\nIf cluster count is small and cluster WIDTH << spacing, depth is CATEGORICAL (layer index)")
print("and the calo should mirror the tracker: layer_class + within-layer residual.")
PYEOF
