"""Build a per-particle tracker-hit slice in the paper-style (arXiv:2512.24254) SURFACE-LOCAL
frame, the surface-granular counterpart of build_tracker_slice.py.

Per hit:  (module_index, x_res, y_res, z_res, time_res)
   module_index = (volume,layer,surface) module id  (see genpu.module_geometry)
   *_res        = (physical_local - MODULE_MEANS[m]) / MODULE_STDS[m]  ~N(0,1), small & bounded
   hits SORTED inner->outer by physical r (matches AR generation order).

Conditioning: the same 7 CONT_FEATURES as build_tracker_slice
   [log_pt, eta, log_E, charge, mass, vr, vz]  (phi excluded — phi-invariant response).

Reads the RAW source (needs surface_id + x,y — both dropped by stage2), reusing the same
event-join + particle-grouping as preprocessing.py. CPU only. Requires module_geometry.npz.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from genpu.data import load_shard, explode_list_columns, build_event_index, compute_eta_phi
from genpu.preprocessing import pdg_to_class
from genpu.module_geometry import ModuleGeometry


def build(shards, pdg_classes, min_hits, population, mg: ModuleGeometry, max_events=0):
    cont_list, pdg_list, hits_list, lengths = [], [], [], []
    n_unseen_hits = 0
    for sh in shards:
        p = load_shard("particles", sh); t = load_shard("tracker_hits", sh)
        p_idx = build_event_index(p); t_idx = build_event_index(t)
        eids = sorted(p_idx.keys())
        if max_events:
            eids = eids[:max_events]
        n_keep = 0
        for eid in eids:
            if eid not in t_idx:
                continue
            pe = explode_list_columns(p, p_idx[eid])
            te = explode_list_columns(t, t_idx[eid])
            # particle features / selection
            pt, eta, _phi = compute_eta_phi(pe["px"].astype(np.float32), pe["py"].astype(np.float32),
                                            pe["pz"].astype(np.float32))
            log_pt = np.log(np.clip(pt, 1e-6, None))
            pdg_cls = pdg_to_class(pe["pdg_id"].astype(np.int64))
            charge = pe["charge"].astype(np.float32) if "charge" in pe else np.zeros_like(pt)
            mass = pe["mass"].astype(np.float32) if "mass" in pe else np.zeros_like(pt)
            vx = pe["vx"].astype(np.float32) if "vx" in pe else np.zeros_like(pt)
            vy = pe["vy"].astype(np.float32) if "vy" in pe else np.zeros_like(pt)
            vz = pe["vz"].astype(np.float32) if "vz" in pe else np.zeros_like(pt)
            energy = pe["energy"].astype(np.float32) if "energy" in pe else np.zeros_like(pt)
            primary = pe["primary"].astype(np.float32) if "primary" in pe else np.zeros_like(pt)
            pid = pe["particle_id"].astype(np.int64)
            vr = np.hypot(vx, vy)

            # tracker hits grouped by particle_id (searchsorted, as in preprocessing)
            hpid = te["particle_id"].astype(np.int64)
            hx, hy, hz = te["x"].astype(np.float32), te["y"].astype(np.float32), te["z"].astype(np.float32)
            htime = te["time"].astype(np.float32) if "time" in te else np.zeros_like(hx)
            hvol, hlay, hsurf = te["volume_id"], te["layer_id"], te["surface_id"]
            order = np.argsort(hpid, kind="stable")
            hpid_s = hpid[order]
            hx, hy, hz, htime = hx[order], hy[order], hz[order], htime[order]
            hvol = np.asarray(hvol)[order]; hlay = np.asarray(hlay)[order]; hsurf = np.asarray(hsurf)[order]

            sel_mask = np.isin(pdg_cls, pdg_classes)
            if population == "primary":
                sel_mask &= primary > 0.5
            elif population == "secondary":
                sel_mask &= primary <= 0.5
            for i in np.where(sel_mask)[0]:
                lo = np.searchsorted(hpid_s, pid[i], side="left")
                hi = np.searchsorted(hpid_s, pid[i], side="right")
                if hi - lo < min_hits:
                    continue
                x, y, z, tm = hx[lo:hi], hy[lo:hi], hz[lo:hi], htime[lo:hi]
                midx = mg.to_index(hvol[lo:hi], hlay[lo:hi], hsurf[lo:hi])
                keep = midx >= 0
                n_unseen_hits += int((~keep).sum())
                if keep.sum() < min_hits:
                    continue
                x, y, z, tm, midx = x[keep], y[keep], z[keep], tm[keep], midx[keep]
                r = np.hypot(x, y)
                o = np.argsort(r, kind="stable")               # inner->outer
                x, y, z, tm, midx = x[o], y[o], z[o], tm[o], midx[o]
                xyzt = np.stack([x, y, z, tm], axis=1).astype(np.float32)
                resid = mg.local_residual(midx, xyzt)          # (n,4) ~N(0,1)
                hits = np.concatenate([midx[:, None].astype(np.float32), resid], axis=1)

                log_E = np.log(max(float(energy[i]), 1e-6))
                cont_list.append([log_pt[i], eta[i], log_E, charge[i], mass[i], vr[i], vz[i]])
                pdg_list.append(float(pdg_cls[i]))
                hits_list.append(hits.astype(np.float32))
                lengths.append(len(midx))
                n_keep += 1
        print(f"  shard {sh}: kept {n_keep} particles (pdg={pdg_classes}, >= {min_hits} hits)", flush=True)

    cont = np.asarray(cont_list, np.float32)
    pdg = np.asarray(pdg_list, np.float32)
    hits = np.concatenate(hits_list).astype(np.float32)
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int32)
    return cont, pdg, hits, offsets, n_unseen_hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/scratch/gpfs/IOJALVO/lv7805/genpu_data/tracker_slice/surface_pion.npz")
    ap.add_argument("--module_geometry", default=None)
    ap.add_argument("--shards", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--pdg_class", type=int, nargs="+", default=[3, 4])
    ap.add_argument("--min_hits", type=int, default=1)
    ap.add_argument("--population", choices=["all", "primary", "secondary"], default="all")
    ap.add_argument("--max_events", type=int, default=0)
    args = ap.parse_args()

    mg = ModuleGeometry(args.module_geometry)
    print(f"module vocab: {mg.n_modules:,} modules")
    cont, pdg, hits, offsets, n_unseen = build(
        args.shards, args.pdg_class, args.min_hits, args.population, mg, args.max_events)
    S = cont.shape[0]
    norm = {"cont_mean": cont.mean(0), "cont_std": cont.std(0) + 1e-6}
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, cont=cont, pdg=pdg, hits=hits, offsets=offsets,
                        n_modules=np.int64(mg.n_modules), **norm)

    nph = np.diff(offsets)
    print(f"\nwrote {out}")
    print(f"  particles: {S}   hits: {hits.shape[0]}   hits/particle mean={nph.mean():.2f} "
          f"median={np.median(nph):.0f} max={nph.max()}")
    print(f"  unseen-module hits dropped: {n_unseen}")
    print(f"  local resid std  x={hits[:,1].std():.3f} y={hits[:,2].std():.3f} "
          f"z={hits[:,3].std():.3f} t={hits[:,4].std():.3f}   (target ~1.0)")
    print(f"  local resid p99 |x|={np.percentile(np.abs(hits[:,1]),99):.2f} "
          f"|y|={np.percentile(np.abs(hits[:,2]),99):.2f} |z|={np.percentile(np.abs(hits[:,3]),99):.2f}")
    print(f"  distinct modules used: {len(np.unique(hits[:,0]))}")
    Path(str(out) + ".summary.json").write_text(json.dumps(
        {"particles": int(S), "hits": int(hits.shape[0]), "n_modules": int(mg.n_modules),
         "pdg_class": args.pdg_class, "shards": args.shards, "unseen_hits": int(n_unseen)}, indent=2))


if __name__ == "__main__":
    main()
