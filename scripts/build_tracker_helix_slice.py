"""Build a v4 HELIX-anchored slice. In each particle's own frame (rotated by -phi0 so the initial
momentum points along +x — keeps the phi-invariant conditioning contract; r,z,layer are unchanged),
store per hit [layer_class, dev_x, dev_y, dev_z, time] where dev = rotated_hit - helix_ref[layer].
Also store the rotated vertex (vxr,vyr) + phi0 so the trainer/gate can recompute helix_ref from cont.
Reads raw source (needs x,y + volume/layer). CPU only.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from genpu.data import load_shard, explode_list_columns, build_event_index, compute_eta_phi
from genpu.preprocessing import pdg_to_class
from genpu.helix import layer_references
from genpu.detector_geometry import LAYER_LIST, LAYER_MEANS

# vectorized (vol,layer)->layer_class lookup
_LUT = np.full(40000, 0, np.int32)
for _i, (_v, _l) in enumerate(LAYER_LIST):
    _LUT[_v * 1000 + _l] = _i


def layer_class(vol, layer):
    return _LUT[np.asarray(vol).astype(np.int64) * 1000 + np.asarray(layer).astype(np.int64)]


def build(shards, pdg_classes, min_hits, population, max_events=0):
    cont_l, pdg_l, hits_l, lens, vxr_l, vyr_l, phi0_l = [], [], [], [], [], [], []
    for sh in shards:
        p = load_shard("particles", sh); t = load_shard("tracker_hits", sh)
        p_idx = build_event_index(p); t_idx = build_event_index(t)
        eids = sorted(p_idx.keys())
        if max_events:
            eids = eids[:max_events]
        nkeep = 0
        for eid in eids:
            if eid not in t_idx:
                continue
            pe = explode_list_columns(p, p_idx[eid]); te = explode_list_columns(t, t_idx[eid])
            px, py, pz = pe["px"].astype(np.float64), pe["py"].astype(np.float64), pe["pz"].astype(np.float64)
            pT, eta, phi0 = compute_eta_phi(px, py, pz)
            log_pt = np.log(np.clip(pT, 1e-6, None))
            pdgc = pdg_to_class(pe["pdg_id"].astype(np.int64))
            charge = pe["charge"].astype(np.float64) if "charge" in pe else np.zeros_like(px)
            mass = pe["mass"].astype(np.float64) if "mass" in pe else np.zeros_like(px)
            vx = pe["vx"].astype(np.float64) if "vx" in pe else np.zeros_like(px)
            vy = pe["vy"].astype(np.float64) if "vy" in pe else np.zeros_like(px)
            vz = pe["vz"].astype(np.float64) if "vz" in pe else np.zeros_like(px)
            energy = pe["energy"].astype(np.float64) if "energy" in pe else np.zeros_like(px)
            primary = pe["primary"].astype(np.float64) if "primary" in pe else np.zeros_like(px)
            pid = pe["particle_id"].astype(np.int64)

            hpid = te["particle_id"].astype(np.int64)
            hx, hy, hz = te["x"].astype(np.float64), te["y"].astype(np.float64), te["z"].astype(np.float64)
            htime = te["time"].astype(np.float64) if "time" in te else np.zeros_like(hx)
            hvol, hlay = np.asarray(te["volume_id"]), np.asarray(te["layer_id"])
            order = np.argsort(hpid, kind="stable")
            hpid = hpid[order]; hx, hy, hz, htime = hx[order], hy[order], hz[order], htime[order]
            hvol, hlay = hvol[order], hlay[order]

            sel = np.isin(pdgc, pdg_classes)
            if population == "primary":
                sel &= primary > 0.5
            elif population == "secondary":
                sel &= primary <= 0.5
            for i in np.where(sel)[0]:
                lo = np.searchsorted(hpid, pid[i], "left"); hi = np.searchsorted(hpid, pid[i], "right")
                if hi - lo < min_hits:
                    continue
                x, y, z, tm = hx[lo:hi], hy[lo:hi], hz[lo:hi], htime[lo:hi]
                vol, lay = hvol[lo:hi], hlay[lo:hi]
                r = np.hypot(x, y); o = np.argsort(r, kind="stable")          # inner->outer
                x, y, z, tm, vol, lay = x[o], y[o], z[o], tm[o], vol[o], lay[o]
                ph = phi0[i]; cph, sph = np.cos(ph), np.sin(ph)
                xr = x * cph + y * sph; yr = -x * sph + y * cph               # rotate by -phi0
                vxr = vx[i] * cph + vy[i] * sph; vyr = -vx[i] * sph + vy[i] * cph
                lc = layer_class(vol, lay)
                ref = layer_references(np.array([pT[i]]), np.array([0.0]), np.array([eta[i]]),
                                       np.array([charge[i]]), np.array([vxr]), np.array([vyr]),
                                       np.array([vz[i]]))[0]                   # (48,3)
                dev = np.nan_to_num(np.stack([xr, yr, z], 1) - ref[lc])       # (n,3) rotated-frame deviation
                hh = np.concatenate([lc[:, None].astype(np.float32), dev.astype(np.float32),
                                     tm[:, None].astype(np.float32)], 1)
                log_E = np.log(max(float(energy[i]), 1e-6))
                cont_l.append([log_pt[i], eta[i], log_E, charge[i], mass[i], np.hypot(vx[i], vy[i]), vz[i]])
                pdg_l.append(float(pdgc[i])); hits_l.append(hh); lens.append(len(lc))
                vxr_l.append(vxr); vyr_l.append(vyr); phi0_l.append(ph); nkeep += 1
        print(f"  shard {sh}: kept {nkeep}", flush=True)
    cont = np.asarray(cont_l, np.float32); pdg = np.asarray(pdg_l, np.float32)
    hits = np.concatenate(hits_l).astype(np.float32)
    offsets = np.concatenate([[0], np.cumsum(lens)]).astype(np.int32)
    vxr = np.asarray(vxr_l, np.float32); vyr = np.asarray(vyr_l, np.float32); phi0 = np.asarray(phi0_l, np.float32)
    return cont, pdg, hits, offsets, vxr, vyr, phi0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/scratch/gpfs/IOJALVO/lv7805/genpu_data/tracker_slice/helix_pion.npz")
    ap.add_argument("--shards", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--pdg_class", type=int, nargs="+", default=[3, 4])
    ap.add_argument("--min_hits", type=int, default=1)
    ap.add_argument("--population", choices=["all", "primary", "secondary"], default="all")
    ap.add_argument("--max_events", type=int, default=0)
    args = ap.parse_args()
    cont, pdg, hits, offsets, vxr, vyr, phi0 = build(
        args.shards, args.pdg_class, args.min_hits, args.population, args.max_events)
    norm = {"cont_mean": cont.mean(0), "cont_std": cont.std(0) + 1e-6}
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, cont=cont, pdg=pdg, hits=hits, offsets=offsets,
                        vxr=vxr, vyr=vyr, phi0=phi0, **norm)
    nph = np.diff(offsets); dev = hits[:, 1:4]
    print(f"\nwrote {out}  particles={cont.shape[0]}  hits={hits.shape[0]}  hits/particle mean={nph.mean():.2f}")
    print(f"  dev [mm]  x: p50 {np.percentile(np.abs(dev[:,0]),50):.2f} p99 {np.percentile(np.abs(dev[:,0]),99):.1f}  "
          f"y: p50 {np.percentile(np.abs(dev[:,1]),50):.2f} p99 {np.percentile(np.abs(dev[:,1]),99):.1f}  "
          f"z: p50 {np.percentile(np.abs(dev[:,2]),50):.2f} p99 {np.percentile(np.abs(dev[:,2]),99):.1f}")
    frac_clip = np.mean((np.abs(dev[:,0])>120)|(np.abs(dev[:,1])>120)|(np.abs(dev[:,2])>300))
    print(f"  frac hits beyond bin range (clipped): {frac_clip:.4f}")
    Path(str(out)+".summary.json").write_text(json.dumps(
        {"particles": int(cont.shape[0]), "hits": int(hits.shape[0]), "pdg_class": args.pdg_class,
         "shards": args.shards}, indent=2))


if __name__ == "__main__":
    main()
