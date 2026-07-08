"""Cascade spike step 1: characterize PHOTON CONVERSIONS in the pu0 truth.

A photon -> e+e- conversion is the cleanest, most regular material effect and a
big chunk of the cascade (~223k photon parents). This measures the conditional a
generator would have to reproduce:
  - conversion fraction (photons with e+/- daughters)
  - conversion radius (where the e+e- are produced) + path length from the photon
  - conversion fraction vs photon energy
  - e+e- energy split (leading-daughter energy fraction)
Tells us whether photon->conversion is regular enough to learn cheaply.
"""
from __future__ import annotations
import argparse, glob
import numpy as np
import pyarrow as pa


def load(sub, shard):
    base = "/scratch/gpfs/IOJALVO/lv7805/genpu_cache/CERN___collider_ml-release-1"
    f = sorted(glob.glob(f"{base}/pileup_only_pu0_{sub}/*/*/*.arrow"))[shard]
    try:
        return pa.ipc.open_file(pa.memory_map(f)).read_all()
    except pa.lib.ArrowInvalid:
        return pa.ipc.open_stream(pa.memory_map(f)).read_all()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n_events", type=int, default=3000)
    args = ap.parse_args()
    t = load("particles", args.shard)
    N = min(args.n_events, t.num_rows)
    pid = t.column("particle_id").to_pylist(); parent = t.column("parent_id").to_pylist()
    pdg = t.column("pdg_id").to_pylist(); energy = t.column("energy").to_pylist()
    vx = t.column("vx").to_pylist(); vy = t.column("vy").to_pylist()

    n_photon = 0; n_conv = 0
    conv_r = []          # conversion radius (mm) = e+e- production radius
    conv_path = []       # conv_r - photon_production_r
    ph_E_all = []; ph_conv = []      # photon energy + whether it converted
    ph_anydau = []       # whether the photon has ANY daughter particle
    ph_convr = []        # per-photon median conversion radius (nan if no e daughter)
    esplit = []          # leading e± energy / (sum e± energy)

    for ev in range(N):
        ids = np.asarray(pid[ev], np.int64); par = np.asarray(parent[ev], np.int64)
        pg = np.asarray(pdg[ev], np.int64); en = np.asarray(energy[ev], np.float64)
        r = np.hypot(np.asarray(vx[ev]), np.asarray(vy[ev]))
        idx = {int(i): k for k, i in enumerate(ids)}
        # daughters grouped by parent
        dau = {}
        for k in range(len(ids)):
            p = idx.get(int(par[k]))
            if p is not None:
                dau.setdefault(p, []).append(k)
        for k in range(len(ids)):
            if pg[k] != 22:
                continue
            n_photon += 1
            ph_E_all.append(en[k])
            alldau = dau.get(k, [])
            ph_anydau.append(1.0 if len(alldau) else 0.0)
            ee = [dk for dk in alldau if abs(pg[dk]) == 11]
            ph_conv.append(1.0 if len(ee) >= 1 else 0.0)
            if len(ee) >= 1:
                n_conv += 1
                cr = float(np.median([r[dk] for dk in ee]))   # conversion radius
                conv_r.append(cr); conv_path.append(cr - float(r[k])); ph_convr.append(cr)
                if len(ee) >= 2:
                    ees = sorted([en[dk] for dk in ee], reverse=True)
                    esplit.append(ees[0] / max(sum(ees), 1e-9))
            else:
                ph_convr.append(np.nan)

    conv_r = np.array(conv_r); conv_path = np.array(conv_path)
    ph_E_all = np.array(ph_E_all); ph_conv = np.array(ph_conv); esplit = np.array(esplit)
    print(f"events={N}  photons={n_photon}  converted(>=1 e daughter)={n_conv} ({n_conv/n_photon:.1%})")
    print(f"\nconversion radius (mm): median={np.median(conv_r):.0f} p10={np.percentile(conv_r,10):.0f} p90={np.percentile(conv_r,90):.0f}")
    for lo, hi in [(0,50),(50,200),(200,600),(600,1100),(1100,1e9)]:
        print(f"  conv_r [{lo:5.0f},{hi:6.0f}) mm : {((conv_r>=lo)&(conv_r<hi)).mean():6.1%}")
    print(f"\nconversion PATH length (conv_r - photon_r), mm: median={np.median(conv_path):.0f} "
          f"(neg frac {np.mean(conv_path<0):.1%} = photon born past conv point / rounding)")
    ph_anydau = np.array(ph_anydau); ph_convr = np.array(ph_convr)
    print(f"\nvs photon energy — conv_frac (e daughter) | any_daughter_frac | median conv_r:")
    eb = np.quantile(ph_E_all, np.linspace(0, 1, 6))
    for b in range(5):
        m = (ph_E_all >= eb[b]) & (ph_E_all < eb[b+1] + (1e-6 if b == 4 else 0))
        cr_m = np.nanmedian(ph_convr[m]) if np.any(~np.isnan(ph_convr[m])) else float("nan")
        print(f"  E [{eb[b]:7.3f},{eb[b+1]:7.3f}) GeV : conv={ph_conv[m].mean():5.1%}  "
              f"any_dau={ph_anydau[m].mean():5.1%}  conv_r={cr_m:6.0f}mm  (n={m.sum()})")
    if len(esplit):
        print(f"\ne+e- leading-energy fraction: mean={esplit.mean():.3f} std={esplit.std():.3f} "
              f"(0.5=symmetric split; median={np.median(esplit):.3f})")


if __name__ == "__main__":
    main()
