"""Build the v2 per-particle tracker slice: v1 position residuals + a per-hit DIRECTION
state (the momentum-direction proxy) derived from the TRUE hit positions, + n_hits for the
END-token target. Built from the SOURCE arrow shards (they carry true_x/y/z; the preprocessed
stage2 dropped it).

Per hit stored (7 cols): [layer_class, r_resid, phi_resid, z_resid, time_resid, dir_theta, dir_alpha]
  *_resid    = (physical - LAYER_MEANS[lc]) / LAYER_STDS[lc]           (unchanged v1 encoding)
  dir_theta  = polar angle of the tangent (z vs transverse), phi-INVARIANT
  dir_alpha  = transverse turning angle (azimuthal vs radial at the hit), phi-INVARIANT
The tangent is normalize(true_posᵢ - true_posᵢ₋₁) (first hit: from the vertex), decomposed in the
LOCAL cylindrical frame at the hit so both angles are azimuthally symmetric (matches the
conditioning contract which excludes absolute phi).
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from genpu.data import load_shard, explode_list_columns
from genpu.detector_geometry import LAYER_MEANS, LAYER_STDS, N_LAYERS, hits_to_layer_class
from genpu.preprocessing import pdg_to_class


def tangent_angles(pos, vertex):
    """pos: (n,3) TRUE xyz sorted inner->outer. Return (theta, alpha) per hit, phi-invariant.
    tangent_i = normalize(pos_i - pos_{i-1}); pos_{-1} = vertex."""
    prev = np.vstack([vertex[None, :], pos[:-1]])
    t = pos - prev
    t /= (np.linalg.norm(t, axis=1, keepdims=True) + 1e-9)
    phi_hit = np.arctan2(pos[:, 1], pos[:, 0])
    cr, sr = np.cos(phi_hit), np.sin(phi_hit)
    t_r = t[:, 0] * cr + t[:, 1] * sr          # radial component
    t_phi = -t[:, 0] * sr + t[:, 1] * cr        # azimuthal component
    t_z = t[:, 2]
    theta = np.arctan2(np.hypot(t_r, t_phi), t_z)   # polar: z vs transverse
    alpha = np.arctan2(t_phi, t_r)                  # transverse turning
    return theta.astype(np.float32), alpha.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/scratch/gpfs/IOJALVO/lv7805/genpu_data/tracker_slice/v2_pion.npz")
    ap.add_argument("--shards", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--pdg_class", type=int, nargs="+", default=[3, 4])
    ap.add_argument("--min_hits", type=int, default=1)
    ap.add_argument("--max_particles", type=int, default=0)
    args = ap.parse_args()

    cont_l, pdg_l, hits_l, lengths = [], [], [], []
    for sh in args.shards:
        P = load_shard("particles", sh)
        T = load_shard("tracker_hits", sh)
        n_ev = min(P.num_rows, T.num_rows)
        for e in range(n_ev):
            pe = explode_list_columns(P, e)
            te = explode_list_columns(T, e)
            pcls = pdg_to_class(pe["pdg_id"].astype(np.int64))
            keep = np.where(np.isin(pcls, args.pdg_class))[0]
            if len(keep) == 0:
                continue
            # per-particle conditioning
            pt = np.hypot(pe["px"], pe["py"]); p = np.sqrt(pt**2 + pe["pz"]**2)
            eta = np.arcsinh(pe["pz"] / (pt + 1e-9))
            log_pt = np.log(pt + 1e-6); log_E = np.log(np.clip(pe["energy"], 1e-6, None))
            vr = np.hypot(pe["vx"], pe["vy"])
            tpid = te["particle_id"].astype(np.int64)
            for i in keep:
                pid_i = pe["particle_id"][i]
                m = tpid == pid_i
                nh = int(m.sum())
                if nh < args.min_hits:
                    continue
                x, y, z = te["x"][m], te["y"][m], te["z"][m]
                tx, ty, tz = te["true_x"][m], te["true_y"][m], te["true_z"][m]
                lc = hits_to_layer_class(te["volume_id"][m], te["layer_id"][m]).astype(np.int32).clip(0, N_LAYERS - 1)
                r = np.hypot(x, y); phi = np.arctan2(y, x)
                order = np.argsort(r, kind="stable")
                r, phi, z = r[order], phi[order], z[order]
                lc = lc[order]; time = te["time"][m][order].astype(np.float32)
                tpos = np.stack([tx, ty, tz], axis=1)[order]
                vtx = np.array([pe["vx"][i], pe["vy"][i], pe["vz"][i]], np.float32)
                phys = np.stack([r, phi, z, time], axis=1).astype(np.float32)
                resid = (phys - LAYER_MEANS[lc]) / LAYER_STDS[lc]
                theta, alpha = tangent_angles(tpos, vtx)
                hit = np.concatenate([lc[:, None].astype(np.float32), resid,
                                      theta[:, None], alpha[:, None]], axis=1)  # (nh, 7)
                cont_l.append([log_pt[i], eta[i], log_E[i], pe["charge"][i], pe["mass"][i], vr[i], pe["vz"][i]])
                pdg_l.append(float(pcls[i])); hits_l.append(hit); lengths.append(nh)
            if args.max_particles and len(cont_l) >= args.max_particles:
                break
        print(f"  shard {sh}: running total {len(cont_l)} particles", flush=True)
        if args.max_particles and len(cont_l) >= args.max_particles:
            break

    cont = np.asarray(cont_l, np.float32); pdg = np.asarray(pdg_l, np.float32)
    hits = np.concatenate(hits_l).astype(np.float32)
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int32)
    norm = {"cont_mean": cont.mean(0), "cont_std": cont.std(0) + 1e-6}
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, cont=cont, pdg=pdg, hits=hits, offsets=offsets, **norm)
    nph = np.diff(offsets)
    print(f"wrote {out}\n  particles={cont.shape[0]}  hits={hits.shape[0]}  hits/p mean={nph.mean():.1f} max={nph.max()}")
    print(f"  dir_theta mean={hits[:,5].mean():.3f} std={hits[:,5].std():.3f}  "
          f"dir_alpha mean={hits[:,6].mean():.3f} std={hits[:,6].std():.3f}")
    Path(str(out) + ".summary.json").write_text(json.dumps(
        {"particles": int(cont.shape[0]), "hits": int(hits.shape[0]), "pdg_class": args.pdg_class}, indent=2))


if __name__ == "__main__":
    main()
