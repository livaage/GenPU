"""Close the tracker-cascade loop (photon conversions).

For converting photons: generate e± via the conversion model (radius + energy
split + collinear kinematics), and separately take the TRUTH e± daughters. Run
the SAME multi-species tracker head on both, and compare the resulting tracker
hits (layer occupancy, radius). Both go through the same head, so differences are
purely the CASCADE generation quality — the key thing the loop validates.

n_hits is fixed (same for truth and gen) since there is no count head yet, so this
tests hit PLACEMENT (which layers/radii the e± populate — set by the conversion
vertex + kinematics), not hit count.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from genpu.models.tracker_model import TrackerModel
from conversion_spike import ConvModel  # reuse the conversion model class

M_E = 5.11e-4  # electron mass (GeV)


def elec_cont(log_pt, eta, log_E, charge, vr, vz):
    """Build the 7 tracker CONT_FEATURES for an electron."""
    return np.stack([log_pt, eta, log_E, charge, np.full_like(log_pt, M_E), vr, vz], axis=1)


def elec_pdg(charge):   # e- (charge<0) -> class 0 ; e+ -> class 1
    return (charge > 0).astype(np.int64)


def run_tracker(model, cont, pdg, n_hits, dev, batch=8192):
    """Return flat (layer, r) arrays for all generated hits."""
    layers, rs = [], []
    contS = torch.as_tensor((cont - model.cont_mean.cpu().numpy()) / model.cont_std.cpu().numpy(),
                            dtype=torch.float32, device=dev)
    pdgT = torch.as_tensor(pdg, device=dev)
    for s in range(0, len(cont), batch):
        e = min(s + batch, len(cont))
        with torch.no_grad():
            ce = model.cond_embed(contS[s:e], pdgT[s:e])
            nh = torch.full((e - s,), n_hits, dtype=torch.long, device=dev)
            hits, lay = model.tracker.generate(ce, nh)
        hits = hits.cpu().numpy(); lay = lay.cpu().numpy()
        for j in range(e - s):
            layers.append(lay[j, :n_hits]); rs.append(hits[j, :n_hits, 0])
    return np.concatenate(layers), np.concatenate(rs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracker_ckpt", required=True)
    ap.add_argument("--tracker_slice", default="/scratch/gpfs/IOJALVO/lv7805/genpu_data/tracker_slice/multispecies.npz")
    ap.add_argument("--conv_ckpt", default="/scratch/gpfs/IOJALVO/lv7805/genpu_data/cascade/photon_conv_model.pt")
    ap.add_argument("--loop_slice", default="/scratch/gpfs/IOJALVO/lv7805/genpu_data/cascade/conv_loop.npz")
    ap.add_argument("--n_hits", type=int, default=6)
    ap.add_argument("--max_photons", type=int, default=40000)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    tn = np.load(args.tracker_slice)
    tracker = TrackerModel({"cont_mean": tn["cont_mean"], "cont_std": tn["cont_std"]}).to(dev)
    tracker.load_state_dict(torch.load(args.tracker_ckpt, map_location=dev)["model"]); tracker.eval()
    cc = torch.load(args.conv_ckpt, map_location=dev)
    conv = ConvModel(cc["norm"]).to(dev); conv.load_state_dict(cc["model"]); conv.eval()

    d = np.load(args.loop_slice)
    ph = d["ph"]; ee = d["ee_flat"]; ee_vz = d["ee_vz"]; off = d["ee_off"]
    S = min(len(ph), args.max_photons)
    ph = ph[:S]; off = off[:S + 1]; ee = ee[:off[-1]]; ee_vz = ee_vz[:off[-1]]

    # ---- truth e± -> tracker ----
    t_cont = elec_cont(ee[:, 0], ee[:, 1], ee[:, 3], ee[:, 4], ee[:, 5], ee_vz)
    t_pdg = elec_pdg(ee[:, 4])
    t_lay, t_r = run_tracker(tracker, t_cont, t_pdg, args.n_hits, dev)

    # ---- generated e± (conversion model, collinear approx) -> tracker ----
    pcond = torch.as_tensor(ph[:, [0, 1, 3, 4]], device=dev)  # [log_E, eta, vr, vz]
    with torch.no_grad():
        clog, (rmu, rls), (smu, sls) = conv.heads(pcond)
        g_lcr = (rmu + torch.randn_like(rmu) * rls.exp()).cpu().numpy()
        g_split = (smu + torch.randn_like(smu) * sls.exp()).clamp(0.5, 1.0).cpu().numpy()
    logE_ph, eta_ph, phi_ph, vr_ph, vz_ph = ph[:, 0], ph[:, 1], ph[:, 2], ph[:, 3], ph[:, 4]
    E_ph = np.exp(logE_ph)
    conv_r = np.exp(g_lcr)
    vz_conv = vz_ph + (conv_r - vr_ph) * np.sinh(eta_ph)
    # two electrons: leading (split) and sub (1-split), collinear with photon
    g_cont, g_pdg = [], []
    for frac, ch in [(g_split, 1.0), (1.0 - g_split, -1.0)]:
        E = np.clip(frac * E_ph, 1e-6, None)
        pt = E / np.cosh(eta_ph)
        g_cont.append(elec_cont(np.log(pt), eta_ph, np.log(E), np.full_like(E, ch), conv_r, vz_conv))
        g_pdg.append(np.full(len(E), 1 if ch > 0 else 0, dtype=np.int64))
    g_cont = np.concatenate(g_cont); g_pdg = np.concatenate(g_pdg)
    g_lay, g_r = run_tracker(tracker, g_cont, g_pdg, args.n_hits, dev)

    # ---- compare truth-e± vs gen-e± tracker hits ----
    print("=" * 56); print("TRACKER-CASCADE LOOP (photon conversions)"); print("=" * 56)
    print(f"photons={S}  truth-e±={len(t_cont)}  gen-e±={len(g_cont)}  (n_hits/e±={args.n_hits})")
    print(f"\nlayer_class:  truth mean {t_lay.mean():.2f} std {t_lay.std():.2f}   "
          f"gen mean {g_lay.mean():.2f} std {g_lay.std():.2f}")
    print(f"frac inner (<12):  truth {(t_lay<12).mean():.1%}   gen {(g_lay<12).mean():.1%}")
    print(f"hit radius (mm):  truth mean {t_r.mean():.0f} std {t_r.std():.0f}   "
          f"gen mean {g_r.mean():.0f} std {g_r.std():.0f}")
    out = Path(args.tracker_ckpt).parent / "eval" / "cascade_loop.json"; out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({
        "layer_mean": {"truth": float(t_lay.mean()), "gen": float(g_lay.mean())},
        "frac_inner": {"truth": float((t_lay < 12).mean()), "gen": float((g_lay < 12).mean())},
        "r_mean": {"truth": float(t_r.mean()), "gen": float(g_r.mean())}}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
