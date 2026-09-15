"""Derive the CALO CELL GRID -- the geometry PIPELINE.md gap #1 needs and nobody has measured.

WHY THIS EXISTS
The calo model emits a continuous (d_eta, d_phi, depth) point cloud and nothing ever snaps it onto
a real cell (`sample_showers` feeds raw sampled depth straight to `depth_to_region_local`). Gap 3e
already showed the LONGITUDINAL axis is a ladder of 48 / 36 exact planes for ~83% of endcap energy.
The TRANSVERSE axis has never been characterised at all: gap 2a measured only that ~2e7 distinct
cell positions exist and that quantising to 5 mm merges 2.4% of them while 10 mm merges 50% -- the
signature of a 5-10 mm pitch, but not the grid itself.

WHAT THE LOGIN-NODE PEEK ALREADY ESTABLISHED (2026-09-14, shard 0, 400 events)
  - Endcap ECAL (det 9/11) nearest-neighbour spacing within ONE layer is sharply peaked at
    5.1 mm, with secondary modes at 7.21 (= 5.1*sqrt2, the diagonal), 10.2 (= 2*5.1) and
    11.4 (= 5.1*sqrt5). That is a SQUARE lattice of pitch 5.1 mm, seen through sparse occupancy.
  - It is NOT one global axis-aligned lattice: circular concentration R = |<exp(2*pi*i*x/5.1)>|
    is 0.21 (1.0 would be a perfect single grid), and that holds per-layer, so it is not layer
    mixing either.
  - Rotating by phi-sector helps but does not resolve it: sweeping 4..64 sectors peaks at N=8
    with R = 0.50. So there IS module structure, and it is not a pure 8-fold rotation.

So point->cell is the SAME class of problem as the tracker's: locally regular lattices carried by
modules that are individually rotated and offset in the global frame. `module_geometry.npz` is the
precedent, and the 2026-09-07 tracker finding is the warning -- "local" must mean ROTATED INTO THE
MODULE FRAME, not merely centred and scaled, or the plane/lattice constraint is thrown away again.

WHAT THIS SCRIPT MEASURES (characterisation, NOT yet a production LUT)
  A. pitch          NN-distance spectrum per detector and per layer -> the lattice constant, and
                    whether it is constant across layers and detectors.
  B. tiling         Cells binned by (phi sector, radial band) -- a PARAMETRIC fit, not an
                    enumeration. The first draft of this script used connected components of the
                    "neighbours at ~pitch" graph and it FAILED for a reason worth recording:
                    occupancy never saturates. At 150 events layer 0 of det 9 gave 4,314 distinct
                    positions from 4,431 hits (0.974 distinct per hit), the graph fragmented into
                    3,558 components with a largest of 11 cells, and no component reached 20.
                    A layer holds ~2.6e5 cells (pi*(1497^2-317^2)/5.1^2), gap 2a measured ~2e7
                    across the calorimeter with 43% still unseen at event 4,000, and one shard is
                    ~7k events -- so ENUMERATING the cells is not on the table at any event count
                    we can afford. The lattice must be FITTED from sparse samples instead, which
                    needs only that NN pairs at ~pitch exist (29% of NN distances at this
                    occupancy), not that neighbourhoods be contiguous.
  C. orientation    Per bin, the angle of its ~pitch NN vectors modulo 90 deg (a square lattice's
                    axes are indistinguishable), then the origin by circular fit in the rotated
                    frame. Reports R BEFORE and AFTER the fit: the rise is the evidence that the
                    bin really carries one lattice. Angles clustered at 2*pi*k/N would confirm an
                    N-fold mechanical layout and give an analytic snap; angles spread continuously
                    would mean a per-module table is unavoidable.
  D. on-grid frac   THE ACCEPTANCE TEST, and the only number that decides whether this is usable:
                    after fitting each bin's (angle, origin), what fraction of real cells sit
                    within tol of a lattice site, and what is the median residual. The endcap layer
                    derivation cleared 100.0% on the longitudinal axis (job 12884120); anything far
                    below that here means the transverse model is wrong, not merely imprecise.

Deliberately NOT done here: the barrel. Staves tile a cylinder, gap #2 already showed barrel cells
from different layers overlap in radius (max gap 0.985 mm over a 106 mm span), and gap 3e measured
only 4.6% of barrel cells on a plane. The barrel needs its own treatment and conflating the two is
how the depth axis ended up ambiguous in the first place.

Usage:
  python scripts/calo_cell_grid_derive.py --shards 0 --events 4000 --out calo_cell_grid.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from genpu.data import load_shard  # noqa: E402

# from build_calo_slice_v2 (measured 2026-08-24, job 12884120). det -> (|z| of layer 0, pitch, n)
ENDCAP_LAYERS = {9: (3202.4, 5.050, 48), 11: (3202.4, 5.050, 48),
                 12: (3647.5, 51.000, 36), 14: (3647.5, 51.000, 36)}
BARREL_DETS = (10, 13)


def _flat(table, col):
    """Concatenate a per-event list column into one flat numpy array, entirely inside Arrow."""
    return table.column(col).combine_chunks().flatten().to_numpy(zero_copy_only=False)


def nn_dist_and_vec(P, chunk=4096, dev="cpu"):
    """Nearest-neighbour distance and vector for each point. torch, not scipy: scipy.spatial's
    compiled extensions fail to import in the genpu2 env (CXXABI mismatch against the system
    libstdc++), which is an environment fact, not a reason to change environments."""
    T = torch.as_tensor(P, dtype=torch.float64, device=dev)
    d_out, v_out = [], []
    for i in range(0, len(T), chunk):
        d = torch.cdist(T[i:i + chunk], T)
        j = torch.arange(i, min(i + chunk, len(T)), device=dev)
        d[torch.arange(d.shape[0], device=dev), j] = float("inf")
        dm, am = d.min(1)
        d_out.append(dm)
        v_out.append(T[am] - T[i:i + chunk])
    return torch.cat(d_out).cpu().numpy(), torch.cat(v_out).cpu().numpy()


def pitch_from_nn(nn, lo=1.0, hi=80.0, binw=0.02):
    """Lattice constant = the MODE of the NN-distance spectrum. Sparse occupancy puts mass at
    pitch*sqrt2, 2*pitch, pitch*sqrt5 ... so the mode is the right estimator and the presence of
    those harmonics is the confirmation that it IS a lattice."""
    m = (nn > lo) & (nn < hi)
    if m.sum() < 50:
        return None, {}
    h, e = np.histogram(nn[m], bins=np.arange(lo, hi, binw))
    c = 0.5 * (e[1:] + e[:-1])
    p = float(c[int(np.argmax(h))])
    harm = {}
    for lab, mult in (("x1", 1.0), ("sqrt2", np.sqrt(2)), ("x2", 2.0), ("sqrt5", np.sqrt(5))):
        w = np.abs(nn - p * mult) < 0.15 * mult
        harm[lab] = round(float(w.mean()), 4)
    return p, harm


def refine_pitch(P, vec, at_pitch, p_seed, n_patch=24, span=0.02, steps=4001):
    """Refine the lattice constant from a MODE SEED by maximising circular concentration.

    `pitch_from_nn` returns the mode of a histogram and therefore quantises the answer to the bin
    width. On 2026-09-15 that produced 5.0900 where the truth is 5.1 mm EXACTLY -- a 0.0100 mm
    error, which sounds negligible and is not: phase slips a full pitch every ~510 cells, so every
    fit needed radial bands narrow enough to keep the slip under half a pitch, and `--r_band` was
    silently doing that rather than describing any real structure. With 5.09998 a SINGLE origin
    covers r 317-1481 mm at on-grid 1.0000, median residual 0.0011 mm.

    Method: take the densest local patches (small enough that one lattice surely holds), fit each
    patch's angle from its ~pitch NN vectors, rotate, and scan pitch for the peak of
    0.5*(R_u + R_v). Report the MEDIAN over patches -- one patch can be degenerate, the median of
    two dozen is not. Patches are local so a slightly-wrong seed cannot bias the angle.
    """
    r = np.hypot(P[:, 0], P[:, 1])
    ph = np.arctan2(P[:, 1], P[:, 0])
    # ADAPTIVE patch grid. A hardcoded 64x16 silently produced ZERO usable patches on HCAL
    # (~9k distinct cells -> ~9 per patch, under the 40 floor), so `refine_pitch` returned the
    # seed unchanged and the log looked like a refinement that had simply agreed with the mode.
    # Size the grid from the point count instead, targeting ~200 cells per patch, and favour phi
    # resolution over radial since a module spans more radius than azimuth.
    # FULL RADIAL EXTENT per patch (n_r = 1). Pitch is estimated from how far the lattice phase
    # stays coherent, so the estimator's precision goes as the LEVER ARM, not as the number of
    # patches. Subdividing radially (a 13x3 grid on HCAL) shortened the arm, broadened the
    # concentration peak, and returned 30.131 mm against a true ~29.99 -- on-grid 0.127, WORSE
    # than not refining at all. Wedges spanning the full radius gave 5.09998 on ECAL (R = 1.0000
    # over r 317-1481) and ~29.993 on HCAL. A wedge straddling two module orientations yields a
    # bad angle and hence a bad pitch, which is why the MEDIAN over wedges is taken, not the best.
    n_r = 1
    n_phi = int(np.clip(len(P) // 200, 8, 64))
    key = (np.floor((ph + np.pi) / (2 * np.pi / n_phi)).astype(int) * n_r
           + np.clip(((r - r.min()) / max((r.max() - r.min()) / n_r, 1e-9)).astype(int), 0, n_r - 1))
    uk, cnt = np.unique(key, return_counts=True)
    grid = np.linspace(p_seed * (1 - span), p_seed * (1 + span), steps)
    out = []
    for k in uk[np.argsort(cnt)[::-1][:n_patch]]:
        sel = key == k
        if sel.sum() < 30 or (sel & at_pitch).sum() < 6:
            continue
        ang, _ = lattice_angle(vec[sel & at_pitch])
        ca, sa = np.cos(-ang), np.sin(-ang)
        Q = P[sel]
        u = Q[:, 0] * ca - Q[:, 1] * sa
        v = Q[:, 0] * sa + Q[:, 1] * ca
        # vectorised over the pitch grid: (steps, n) would be large, so loop in chunks of the grid
        best_p, best_R = p_seed, -1.0
        for i in range(0, len(grid), 500):
            g = grid[i:i + 500][:, None]
            R = 0.5 * (np.abs(np.exp(2j * np.pi * u[None, :] / g).mean(1))
                       + np.abs(np.exp(2j * np.pi * v[None, :] / g).mean(1)))
            j = int(np.argmax(R))
            if R[j] > best_R:
                best_R, best_p = float(R[j]), float(g[j, 0])
        out.append(best_p)
    if not out:
        return p_seed, 0, float("nan")
    # SPREAD IS THE DIAGNOSTIC, not a nuisance. If one lattice constant really exists, every wedge
    # must find it and the spread is ~0: ECAL gives 5.09998 with an IQR of order 1e-4 mm. HCAL's
    # estimate MOVES with the partition (29.993 per-sector on 12k events, 30.019 over 24 wedges,
    # 30.131 with a radial subdivision) and on-grid never exceeds ~0.8 for any of them. A pitch
    # that depends on how the data is sliced means there is no single pitch, i.e. the rotated-
    # square-lattice model is WRONG there -- which is a result about HCAL, not a tuning failure.
    q1, q3 = np.percentile(out, [25, 75])
    return float(np.median(out)), len(out), float(q3 - q1)


def grid_R(v, pitch):
    """Circular concentration of v modulo pitch. 1.0 = every value on one lattice, 0 = uniform."""
    if len(v) == 0:
        return 0.0, 0.0
    zc = np.mean(np.exp(2j * np.pi * np.mod(v, pitch) / pitch))
    return float(abs(zc)), float(np.angle(zc) / (2 * np.pi) * pitch)


def annulus_subsample(P, max_pts, n_keep=3):
    """Cut the point set down to `max_pts` while PRESERVING LOCAL NEIGHBOURHOODS.

    The obvious `rng.choice(len(P), max_pts)` is WRONG here and cost job 13920087 two of its
    three ECAL layers. Deleting points at random from a lattice deletes each cell's true nearest
    neighbours, so the NN-distance mode climbs the harmonics and `pitch_from_nn` returns a
    multiple of the pitch instead of the pitch. Measured on det 9 layer 0 (12,485 distinct
    positions, true pitch 5.09 mm):

        retained   1.00     0.50     0.25     0.17
        pitch      5.09     7.21     11.41    11.41
                   ok       =5.09*sqrt2       =5.09*sqrt5

    The job kept 17% at layer 0 and duly reported 7.21 mm, then failed every sector fit built on
    it. Layer 30 survived at 21% by luck of density, not by design.

    So subsample by whole RADIAL ANNULI: every kept point keeps all its neighbours except at the
    two annulus edges, and an annulus many pitches wide makes that negligible. Annuli rather than
    phi wedges DELIBERATELY -- wedge boundaries would impose an angular period on the data and the
    whole point of the sector scan is to measure that period, so wedges would be circular.
    Annuli span full phi and impose nothing on it.
    """
    r = np.hypot(P[:, 0], P[:, 1])
    order = np.argsort(r)
    step = int(np.ceil(len(P) / max_pts))
    n_ann = max(step * n_keep, 1)
    edges = np.linspace(0, len(P), n_ann + 1).astype(int)   # equal-COUNT annuli
    take = [order[edges[i]:edges[i + 1]] for i in range(0, n_ann, step)][:n_keep]
    return P[np.concatenate(take)] if take else P


def lattice_angle(vec):
    """Dominant lattice orientation from NN vectors, modulo 90 deg (a square lattice's two axes
    are indistinguishable). Circular mean at 4x frequency -> the 4-fold-symmetric mean angle."""
    a = np.arctan2(vec[:, 1], vec[:, 0])
    zc = np.mean(np.exp(4j * a))
    return float(np.angle(zc) / 4.0), float(abs(zc))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=int, nargs="+", default=[0])
    ap.add_argument("--events", type=int, default=4000, help="events per shard")
    ap.add_argument("--dets", type=int, nargs="+", default=[9, 11, 12, 14])
    ap.add_argument("--layers", type=int, nargs="+", default=[0, 4, 20],
                    help="layer indices to decompose into modules (part B/C/D)")
    ap.add_argument("--tol", type=float, default=0.5, help="on-grid tolerance [mm] for part D")
    ap.add_argument("--sectors", type=int, default=0,
                    help="phi sectors for the bin fit; 0 = scan --sectors_scan and report the best")
    ap.add_argument("--sectors_scan", type=int, nargs="+",
                    default=[1, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128])
    ap.add_argument("--r_band", type=float, default=1e9,
                    help="radial band width [mm] for the bin fit; huge = phi sectors only")
    ap.add_argument("--min_bin", type=int, default=40, help="min cells in a bin to fit it")
    ap.add_argument("--r_accept", type=float, default=0.95,
                    help="post-fit R a sector count must reach to be accepted (smallest N wins)")
    ap.add_argument("--min_bins", type=int, default=8,
                    help="a sector count needs at least this many fitted bins to be believed")
    ap.add_argument("--n_annuli", type=int, default=3,
                    help="how many whole radial annuli to keep when subsampling (see "
                         "annulus_subsample -- random subsampling breaks the pitch estimator)")
    ap.add_argument("--max_pts", type=int, default=60000,
                    help="cap on distinct positions per layer for the O(n^2) parts")
    ap.add_argument("--out", type=str, default="calo_cell_grid.json")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device {dev}   shards {a.shards}   events/shard {a.events}")

    det_l, x_l, y_l, z_l = [], [], [], []
    for sh in a.shards:
        t = load_shard("calo_hits", sh)
        t = t.slice(0, min(a.events, t.num_rows))
        # FLATTEN IN ARROW, never via to_pylist: these columns hold ~5.5k cells per event, so
        # 3 shards x 7k events is ~1.1e8 values per column and a Python list of that many floats
        # is ~30x the numpy footprint. `flatten()` on the chunked list column gives the same
        # concatenation with no Python objects at all. (The login-node peek used to_pylist on a
        # few hundred events, which is fine; at job scale it is an OOM.)
        det_l.append(_flat(t, "detector").astype(np.int16))
        x_l.append(_flat(t, "x").astype(np.float64))
        y_l.append(_flat(t, "y").astype(np.float64))
        z_l.append(_flat(t, "z").astype(np.float64))
        print(f"  shard {sh}: {len(det_l[-1]):,} cell-hits over {t.num_rows:,} events")
    det = np.concatenate(det_l); x = np.concatenate(x_l)
    y = np.concatenate(y_l); z = np.concatenate(z_l)
    n_events = sum(min(a.events, load_shard("calo_hits", s).num_rows) for s in a.shards)
    print(f"total {len(det):,} cell-hits, {n_events:,} events\n")

    out = {"n_events": int(n_events), "shards": a.shards, "detectors": {}}

    for d in a.dets:
        if d not in ENDCAP_LAYERS:
            print(f"det {d}: not an endcap detector -- skipped (barrel needs its own treatment)")
            continue
        z0, step, nlay = ENDCAP_LAYERS[d]
        md = det == d
        if md.sum() < 1000:
            print(f"det {d}: only {md.sum()} hits, skipped")
            continue
        lay = np.clip(np.rint((np.abs(z[md]) - z0) / step), 0, nlay - 1).astype(int)
        off_grid = np.abs((np.abs(z[md]) - z0) - lay * step)
        print(f"=== det {d}: {md.sum():,} hits, {nlay} layers at {step} mm")
        print(f"    longitudinal check: median |off-plane| {np.median(off_grid):.4f} mm, "
              f"p99 {np.percentile(off_grid, 99):.4f} mm  [gap 3e reported 0.000]")
        drec = {"n_hits": int(md.sum()), "layer_pitch": step,
                "long_offgrid_med": round(float(np.median(off_grid)), 5),
                "layers": {}}
        X, Y = x[md], y[md]

        for li in a.layers:
            ml = lay == li
            if ml.sum() < 200:
                print(f"  layer {li}: {ml.sum()} hits -- too few, skipped")
                continue
            P = np.unique(np.round(np.stack([X[ml], Y[ml]], 1), 3), axis=0)
            sat = len(P) / max(ml.sum(), 1)
            n_full = len(P)
            if n_full > a.max_pts:
                P = annulus_subsample(P, a.max_pts, a.n_annuli)
            nn, vec = nn_dist_and_vec(P, dev=dev)
            pitch, harm = pitch_from_nn(nn)
            if pitch is None:
                print(f"  layer {li}: no pitch found")
                continue
            # MODE IS ONLY A SEED -- refine before anything downstream uses it. See refine_pitch.
            at_seed = np.abs(nn - pitch) < 0.15 * pitch
            p_ref, n_patch, p_iqr = refine_pitch(P, vec, at_seed, pitch)
            if n_patch:
                stable = p_iqr < 0.002 * p_ref
                print(f"    pitch refined {pitch:.4f} -> {p_ref:.5f} mm "
                      f"(median of {n_patch} wedges, IQR {p_iqr:.5f} mm) "
                      f"{'STABLE' if stable else '** UNSTABLE: no single pitch -> lattice model suspect **'}")
                pitch = p_ref
            else:
                print(f"    WARNING: pitch NOT refined -- no patch had enough cells; the mode "
                      f"value {pitch:.4f} is quantised to the {0.02} mm histogram bin and every "
                      f"number below inherits that error")
            Rx, ox = grid_R(P[:, 0], pitch)
            Ry, oy = grid_R(P[:, 1], pitch)
            print(f"  layer {li}: {ml.sum():,} hits -> {n_full:,} distinct "
                  f"(distinct/hit {sat:.3f}; ->1 means occupancy NOT saturated, add events)"
                  + (f"  [subsampled to {len(P):,} in {a.n_annuli} radial annuli]"
                     if len(P) < n_full else ""))
            print(f"    pitch {pitch:.4f} mm   NN mass at x1/sqrt2/x2/sqrt5 = "
                  f"{harm['x1']:.3f}/{harm['sqrt2']:.3f}/{harm['x2']:.3f}/{harm['sqrt5']:.3f}")
            print(f"    global-grid R: x {Rx:.4f}  y {Ry:.4f}   [1.0 = one axis-aligned lattice]")

            # B/C/D: PARAMETRIC fit on spatial bins. `--sectors 0` sweeps candidate phi-sector
            # counts and reports which one maximises the post-fit concentration, so the mechanical
            # symmetry is measured rather than assumed; a positive value fixes it.
            r_all = np.hypot(P[:, 0], P[:, 1])
            ph_all = np.arctan2(P[:, 1], P[:, 0])
            # only NN vectors AT the pitch carry lattice orientation; longer ones jump modules
            at_pitch = np.abs(nn - pitch) < 0.15 * pitch
            cand = a.sectors_scan if a.sectors <= 0 else [a.sectors]
            best = None
            for NS in cand:
                sec = np.floor((ph_all + np.pi) / (2 * np.pi / NS)).astype(int)
                nb = max(int(np.ceil((r_all.max() - r_all.min()) / max(a.r_band, 1e-6))), 1)
                rb = np.clip(((r_all - r_all.min()) / max(a.r_band, 1e-6)).astype(int), 0, nb - 1)
                key = sec * nb + rb
                angs, res_all, ongrid_n, tot_n, Rpost, secs = [], [], 0, 0, [], []
                for k in np.unique(key):
                    sel = key == k
                    if sel.sum() < a.min_bin or (sel & at_pitch).sum() < 8:
                        continue
                    ang, conc = lattice_angle(vec[sel & at_pitch])
                    ca, sa = np.cos(-ang), np.sin(-ang)
                    Q = P[sel]
                    u = Q[:, 0] * ca - Q[:, 1] * sa
                    v = Q[:, 0] * sa + Q[:, 1] * ca
                    Ru, ou = grid_R(u, pitch)
                    Rv, ov = grid_R(v, pitch)
                    Rpost.append(0.5 * (Ru + Rv))
                    angs.append((ang, conc, int(sel.sum())))
                    secs.append(int(k // nb) if nb else 0)
                    ru = np.mod(u - ou + pitch / 2, pitch) - pitch / 2
                    rv = np.mod(v - ov + pitch / 2, pitch) - pitch / 2
                    r = np.hypot(ru, rv)
                    res_all.append(r)
                    ongrid_n += int((r < a.tol).sum())
                    tot_n += len(r)
                if not tot_n:
                    continue
                score = float(np.median(Rpost))
                rec = dict(NS=NS, nbins=len(angs), score=score, secs=secs,
                           ongrid=ongrid_n / tot_n, res=np.concatenate(res_all), angs=angs)
                print(f"      sectors={NS:3d}  bins={len(angs):4d}  post-fit R(med)={score:.4f}  "
                      f"on-grid={rec['ongrid']:.4f}  res_med={np.median(rec['res']):.4f} mm")
                # PARSIMONY, not max score. The first draft took argmax(R) and picked 128 sectors
                # on a 3-bin fit (R 0.9956) over 32 sectors on 89 bins (0.9937) -- a selection
                # artifact, since any multiple of the true sector count also aligns with module
                # boundaries and simply has fewer cells per bin. The physical answer is the
                # SMALLEST N that resolves the lattice, so take the first to clear `--r_accept`
                # with enough bins to mean anything.
                if score >= a.r_accept and len(angs) >= a.min_bins and best is None:
                    best = rec
            if best is None:
                print(f"    NO sector count reached R >= {a.r_accept} with >= {a.min_bins} bins -- "
                      f"the lattice model does not hold here, or occupancy is too sparse to fit it")
                continue
            res = best["res"]
            aa = np.array([t[0] for t in best["angs"]])
            cc = np.array([t[1] for t in best["angs"]])
            print(f"    BEST sectors={best['NS']}  bins={best['nbins']}  "
                  f"global R {0.5*(Rx+Ry):.4f} -> post-fit {best['score']:.4f}")
            print(f"    orientation: lattice concentration median {np.median(cc):.3f}; "
                  f"angle mod 90deg span {np.degrees(aa.min()):.2f}..{np.degrees(aa.max()):.2f} deg")
            # ANGLE PATTERN. ECAL gives one angle per sector and nothing else; HCAL (2026-09-15)
            # gave 0 / 22.5 / 45 deg with 45 on sectors 4,12,20 (= 4 mod 8) and 0 on 8,16
            # (= 0 mod 8) -- i.e. possibly 8-fold with ALTERNATING orientations, of which the
            # accepted 32 would be an alias. Printed so that is decidable from the log.
            deg = np.degrees(aa)
            clusters = np.unique(np.round(deg / 2.5) * 2.5)
            print(f"    angle clusters (2.5deg): "
                  + ", ".join(f"{c:+.1f}x{int((np.abs(deg - c) < 1.25).sum())}" for c in clusters))
            if best["NS"] % 8 == 0:
                per = best["NS"] // 8
                byres = {}
                for (ang_, _c, _n), sb in zip(best["angs"], best.get("secs", [])):
                    byres.setdefault(sb % 8, []).append(np.degrees(ang_))
                if byres:
                    print("    angle by (sector mod 8): " + "  ".join(
                        f"{k}:{np.median(v):+.1f}" for k, v in sorted(byres.items())))
            print(f"    ON-GRID: {best['ongrid']:.4f} within {a.tol} mm   "
                  f"median residual {np.median(res):.4f} mm, p95 {np.percentile(res,95):.4f} mm")
            drec["layers"][str(li)] = {
                "n_hits": int(ml.sum()), "n_distinct": int(len(P)), "saturation": round(sat, 4),
                "pitch": round(float(pitch), 5), "nn_harmonics": harm,
                "global_R": [round(Rx, 4), round(Ry, 4)],
                "best_sectors": int(best["NS"]), "n_bins": int(best["nbins"]),
                "post_fit_R": round(best["score"], 4),
                "module_angles_deg": [round(float(np.degrees(t[0])), 3) for t in best["angs"]],
                "ongrid_frac": round(float(best["ongrid"]), 4),
                "residual_med": round(float(np.median(res)), 5),
                "residual_p95": round(float(np.percentile(res, 95)), 5)}
        out["detectors"][str(d)] = drec

    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {a.out}")
    print("READ THE 'ON-GRID' LINE FIRST: the longitudinal derivation cleared 1.0000; a transverse "
          "number far below that means the lattice model is WRONG, not merely imprecise, and a "
          "per-module LUT built on it would snap cells to sites that do not exist.")


if __name__ == "__main__":
    main()
