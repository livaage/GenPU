"""Calo front-face geometry and the truth-helix CORE ANCHOR (Phase 1).

The shower core (centroid offset from the particle direction) carries 83-93% of the "shower width"
variance the event gate reacts to, and it is a ~1.5 m magnetic-bending displacement that the
GlobalHead mixture was being asked to predict from vertex kinematics. Phase 0b measured how much of
it a truth-helix extrapolation to the calo front face explains (`scripts/calo_helix_core_probe.py`):
phi tightening 25.5x (proton), 11.2x (e+-), 4.6x (pion), 2.9x (photon, where the whole effect is
vertex displacement). All GO.

So: store the core as a RESIDUAL from that prediction and add the anchor back at generation. The
anchor is a deterministic function of TRUTH conditioning only (pT, eta, phi, charge, vertex) — the
same status as the existing `cont` features — so there is no exposure-bias risk, unlike conditioning
on sampled quantities (which failed on 2026-08-13).

This module is the single implementation: the slice builder, the metrics/eval path and the probe all
call `core_anchor()`, so the training frame and the generation frame cannot drift apart.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

from genpu.helix import helix_at_r, helix_at_z, circle_params

DEFAULT_GEOMETRY = str(Path(__file__).resolve().parents[2] / "calo_geometry.json")


def wrap_pi(d):
    return (d + np.pi) % (2 * np.pi) - np.pi


def load_front_face(path: str | None = None):
    """(barrel_r [mm], endcap_|z| [mm]) of the calo front face, from build_calo_geometry.py.

    NOTE this is a SINGLE face for the whole calorimeter — det 10 for the barrel, det 9 for the
    endcap, i.e. the ECAL. Depth measured from it runs continuously across the ECAL/HCAL boundary
    and cannot say which side a cell is on; see `load_det_front_faces` and job 13033802.
    """
    g = json.loads(Path(path or DEFAULT_GEOMETRY).read_text())["front_face"]
    return float(g["barrel_r"]), float(g["endcap_absz"])


# ---------------------------------------------------------------- calorimeter sectioning
# Measured 2026-08-27 (job 13033802, `scripts/calo_section_split.py`). The calorimeter is TWO
# detectors and a cell's energy scale depends on which:
#
#   region        det    depth p1..p99 (mm)   <logE>   layer pitch   E frac
#   barrel ECAL    10      -6 ..  228         -7.97     5.050 mm      0.161
#   endcap ECAL   9,11     -10 ..  227        -8.07     5.050 mm      0.559
#   barrel HCAL    13     388 .. 1446         -6.86    51.000 mm      0.008
#   endcap HCAL  12,14    435 .. 2220         -6.89    51.000 mm      0.272
#
# Mean cell log-E STEPS by +1.12 (3.06x) across the physical gap, because an HCAL cell integrates
# ~10x more material. That step is deterministic geometry the model should be GIVEN, not made to
# infer — and it cannot infer it, because barrel and endcap overlap on the single-face depth axis
# while transitioning at DIFFERENT depths (388 vs 435 mm).
ECAL_DETS = (9, 10, 11)
HCAL_DETS = (12, 13, 14)
BARREL_DETS = (10, 13)
N_CALO_REGIONS = 4


def load_det_front_faces(path: str | None = None):
    """{det: (is_barrel, face)} — face is r [mm] for barrel dets, |z| [mm] for endcap dets.

    Same p5 convention as `front_face` (which is just det 10 / det 9 of this table), so
    depth-within-section is directly comparable to the existing global depth for ECAL cells.
    """
    g = json.loads(Path(path or DEFAULT_GEOMETRY).read_text())["detectors"]
    return {int(k): (bool(v["is_barrel"]),
                     float(v["r_p5"] if v["is_barrel"] else v["absz_p5"]))
            for k, v in g.items()}


def load_region_boundaries(path: str | None = None):
    """{is_barrel: (ecal_back, hcal_front, hcal_offset)} in GLOBAL-depth mm.

    `ecal_back` is the ECAL's p95 extent, `hcal_front` the HCAL's front face, both expressed on the
    single-front-face depth axis `points_flat[:, 2]` uses. `hcal_offset` = hcal_front - ecal_front is
    exactly the constant that separates `depth` from `depth_local` for HCAL cells (verified on the
    rebuilt slices: 389.396 mm barrel, 435.000 mm endcap, spread < 3e-4 mm).
    """
    g = json.loads(Path(path or DEFAULT_GEOMETRY).read_text())["detectors"]
    out = {}
    for is_barrel, ecal_d, hcal_d in ((True, 10, 13), (False, 9, 12)):
        key = "r" if is_barrel else "absz"
        ef = float(g[str(ecal_d)][f"{key}_p5"])
        out[is_barrel] = (float(g[str(ecal_d)][f"{key}_p95"]) - ef,
                          float(g[str(hcal_d)][f"{key}_p5"]) - ef,
                          float(g[str(hcal_d)][f"{key}_p5"]) - ef)
    return out


# Per-CELL barrel cut for the generation path, fitted on `multispecies_v2_h5` (41.65M cells,
# 2026-08-27). Sits BELOW the geometric eta_transition (1.666) because cells spread around the
# shower core, so the effective boundary for a CELL is lower than for the incident particle.
#   |eta| < 1.60  ->  barrel accuracy 0.9951, REGION accuracy 0.9951, depth_local RMS err 1.2 mm
# SECTION (ECAL vs HCAL) accuracy is 1.00000 at EVERY threshold tried, because both midpoints fall
# inside the physical gap where no real cell exists -- so the 3x sampling step, the thing this
# conditioning exists to convey, is assigned exactly regardless of the barrel guess.
ETA_BARREL_CUT = 1.60


def depth_to_region_local(depth, is_barrel, path: str | None = None):
    """GENERATION-side inverse of the slice's section fields: (region token, depth_local).

    The model samples a continuous GLOBAL depth and no detector id, but region and depth-within-
    section are a DETERMINISTIC function of that depth plus barrel-vs-endcap -- which generation
    already has, as the anchor branch. So no new sampled quantity is needed and the generated frame
    matches the training frame by construction, the same reason the anchor itself lives here.

    Cells landing in the PHYSICAL GAP between the sections (barrel 183..389 mm, endcap 207..435 mm)
    are assigned by midpoint. Real cells cannot be there; generated ones can, because the point flow
    is continuous and knows nothing about the gap.
    """
    depth = np.asarray(depth, np.float64)
    is_barrel = np.asarray(is_barrel, bool)
    B = load_region_boundaries(path)
    mid = np.where(is_barrel, 0.5 * (B[True][0] + B[True][1]), 0.5 * (B[False][0] + B[False][1]))
    off = np.where(is_barrel, B[True][2], B[False][2])
    is_hcal = depth >= mid
    region = (2 * is_hcal + ~is_barrel).astype(np.int8)
    return region, (depth - np.where(is_hcal, off, 0.0)).astype(np.float32)


def det_region(det):
    """4-way region token: 0 barrel-ECAL, 1 endcap-ECAL, 2 barrel-HCAL, 3 endcap-HCAL.

    This is the token the energy head needs: it carries BOTH the sampling section (which sets the
    3x cell-energy step) and barrel-vs-endcap (which sets where along depth that step happens).
    """
    det = np.asarray(det, np.int64)
    return (2 * np.isin(det, HCAL_DETS) + ~np.isin(det, BARREL_DETS)).astype(np.int8)


def xyz_to_eta_phi(x, y, z):
    r = np.hypot(x, y)
    theta = np.arctan2(np.maximum(r, 1e-9), z)
    return -np.log(np.maximum(np.tan(theta / 2), 1e-12)), np.arctan2(y, x)


def straight_line(pt, phi0, eta, vx, vy, vz, R, Z):
    """Neutral propagation: straight from the vertex along (eta, phi0). Barrel crossing if it
    happens before the endcap plane, else the endcap plane. Returns (x, y, z, hit_barrel, ok) —
    `hit_barrel` distinguishes the two surfaces, which matters because the branch is a conditioning
    feature for the GlobalHead (an earlier version labelled every straight line 'barrel')."""
    cph, sph = np.cos(phi0), np.sin(phi0)
    # |v_perp + t*(cph,sph)| = R  ->  t^2 + 2t(vx cph + vy sph) + (vr^2 - R^2) = 0
    b = vx * cph + vy * sph
    disc = np.maximum(b ** 2 - (vx ** 2 + vy ** 2 - R ** 2), 0.0)
    t_bar = -b + np.sqrt(disc)                                   # outgoing root
    sh = np.sinh(eta)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_end = (np.sign(sh) * Z - vz) / np.where(np.abs(sh) < 1e-9, np.nan, sh)
    t_end = np.where(np.isfinite(t_end) & (t_end > 0), t_end, np.inf)
    t_bar = np.where(t_bar > 0, t_bar, np.inf)
    hit_barrel = t_bar <= t_end
    t = np.minimum(t_bar, t_end)
    ok = np.isfinite(t)
    t = np.where(ok, t, 0.0)
    return vx + t * cph, vy + t * sph, vz + t * sh, hit_barrel, ok


def turning_point(pt, phi0, q, eta, vx, vy, vz):
    """(x, y, z) at the OUTERMOST radius the trajectory circle reaches, |c| + R.

    A soft charged particle (median pT 0.27 GeV, born at vr ~ 420 mm) curls back before r_calo and
    so has no face crossing at all — 63% of pion showers, 72% of e±. Their calo cells come from
    decay/secondary energy booked to the parent, which leaves from around where the parent stopped
    going outward. Measured on shard 0, anchoring those showers here instead of at the particle
    direction tightens sigma(core_phi) 2.4x (pion, proton) / 2.0x (e±) and sigma(core_eta) 4.6x.
    """
    cx, cy, Rc = circle_params(pt, phi0, q, vx, vy)
    ang = np.arctan2(cy, cx)                       # far point sits along the centre direction
    x = cx + Rc * np.cos(ang); y = cy + Rc * np.sin(ang)
    a0 = np.arctan2(vy - cy, vx - cx)
    w = -np.sign(q)                                # sweep direction, sign=-1 convention (helix.py)
    s = Rc * np.mod(w * (ang - a0), 2 * np.pi)     # forward arc from the vertex, at most one turn
    return x, y, vz + s * np.sinh(eta)


# anchor branch codes, reported so every run can attribute its result to a population
MODE_BARREL, MODE_ENDCAP, MODE_TURNING, MODE_NONE = 0, 1, 2, 3

# pT [GeV] above which kind="auto" switches a CHARGED particle from the helix to the straight line.
# Measured on e± (shard 0, 120k showers): the helix anchor's phi tightening runs 4.52x, 2.14x, 1.36x,
# 1.08x, 1.04x, 0.43x across pT bins of median 0.015 .. 0.254 GeV — it collapses and finally inverts —
# while the straight line runs 1.54x, 1.94x, 2.55x, 2.63x, 2.55x, 2.38x. They cross between the
# 0.028 and 0.047 GeV bins. Physically: a soft electron curls and deposits where it stops, a stiffer
# one radiates bremsstrahlung photons that fly STRAIGHT from the radiation point.
# The hybrid takes the better branch in every bin (4.49, 2.17, 2.55, 2.64, 2.55, 2.38) for a pooled
# 2.69x, vs 1.86x helix and 2.10x line.
#
# CALIBRATED ON e± AND MEANT FOR THEM. Bremsstrahlung is an electron effect (radiative loss goes as
# 1/m^2), and hadrons do not have it — a pion's median pT is 0.44 GeV, more than 10x this threshold,
# so kind="auto" would put essentially every pion on the straight line and throw away the helix's
# measured 15-19x face-branch gain. Use kind="helix" for hadrons.
AUTO_PT_SPLIT = 0.035


def helix_to_face(pt, phi0, eta, q, vx, vy, vz, R, Z, kind="helix", pt_split=AUTO_PT_SPLIT):
    """Extrapolate to the calo front face: barrel cylinder r=R unless the crossing is beyond the
    endcap plane |z|=Z, in which case the endcap plane; curlers that reach neither fall back to
    their turning point. Neutrals take the straight-line limit (same code path, q=0).

    `kind="line"` forces the straight-line path for EVERY particle, charge included. That is the
    bremsstrahlung hypothesis for e±: the deposit comes from photons radiated early, which travel
    straight from the radiation point, so the electron's own curved path is the wrong predictor —
    and measurably gets wronger the stiffer the track (Phase 1: anchor gain 4.52x at the softest
    pT bin inverting to 0.43x at the stiffest, and physical-frame over-dispersion rising to 1.93).

    `kind="auto"` takes the helix below `pt_split` and the line above it — the two regimes measured
    on e± (see AUTO_PT_SPLIT). Still a deterministic function of truth.

    Returns (eta, phi, mode) with mode in {0 barrel, 1 endcap, 2 turning point, 3 no anchor}.
    """
    pt, phi0, eta, q, vx, vy, vz = map(lambda a: np.asarray(a, np.float64),
                                       (pt, phi0, eta, q, vx, vy, vz))
    charged = np.abs(q) >= 0.5
    if kind == "helix":
        neutral = ~charged
    elif kind == "line":
        neutral = np.ones(len(pt), bool)
    elif kind == "auto":
        neutral = ~charged | (pt > pt_split)
    else:
        raise ValueError(f"unknown anchor kind {kind!r}")
    x = np.zeros_like(pt); y = np.zeros_like(pt); z = np.zeros_like(pt)
    mode = np.full(len(pt), MODE_NONE, np.int8)

    if neutral.any():
        m = neutral
        xx, yy, zz, bar, oo = straight_line(pt[m], phi0[m], eta[m], vx[m], vy[m], vz[m], R, Z)
        x[m], y[m], z[m] = xx, yy, zz
        # a straight line always reaches a surface (no turning point), but WHICH one still matters
        mode[m] = np.where(oo, np.where(bar, MODE_BARREL, MODE_ENDCAP), MODE_NONE)

    if (~neutral).any():
        m = ~neutral
        xb, yb, zb, vb = helix_at_r(pt[m], phi0[m], eta[m], q[m], vx[m], vy[m], vz[m], R)
        # barrel crossing valid only if it happens inside the endcap plane
        use_bar = vb & (np.abs(zb) <= Z)
        z_end = np.sign(np.sinh(eta[m])) * Z
        xe, ye, re = helix_at_z(pt[m], phi0[m], eta[m], q[m], vx[m], vy[m], vz[m], z_end)
        # endcap is the next fallback, and it is meaningful only if the track REACHES that plane.
        # helix_at_z clips the arc length to half a turn, so a curler whose turning radius never
        # gets it out to |z|=Z silently returns a point at the clipped arc, not at the plane —
        # which would hand the anchor a wildly wrong eta (measured: a 0.2 GeV track anchored at
        # eta+2.5). Recompute the required arc length and require it to fit inside that cap.
        _, _, Rc = circle_params(pt[m], phi0[m], q[m], vx[m], vy[m])
        sh_m = np.sinh(eta[m])
        s_need = (z_end - vz[m]) / np.where(np.abs(sh_m) < 1e-3, np.sign(sh_m) * 1e-3 + 1e-9, sh_m)
        use_end = ((~use_bar) & (s_need > 0) & (s_need <= np.pi * Rc)
                   & (re <= R * 1.05) & np.isfinite(re))
        xt, yt, zt = turning_point(pt[m], phi0[m], q[m], eta[m], vx[m], vy[m], vz[m])
        use_turn = ~(use_bar | use_end)
        xx = np.where(use_bar, xb, np.where(use_end, xe, xt))
        yy = np.where(use_bar, yb, np.where(use_end, ye, yt))
        zz = np.where(use_bar, zb, np.where(use_end, z_end, zt))
        x[m], y[m], z[m] = xx, yy, zz
        mm = np.where(use_bar, MODE_BARREL, np.where(use_end, MODE_ENDCAP, MODE_TURNING)).astype(np.int8)
        mm = np.where(use_turn & ~np.isfinite(xt + yt + zt), MODE_NONE, mm).astype(np.int8)
        mode[m] = mm

    he, hp = xyz_to_eta_phi(x, y, z)
    return he, hp, mode


def core_anchor(pt, phi0, eta, q, vx, vy, vz, R, Z, kind="helix", pt_split=AUTO_PT_SPLIT):
    """Core ANCHOR in the particle frame: (a_eta, a_phi, mode), where

        a_eta = eta_helix - eta_particle
        a_phi = wrap_pi(phi_helix - phi_particle)

    i.e. exactly the frame the slice's core lives in (`core = mean(cell - particle)`), so the
    anchored core is `core - anchor` and generation reconstructs `core = residual + anchor`.

    Where no extrapolation exists at all (mode 3) the anchor is ZERO — those showers fall back to
    today's particle-direction frame rather than being dropped, and `mode` is returned so both the
    builder and the generator can count each branch. The rule is a deterministic function of truth,
    so training and generation make the identical choice.
    """
    he, hp, mode = helix_to_face(pt, phi0, eta, q, vx, vy, vz, R, Z, kind=kind, pt_split=pt_split)
    ok = mode != MODE_NONE
    a_eta = np.where(ok, he - np.asarray(eta, np.float64), 0.0)
    a_phi = np.where(ok, wrap_pi(hp - np.asarray(phi0, np.float64)), 0.0)
    return a_eta, a_phi, mode
