"""Fixed-initial-momentum helix geometry for the tracker anchor (arXiv:2512.24254-style, but the
trajectory is computed ONCE from initial conditions and only evaluated — never recomputed/fed back).

Transverse motion in a solenoidal field (B along z) is a circle of radius R = pT/(0.3|q|B). z is
linear in the transverse arc length s: z = vz + s*sinh(eta). B calibrated from data (~3.07 T; see
scripts/helix_prebuild.py). All lengths in mm, pT in GeV, B in T.
"""
from __future__ import annotations
import numpy as np

B_FIELD = 3.07   # T, calibrated from real tracks (helix_prebuild.py: implied-B median, IQR 9%)


def circle_params(pT, phi0, q, vx, vy, B=B_FIELD, sign=-1.0):
    """Transverse trajectory circle from initial momentum. Returns center (cx,cy) [mm], R [mm].
    `sign` (±1) selects the curvature side; sign=-1 calibrated against data (helix_prebuild:
    fixed-momentum transverse residual 2.4mm median vs 108mm for +1)."""
    pT = np.asarray(pT, float); phi0 = np.asarray(phi0, float); q = np.asarray(q, float)
    R = pT / (0.3 * np.clip(np.abs(q), 0.5, None) * B) * 1000.0      # mm
    s = sign * np.sign(q)
    nx, ny = -np.sin(phi0), np.cos(phi0)                            # +90° from momentum direction
    cx = np.asarray(vx, float) + R * s * nx
    cy = np.asarray(vy, float) + R * s * ny
    return cx, cy, R


def cross_at_r(cx, cy, R, r):
    """Both intersections of the detector circle |p|=r with the trajectory circle (center c, rad R).
    Returns (x1,y1), (x2,y2), valid mask. Shapes broadcast; r may differ per element."""
    cx, cy, R, r = map(lambda a: np.asarray(a, float), (cx, cy, R, r))
    c2 = cx * cx + cy * cy
    cnorm = np.sqrt(np.clip(c2, 1e-9, None))
    k = (c2 - R * R + r * r) / 2.0
    d = k / cnorm
    disc = r * r - d * d
    valid = disc >= 0
    h = np.sqrt(np.clip(disc, 0.0, None))
    fx, fy = (k / c2) * cx, (k / c2) * cy          # foot of perpendicular from origin to the chord line
    ex, ey = -cy / cnorm, cx / cnorm               # unit vector along the chord line
    return (fx + h * ex, fy + h * ey), (fx - h * ex, fy - h * ey), valid


def arc_length(cx, cy, R, vx, vy, x, y):
    """Transverse arc length from vertex to point (x,y) along the circle (shorter sweep)."""
    a0 = np.arctan2(np.asarray(vy) - cy, np.asarray(vx) - cx)
    a1 = np.arctan2(np.asarray(y) - cy, np.asarray(x) - cx)
    da = np.abs(np.arctan2(np.sin(a1 - a0), np.cos(a1 - a0)))       # |Δangle| in [0,π]
    return R * da


def layer_references(pT, phi0, eta, q, vx, vy, vz, B=B_FIELD, n_s=512, s_cap=4000.0):
    """Expected helix (x,y,z) at every layer — (B,48,3) — by MARCHING arc length s from the vertex
    (unambiguous: the FIRST time the one fixed trajectory reaches each layer). Barrel: first s where
    r(s) crosses the layer's mean radius (rising edge). Endcap: z is monotonic in s, so s is exact.
    Layers the track never reaches fall back to the last-marched point (harmless: the particle's real
    hits aren't there). Computed once per track — no per-layer crossing to disambiguate."""
    from genpu.detector_geometry import LAYER_MEAN_R, LAYER_MEAN_Z, LAYER_IS_BARREL
    pT, phi0, eta, q, vx, vy, vz = map(lambda a: np.asarray(a, float), (pT, phi0, eta, q, vx, vy, vz))
    Bn = pT.shape[0]
    cx, cy, R = circle_params(pT, phi0, q, vx, vy, B)               # (B,)
    w = (-1.0) * np.sign(q)                                         # motion direction (sign=-1 calib)
    a0 = np.arctan2(vy - cy, vx - cx)                               # (B,) vertex angle around center
    sh = np.sinh(eta)                                               # (B,)
    smax = np.minimum(np.pi * R, s_cap)                             # half a turn, capped
    s = (np.linspace(0.0, 1.0, n_s)[None, :]) * smax[:, None]       # (B,n_s)
    a = a0[:, None] + w[:, None] * s / R[:, None]
    xg = cx[:, None] + R[:, None] * np.cos(a)
    yg = cy[:, None] + R[:, None] * np.sin(a)
    rg = np.hypot(xg, yg)                                           # (B,n_s)
    zg = vz[:, None] + s * sh[:, None]

    rL = LAYER_MEAN_R.astype(float); zL = LAYER_MEAN_Z.astype(float); isbar = LAYER_IS_BARREL
    out = np.zeros((Bn, 48, 3), np.float32)
    for L in range(48):
        if isbar[L]:
            reached = rg >= rL[L]                                   # (B,n_s)
            j = np.argmax(reached, axis=1)                          # first True (0 if never)
            j = np.clip(j, 1, n_s - 1)
            # linear interp in s between j-1 and j where r crosses rL
            r0, r1 = rg[np.arange(Bn), j - 1], rg[np.arange(Bn), j]
            frac = np.clip((rL[L] - r0) / np.where(np.abs(r1 - r0) < 1e-6, 1e-6, r1 - r0), 0.0, 1.0)
            sc = s[np.arange(Bn), j - 1] + frac * (s[np.arange(Bn), j] - s[np.arange(Bn), j - 1])
        else:
            sc = (zL[L] - vz) / np.where(np.abs(sh) < 1e-3, np.sign(sh) * 1e-3 + 1e-9, sh)
            sc = np.clip(sc, 0.0, smax)
        ac = a0 + w * sc / R
        out[:, L, 0] = cx + R * np.cos(ac)
        out[:, L, 1] = cy + R * np.sin(ac)
        out[:, L, 2] = vz + sc * sh
    return out


def helix_at_z(pT, phi0, eta, q, vx, vy, vz, z, B=B_FIELD, sign=-1.0):
    """Expected helix crossing at plane z=z (ENDCAP layers). Solve arc length from z, then the
    transverse position on the circle. Returns (x,y,r). sinh(eta) clipped to avoid /0 near eta=0."""
    cx, cy, R = circle_params(pT, phi0, q, vx, vy, B, sign)
    sh = np.sinh(np.asarray(eta, float))
    s = (np.asarray(z, float) - np.asarray(vz, float)) / np.where(np.abs(sh) < 1e-3, np.sign(sh) * 1e-3 + 1e-9, sh)
    s = np.clip(s, 0.0, np.pi * R)                          # forward arc, at most half a turn
    a0 = np.arctan2(np.asarray(vy) - cy, np.asarray(vx) - cx)
    # sweep direction: sign*sign(q), the SAME convention as helix_at_r and layer_references.
    # This was -sign*sign(q) until 2026-08-14, i.e. backwards; measured on 80k real endcap tracker
    # hits (|z|>1200mm, outermost hit of each track), the old sign gives median |dphi| 0.633 rad
    # and |dr| 33mm against the true hit, the correct one 0.016 rad and 8.9mm.
    w = sign * np.sign(np.asarray(q, float))
    a = a0 + w * s / R
    x = cx + R * np.cos(a); y = cy + R * np.sin(a)
    return x, y, np.hypot(x, y)


def helix_at_r(pT, phi0, eta, q, vx, vy, vz, r, B=B_FIELD, sign=-1.0, pick="outgoing"):
    """Expected helix crossing at detector radius r. Returns (x,y,z,valid).
    pick='outgoing' takes the crossing at the smaller arc length from the vertex; 'closest' needs a
    reference and is handled by the caller. z = vz + s*sinh(eta)."""
    cx, cy, R = circle_params(pT, phi0, q, vx, vy, B, sign)
    (x1, y1), (x2, y2), valid = cross_at_r(cx, cy, R, r)
    # pick the OUTGOING crossing = the one further along the INITIAL momentum direction (proj on
    # (cos phi0, sin phi0)). Robust ~82% vs the backward mirror; the disagreements are low-pT curly
    # tracks near their turning radius where the two crossings ~coincide anyway. (Signed sweep was
    # coin-flip; forward-momentum projection is the reliable rule.)
    cph, sph = np.cos(np.asarray(phi0, float)), np.sin(np.asarray(phi0, float))
    take1 = (x1 * cph + y1 * sph) >= (x2 * cph + y2 * sph)
    x = np.where(take1, x1, x2); y = np.where(take1, y1, y2)
    a0 = np.arctan2(np.asarray(vy) - cy, np.asarray(vx) - cx)
    a1 = np.arctan2(y - cy, x - cx)
    w = sign * np.sign(np.asarray(q, float))
    s = R * np.mod(w * (a1 - a0), 2 * np.pi)                        # forward arc length to that crossing
    z = np.asarray(vz, float) + s * np.sinh(np.asarray(eta, float))
    return x, y, z, valid
