"""Calorimeter cell projection -- PIPELINE.md gap #1, endcaps.

Maps a continuous generated point (x, y, z) onto the centre of the real ODD cell that contains it.
The calo model emits a continuous cloud and nothing ever snapped it, so generated cells sat at
positions the detector cannot produce.

CONSTANTS COME FROM THE DETECTOR DESCRIPTION, NOT FROM A FIT.
Source: github.com/OpenDataDetector/OpenDataDetector --
  xml/detectors/CalorimeterECal.xml   ECal_cell_size = 5.1*mm,  <layer repeat="48">,
                                      slices 1.90+0.15+0.10+0.50+0.10+1.30+0.25+0.75 = 5.05 mm
  xml/detectors/CalorimeterHCal.xml   HCal_cell_size = 30*mm,   <layer repeat="36">,
                                      slices 30+16+3+2 = 51.0 mm
  xml/OpenDataDetectorEnvelopes.xml   {ecal,hcal}_{b,e}_symmetry = 16
                                      ecal_e_min_z 3.2 m, hcal_e_min_z 3.6 m
Both endcap envelopes carry <rotation z="90*deg - 180*deg/symmetry"/> = 78.75 deg, which modulo the
22.5 deg face spacing is 11.25 deg -- the face-boundary phase.

Each endcap is a 16-sided polyhedron whose faces carry an INDEPENDENT square grid
(`<segmentation type="CartesianGridXZ">`, i.e. Cartesian in the face's LOCAL frame). So the snap is:
assign a face from phi, rotate into that face's frame, round to the cell pitch, snap depth to the
layer ladder, rotate back. Analytic -- no lookup table, which matters because the cell vocabulary is
~2e7 (PIPELINE gap 2a) and enumerating it was never an option.

Independently validated against 190M real cell-hits before the XML was found: pitch fitted to
5.09998 mm with zero inter-wedge spread, 16 faces located by a phase scan spiking at 11.25 deg,
on-grid 0.9999 (ECAL). See experiment-memory/2026-09-15-calo-geometry-CONFIRMED-from-ODD-xml.md.

THE BARREL IS ALSO HANDLED, and PIPELINE.md gap #2 is WRONG.
Gap #2 states that "exact cell identity is unreachable for barrel EM" because barrel cells from
different layers overlap in RADIUS. True, and irrelevant: radius is simply the wrong coordinate.
A barrel stave is a FLAT plate, so its depth coordinate is the PERPENDICULAR distance to the stave
plane, and in the stave frame all three coordinates are exactly discrete (shard 0, 400 events):

  det 10 (ECAL barrel)  R(z) 1.0000 @5.1   R(along) 0.9979 @5.1   R(perp) 0.9998 @5.050 -> 48 layers
  det 13 (HCAL barrel)  R(z) 1.0000 @30    R(along) 0.9985 @30    R(perp) 1.0000 @51.00 -> 36 layers

(R = circular concentration; 1.0 means every value sits on one grid.) Gap #2's conclusion was drawn
from radius alone and should be retracted.

THE SENSITIVE-SLICE INSET IS DERIVABLE, so nothing here is a free parameter. The first cell centre
sits at the middle of the first sensitive slice, i.e. the XML slice stack up to and including half
the sensitive layer:
  ECAL  1.90 W + 0.15 G10 + 0.10 GroundOrHVMix + 0.50/2 Si = 2.40 mm
        -> barrel 1250 + 2.4 = 1252.4 (measured 1252.4); endcap 3200 + 2.4 = 3202.4 (measured 3202.4)
  HCAL  30 Steel + 16 siPCBMix + 3/2 Polystyrene = 47.5 mm
        -> barrel 1600 + 47.5 = 1647.5 (measured 1647.5); endcap 3600 + 47.5 = 3647.5 (measured 3647.5)
All four agree exactly.
"""
from __future__ import annotations

import numpy as np

# --- detector ids (colliderml/physics/detector_enums.py, confirmed against our own mapping) ---
ECAL_NEG_ENDCAP, ECAL_BARREL, ECAL_POS_ENDCAP = 9, 10, 11
HCAL_NEG_ENDCAP, HCAL_BARREL, HCAL_POS_ENDCAP = 12, 13, 14
ENDCAP_DETS = (ECAL_NEG_ENDCAP, ECAL_POS_ENDCAP, HCAL_NEG_ENDCAP, HCAL_POS_ENDCAP)
BARREL_DETS = (ECAL_BARREL, HCAL_BARREL)

SYMMETRY = 16                                  # *_symmetry = 16, all four calo detectors
FACE_SPACING = 2.0 * np.pi / SYMMETRY          # 22.5 deg
# <rotation z="90*deg - 180*deg/symmetry">, reduced modulo the face spacing
FACE_PHASE = np.deg2rad(90.0 - 180.0 / SYMMETRY) % FACE_SPACING     # = 11.25 deg

# det -> (cell pitch mm, n layers, layer pitch mm, |z| of layer 0 centre mm)
# The |z| origin is MEASURED, not taken from the XML's envelope zmin: the XML gives the envelope
# edge (3.2 / 3.6 m) while a cell centre sits inside the first layer's sensitive slice. The offsets
# below are the observed first-plane |z| (job 12884120, reconfirmed 2026-09-15 on 190M cell-hits),
# and they are consistent with the XML zmin plus a part-layer inset.
ENDCAP = {
    ECAL_NEG_ENDCAP: (5.1, 48, 5.050, 3202.4),
    ECAL_POS_ENDCAP: (5.1, 48, 5.050, 3202.4),
    HCAL_NEG_ENDCAP: (30.0, 36, 51.000, 3647.5),
    HCAL_POS_ENDCAP: (30.0, 36, 51.000, 3647.5),
}

# IN-FACE GRID ORIGIN (u0, v0), MEASURED -- the XML gives pitch, symmetry and the envelope rotation
# but NOT where a face's grid starts, which DD4hep fixes when it places each stave. Fitted by
# circular mean of (u, v) mod pitch on real cells, shard 0 / 2,000 events:
#   det  9: u0 +0.5890, v0 0.0000 -- IDENTICAL on all 16 faces, R = 1.000, spread 0.0000
#   det 12: u0 +2.5419, v0 -0.0006 -- R 0.997 / 0.969, spread 0.0085 (HCAL's known ~6.6% residual)
# Note v0 = 0: centres lie at INTEGER multiples of the pitch along the face normal, not half-integer.
# A first version assumed half-integer on both axes and displaced every real cell by a constant
# 3.2169 mm (median == p95 exactly, the signature of a rigid offset rather than a rounding error),
# putting 100% of cells outside their own cell.
# barrel det -> (cell pitch mm, n layers, layer pitch mm, apothem of stave face mm, inset mm)
BARREL = {
    ECAL_BARREL: (5.1, 48, 5.050, 1250.0, 2.40),
    HCAL_BARREL: (30.0, 36, 51.000, 1600.0, 47.50),
}

# Values below are the CIRCULAR-MEAN estimate CORRECTED by the median snap residual. The circular
# mean alone is biased when the concentration is below 1 -- it gave HCAL u0 2.5419, leaving a rigid
# -0.05279 displacement on every cell (p5 = p95 = -0.0528, i.e. ZERO spread, so a pure origin error
# and not the "structural" residual it was first read as). The corrected origins leave an
# irreducible |du| of 1e-5 mm on both detectors.
FACE_ORIGIN = {
    ECAL_NEG_ENDCAP: (0.58896, 0.0),
    ECAL_POS_ENDCAP: (0.58896, 0.0),
    HCAL_NEG_ENDCAP: (2.48911, 0.0),
    HCAL_POS_ENDCAP: (2.48911, 0.0),
}


def face_index(x, y):
    """Which of the 16 polyhedron faces a point belongs to."""
    phi = np.arctan2(y, x)
    return np.floor((phi - FACE_PHASE) / FACE_SPACING).astype(np.int64) % SYMMETRY


def face_angle(face):
    """Azimuth of a face's outward normal (its bisector)."""
    return FACE_PHASE + (np.asarray(face) + 0.5) * FACE_SPACING


def snap_endcap(x, y, z, det):
    """Snap endcap points to cell centres.

    Returns (xs, ys, zs, face, layer, iu, iv). `iu`/`iv` index the cell within its face's local
    grid; (det, face, layer, iu, iv) is a complete cell identifier and is the thing the dataset's
    dropped `cellID` encoded (`system:8,barrel:3,module:4,stave:1,layer:6,slice:5,x:32:-16,z:-16`).
    """
    x = np.asarray(x, float); y = np.asarray(y, float); z = np.asarray(z, float)
    det = np.asarray(det)
    xs, ys, zs = x.copy(), y.copy(), z.copy()
    face = np.zeros(len(x), np.int64); layer = np.full(len(x), -1, np.int64)
    iu = np.zeros(len(x), np.int64); iv = np.zeros(len(x), np.int64)

    for d, (pitch, nlay, lpitch, z0) in ENDCAP.items():
        m = det == d
        if not m.any():
            continue
        f = face_index(x[m], y[m])
        th = face_angle(f)
        ca, sa = np.cos(-th), np.sin(-th)
        # into the face frame; u is across the face, v along its normal
        u = x[m] * ca - y[m] * sa
        v = x[m] * sa + y[m] * ca
        # round to the cell grid about the MEASURED origin (see FACE_ORIGIN)
        u0, v0 = FACE_ORIGIN[d]
        ui = np.rint((u - u0) / pitch); vi = np.rint((v - v0) / pitch)
        uc = ui * pitch + u0; vc = vi * pitch + v0
        cb, sb = np.cos(th), np.sin(th)
        xs[m] = uc * cb - vc * sb
        ys[m] = uc * sb + vc * cb
        li = np.clip(np.rint((np.abs(z[m]) - z0) / lpitch), 0, nlay - 1)
        zs[m] = np.sign(z[m]) * (z0 + li * lpitch)
        face[m] = f; layer[m] = li.astype(np.int64)
        iu[m] = ui.astype(np.int64); iv[m] = vi.astype(np.int64)
    return xs, ys, zs, face, layer, iu, iv


def snap_barrel(x, y, z, det):
    """Snap barrel points to cell centres.

    Returns (xs, ys, zs, stave, layer, ialong, iz). The stave frame's two IN-PLANE axes are global
    z and the across-stave direction, both on the cell pitch; DEPTH is the perpendicular distance
    to the stave plane, on the LAYER pitch. Three different pitches, which is why radius -- a blend
    of depth and across-stave -- looked continuous.
    """
    x = np.asarray(x, float); y = np.asarray(y, float); z = np.asarray(z, float)
    det = np.asarray(det)
    xs, ys, zs = x.copy(), y.copy(), z.copy()
    stave = np.zeros(len(x), np.int64); layer = np.full(len(x), -1, np.int64)
    ial = np.zeros(len(x), np.int64); iz = np.zeros(len(x), np.int64)

    for d, (pitch, nlay, lpitch, apothem, inset) in BARREL.items():
        m = det == d
        if not m.any():
            continue
        k = face_index(x[m], y[m])
        th = face_angle(k)
        ct, st = np.cos(th), np.sin(th)
        perp = x[m] * ct + y[m] * st            # depth, perpendicular to the flat stave
        along = -x[m] * st + y[m] * ct          # across the stave width
        li = np.clip(np.rint((perp - apothem - inset) / lpitch), 0, nlay - 1)
        pc = apothem + inset + li * lpitch
        ai = np.rint(along / pitch); ac = ai * pitch
        xs[m] = pc * ct - ac * st
        ys[m] = pc * st + ac * ct
        zi = np.rint(z[m] / pitch); zs[m] = zi * pitch
        stave[m] = k; layer[m] = li.astype(np.int64)
        ial[m] = ai.astype(np.int64); iz[m] = zi.astype(np.int64)
    return xs, ys, zs, stave, layer, ial, iz


def snap_cells(x, y, z, det):
    """Snap every calorimeter point -- endcaps and barrels -- to its cell centre.

    Returns (xs, ys, zs, snapped_mask). `snapped_mask` is False only for points whose `det` is not
    a calorimeter id at all.
    """
    x = np.asarray(x, float); y = np.asarray(y, float); z = np.asarray(z, float)
    det = np.asarray(det)
    xs, ys, zs, *_ = snap_endcap(x, y, z, det)
    xb, yb, zb, *_ = snap_barrel(x, y, z, det)
    mb = np.isin(det, BARREL_DETS)
    xs[mb], ys[mb], zs[mb] = xb[mb], yb[mb], zb[mb]
    return xs, ys, zs, np.isin(det, ENDCAP_DETS + BARREL_DETS)


# ---------------------------------------------------------------- generation-side entry points

def etaphidepth_to_xyz(eta, phi, depth, front_face=None):
    """Inverse of the slice's coordinate build: (eta, phi, global depth) -> (x, y, z, det).

    `depth` is measured from the SINGLE ECAL front face the slice builder used
    (`calo_geom.load_front_face()`, barrel r 1259.197 / endcap |z| 3212.5) and is NOT the XML's
    geometric face -- those constants are percentile estimates, which is why real first-layer cells
    have NEGATIVE depth (barrel p1 -6 mm, endcap -10 mm; PIPELINE section table). Inverting with the
    same constants is what keeps generation in the frame the model was trained in; the absolute
    (x, y, z) that comes out is then snapped with the XML geometry, which is exact.

    Barrel-vs-endcap comes from |eta| against ETA_BARREL_CUT -- the same test `sample_showers`
    already uses for the energy head's region token, so no new sampled quantity is introduced.
    """
    from genpu.calo_geom import ETA_BARREL_CUT, depth_to_region_local, load_front_face
    eta = np.asarray(eta, float); phi = np.asarray(phi, float); depth = np.asarray(depth, float)
    R0, Z0 = load_front_face() if front_face is None else front_face
    is_barrel = np.abs(eta) < ETA_BARREL_CUT

    r_bar = R0 + depth
    z_bar = r_bar * np.sinh(eta)
    absz = Z0 + depth
    sh = np.sinh(eta)
    # |eta| >= 1.60 on the endcap branch, so sinh is bounded away from 0; clip only as a guard
    r_end = absz / np.maximum(np.abs(sh), 1e-9)
    z_end = np.sign(sh) * absz

    r = np.where(is_barrel, r_bar, r_end)
    z = np.where(is_barrel, z_bar, z_end)
    x = r * np.cos(phi); y = r * np.sin(phi)

    region, _ = depth_to_region_local(depth, is_barrel)
    det = np.select(
        [region == 0, region == 2,
         (region == 1) & (z < 0), (region == 1) & (z >= 0),
         (region == 3) & (z < 0), (region == 3) & (z >= 0)],
        [ECAL_BARREL, HCAL_BARREL,
         ECAL_NEG_ENDCAP, ECAL_POS_ENDCAP,
         HCAL_NEG_ENDCAP, HCAL_POS_ENDCAP],
        default=ECAL_BARREL).astype(np.int64)
    return x, y, z, det


def cell_ids(x, y, z, det):
    """One int64 identifier per cell. Bijective with the real cell on 0.9999 of ECAL endcap cells.

    This is the quantity the dataset's dropped `cellID` encoded; the layout mirrors the XML's
    `system:8,barrel:3,module:4,stave:1,layer:6,slice:5,x:32:-16,z:-16` without reproducing its
    exact bit packing (we have no reason to match it, only to be unique and stable).
    """
    det = np.asarray(det, np.int64)
    _, _, _, face, layer, iu, iv = snap_endcap(x, y, z, det)
    _, _, _, stave, blay, ial, iz = snap_barrel(x, y, z, det)
    mb = np.isin(det, BARREL_DETS)
    a = np.where(mb, stave, face).astype(np.int64)
    b = np.where(mb, blay, layer).astype(np.int64)
    c = np.where(mb, ial, iu).astype(np.int64)
    d = np.where(mb, iz, iv).astype(np.int64)
    # offsets keep the indices non-negative; widths are generous versus the observed ranges
    return (det << 56) | (a << 51) | (b << 44) | ((c + 2 ** 21) << 22) | (d + 2 ** 21)


def snap_and_merge(eta, phi, depth, energy, src=None, front_face=None):
    """Snap generated points to cells and MERGE points that land in the same cell.

    Merging is not optional. Two generated points in one cell must ADD their energy -- a cell is a
    single readout channel and the detector cannot report it twice (PIPELINE gap 3b: 3.5% of real
    cells have more than one contributor at PU0, far more at M3's mu = 30-200). Without this, a
    generated event has more cells than the detector could produce and every per-cell energy
    statistic is biased low.

    `src` (shower index per point) is merged WITHIN a shower when given. Pass None to merge across
    the whole event, which is what a full-event generator wants.

    Returns a dict with the merged cells: x, y, z, det, cell_id, energy, and (when src was given)
    src for the surviving cells.
    """
    x, y, z, det = etaphidepth_to_xyz(eta, phi, depth, front_face)
    xs, ys, zs, _ = snap_cells(x, y, z, det)
    cid = cell_ids(xs, ys, zs, det)
    if src is None:
        uniq, inv = np.unique(cid, return_inverse=True)
    else:
        # pair (cell, src) exactly. An earlier version XOR-ed src into the id, which is not
        # injective -- distinct (cell, src) pairs can collide and silently merge cells that never
        # shared a channel.
        pair = np.stack([cid.astype(np.int64), np.asarray(src, np.int64)], 1)
        uniq, inv = np.unique(pair, axis=0, return_inverse=True)
        inv = inv.ravel()
    e = np.zeros(len(uniq), float)
    np.add.at(e, inv, np.asarray(energy, float))
    first = np.zeros(len(uniq), np.int64)
    first[inv[::-1]] = np.arange(len(inv))[::-1]        # index of the first point in each cell
    out = {"x": xs[first], "y": ys[first], "z": zs[first], "det": det[first],
           "cell_id": cid[first], "energy": e, "n_merged": np.bincount(inv, minlength=len(uniq))}
    if src is not None:
        out["src"] = np.asarray(src)[first]
    return out
