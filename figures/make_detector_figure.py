"""ColliderML detector r-z cross-section for slides.

Geometry is READ FROM THE CODE, not invented:
  tracker  -> genpu.detector_geometry.LAYER_GEOMETRY  (48 (volume,layer) classes)
  calo     -> calo_geometry.json  (per-detector r/|z| p5..p95, built by build_calo_geometry.py)
  B field  -> genpu.helix.B_FIELD

Emits figures/detector_rz.svg (converted to PNG by rsvg-convert).
"""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from genpu.detector_geometry import LAYER_GEOMETRY          # noqa: E402
from genpu.helix import B_FIELD                              # noqa: E402

GEO = json.loads((ROOT / "calo_geometry.json").read_text())

# ----------------------------------------------------------------- canvas / scaling
W = 1900
ML, MR, MT, MB = 100, 40, 112, 96
ZMAX, RMAX = 5750.0, 3100.0                # mm, half-ranges of the drawn frame
PW = W - ML - MR
S = PW / (2 * ZMAX)                        # ONE scale for r and z -> the figure is to scale
PH = 2 * RMAX * S
H = int(round(PH + MT + MB))
CX = ML + PW / 2
CY = MT + PH / 2

def X(z): return CX + z * S
def Y(r): return CY - r * S                # +r up, -r down (mirrored cross-section)

# ----------------------------------------------------------------- palette
BG      = "#ffffff"
INK     = "#12161c"
MUTED   = "#5b6673"
PIXEL   = "#1f6feb"
STRIP   = "#0e9488"
OUTER   = "#7c5cd6"
ECAL_C  = "#e8871a"
HCAL_C  = "#c2384a"
TRACK_C = "#111827"

out = []
def add(s): out.append(s)

def rect(x0, y0, x1, y1, fill, stroke="none", sw=0, op=1.0, rx=0):
    add(f'<rect x="{min(x0,x1):.1f}" y="{min(y0,y1):.1f}" width="{abs(x1-x0):.1f}" '
        f'height="{abs(y1-y0):.1f}" rx="{rx}" fill="{fill}" fill-opacity="{op}" '
        f'stroke="{stroke}" stroke-width="{sw}"/>')

def band_rz(z0, z1, r0, r1, fill, stroke="none", sw=0, op=1.0):
    """Draw an (|z|,r) band mirrored into +r and -r."""
    for sgn in (+1, -1):
        rect(X(z0), Y(sgn * r0), X(z1), Y(sgn * r1), fill, stroke, sw, op)

def txt(x, y, s, size=26, fill=INK, weight="600", anchor="start", ff="DejaVu Sans, Helvetica, Arial, sans-serif", op=1.0):
    add(f'<text x="{x:.1f}" y="{y:.1f}" font-family="{ff}" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}" fill-opacity="{op}" text-anchor="{anchor}">{s}</text>')

# ----------------------------------------------------------------- tracker geometry from code
# barrel volumes: cylinders at fixed r; endcap volumes: discs at fixed z.
BARREL_HALF_Z = {17: 600.0, 24: 1290.0, 29: 1290.0}   # set by where that volume's endcap discs start
VOL_COLOR = {17: PIXEL, 16: PIXEL, 18: PIXEL,
             24: STRIP, 23: STRIP, 25: STRIP,
             29: OUTER, 28: OUTER, 30: OUTER}

barrels, discs = [], []
for (vol, lay), g in LAYER_GEOMETRY.items():
    mr, sr, mz, sz, *_ , is_barrel = g
    if is_barrel:
        barrels.append((vol, mr, BARREL_HALF_Z[vol]))
    else:
        discs.append((vol, mz, max(mr - 2.0 * sr, 25.0), mr + 2.0 * sr))

# ----------------------------------------------------------------- calo geometry from json
D = GEO["detectors"]
def det(i): return D[str(i)]
ECAL_BARREL = (det(10)["absz_p95"], det(10)["r_p5"], det(10)["r_p95"])
HCAL_BARREL = (det(13)["absz_p95"], det(13)["r_p5"], det(13)["r_p95"])
ECAL_ENDCAP = (det(9)["absz_p5"], det(9)["absz_p95"], det(9)["r_p5"], max(det(9)["r_p95"], det(11)["r_p95"]))
HCAL_ENDCAP = (det(12)["absz_p5"], det(12)["absz_p95"], det(12)["r_p5"], max(det(12)["r_p95"], det(14)["r_p95"]))
E_ECAL = det(9)["energy_frac"] + det(10)["energy_frac"] + det(11)["energy_frac"]
E_HCAL = det(12)["energy_frac"] + det(13)["energy_frac"] + det(14)["energy_frac"]

# ================================================================= draw
add(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
add(f'<rect width="{W}" height="{H}" fill="{BG}"/>')

# --- calorimeter blocks (drawn first, behind everything)
for z0, z1, r0, r1, c in [
    (0, ECAL_BARREL[0], ECAL_BARREL[1], ECAL_BARREL[2], ECAL_C),
    (0, HCAL_BARREL[0], HCAL_BARREL[1], HCAL_BARREL[2], HCAL_C),
    (ECAL_ENDCAP[0], ECAL_ENDCAP[1], ECAL_ENDCAP[2], ECAL_ENDCAP[3], ECAL_C),
    (HCAL_ENDCAP[0], HCAL_ENDCAP[1], HCAL_ENDCAP[2], HCAL_ENDCAP[3], HCAL_C),
]:
    for zs in (+1, -1):
        band_rz(zs * z0, zs * z1, r0, r1, c, stroke=c, sw=2.0, op=0.30)

# --- tracker volume envelope (soft grey box behind the layers)
band_rz(-3060, 3060, 25, 1035, "#9aa4b2", op=0.09)

# --- tracker barrel cylinders
for vol, r, hz in barrels:
    for sgn in (+1, -1):
        y = Y(sgn * r)
        add(f'<line x1="{X(-hz):.1f}" y1="{y:.1f}" x2="{X(hz):.1f}" y2="{y:.1f}" '
            f'stroke="{VOL_COLOR[vol]}" stroke-width="5" stroke-linecap="round"/>')

# --- tracker endcap discs
for vol, z, r0, r1 in discs:
    for sgn in (+1, -1):
        add(f'<line x1="{X(z):.1f}" y1="{Y(sgn*r0):.1f}" x2="{X(z):.1f}" y2="{Y(sgn*r1):.1f}" '
            f'stroke="{VOL_COLOR[vol]}" stroke-width="5" stroke-linecap="round"/>')

# --- beamline + interaction point
add(f'<line x1="{X(-ZMAX):.1f}" y1="{Y(0):.1f}" x2="{X(ZMAX):.1f}" y2="{Y(0):.1f}" '
    f'stroke="{MUTED}" stroke-width="2" stroke-dasharray="10 8"/>')
add(f'<circle cx="{X(0):.1f}" cy="{Y(0):.1f}" r="7" fill="{INK}"/>')
txt(X(0), Y(0) + 34, "IP", 22, INK, anchor="middle")

# --- example particles: tracker hits -> calorimeter shower
def show_particle(pts, shower_xy, n_blobs=11, seed=0, sig=(110.0, 70.0)):
    import random
    rnd = random.Random(seed)
    d = " ".join(("M" if i == 0 else "L") + f"{X(z):.1f},{Y(r):.1f}" for i, (z, r) in enumerate(pts))
    add(f'<path d="{d}" fill="none" stroke="{TRACK_C}" stroke-width="3.2" stroke-linecap="round"/>')
    for (z, r) in pts[1:]:
        add(f'<circle cx="{X(z):.1f}" cy="{Y(r):.1f}" r="4.4" fill="{TRACK_C}"/>')
    sz, sr = shower_xy
    for k in range(n_blobs):
        dz = rnd.gauss(0, sig[0]); dr = rnd.gauss(0, sig[1])
        add(f'<circle cx="{X(sz+dz):.1f}" cy="{Y(sr+dr):.1f}" r="{rnd.uniform(5,12):.1f}" '
            f'fill="{ECAL_C}" fill-opacity="0.9"/>')

# central track, gently bending, into the barrel ECAL
show_particle([(0, 0), (60, 170), (180, 400), (330, 660), (520, 900), (700, 1035), (900, 1259)],
              (1020, 1350), seed=3, sig=(150.0, 55.0))
# forward track into the endcap ECAL
show_particle([(0, 0), (330, 105), (900, 250), (1600, 400), (2400, 560), (3000, 660), (3212, 700)],
              (3300, 740), seed=11, sig=(55.0, 130.0))

# ----------------------------------------------------------------- axes
add(f'<line x1="{ML}" y1="{H-MB+30}" x2="{W-MR}" y2="{H-MB+30}" stroke="{MUTED}" stroke-width="2"/>')
for zm in range(-5, 6):
    x = X(zm * 1000)
    add(f'<line x1="{x:.1f}" y1="{H-MB+30}" x2="{x:.1f}" y2="{H-MB+40}" stroke="{MUTED}" stroke-width="2"/>')
    txt(x, H - MB + 66, f"{zm}", 22, MUTED, "500", "middle")
txt(W - MR, H - MB + 66, "z [m]", 24, MUTED, "700", "end")
add(f'<line x1="{ML-28}" y1="{Y(RMAX)}" x2="{ML-28}" y2="{Y(-RMAX)}" stroke="{MUTED}" stroke-width="2"/>')
for rm in (-2, -1, 0, 1, 2):
    y = Y(rm * 1000)
    add(f'<line x1="{ML-38}" y1="{y:.1f}" x2="{ML-28}" y2="{y:.1f}" stroke="{MUTED}" stroke-width="2"/>')
    txt(ML - 46, y + 8, f"{abs(rm)}", 22, MUTED, "500", "end")
add(f'<text x="{ML-70}" y="{CY}" font-family="DejaVu Sans, Helvetica, Arial, sans-serif" '
    f'font-size="24" font-weight="700" fill="{MUTED}" text-anchor="middle" '
    f'transform="rotate(-90 {ML-70} {CY})">r [m]</text>')

# ----------------------------------------------------------------- labels
txt(ML, 46, "ColliderML detector — r–z cross-section", 38, INK, "700")
txt(ML, 80, f"drawn to scale from the geometry the code uses  ·  solenoid field B = {B_FIELD} T",
    24, MUTED, "500")

def leader(*pts, color=MUTED):
    d = " ".join(("M" if i == 0 else "L") + f"{x:.1f},{y:.1f}" for i, (x, y) in enumerate(pts))
    add(f'<path d="{d}" stroke="{color}" stroke-width="2.2" fill="none"/>')
    add(f'<circle cx="{pts[-1][0]:.1f}" cy="{pts[-1][1]:.1f}" r="4.5" fill="{color}"/>')

def block(x, y, lines, gap=31):
    for i, (t, size, col, w) in enumerate(lines):
        txt(x, y + i * gap, t, size, col, w)

# ---- top band (r > 2.3 m is empty): ECAL left, HCAL right
block(X(-5450), Y(2880), [("ECAL", 33, ECAL_C, "700"),
                          ("5.05 mm layer pitch", 23, MUTED, "500"),
                          (f"{E_ECAL:.0%} of deposited energy", 23, MUTED, "500")], gap=33)
leader((X(-4050), Y(2800)), (X(-3200), Y(2800)), (X(-2600), Y(1500)), color=ECAL_C)

block(X(2900), Y(2880), [("HCAL", 33, HCAL_C, "700"),
                         ("51 mm layer pitch", 23, MUTED, "500"),
                         (f"{E_HCAL:.0%} of deposited energy", 23, MUTED, "500")], gap=33)
leader((X(2800), Y(2800)), (X(2200), Y(2800)), (X(1700), Y(2248)), color=HCAL_C)

# ---- bottom band: TRACKER left, legend centre, endcaps right
block(X(-5450), Y(-2380), [("TRACKER", 33, INK, "700"),
                           ("9 volumes \u2192 48 layer classes", 23, MUTED, "500"),
                           ("18,824 modules", 23, MUTED, "500"),
                           ("r \u2264 1.02 m,  |z| \u2264 3.0 m", 23, MUTED, "500")], gap=33)
leader((X(-4000), Y(-2450)), (X(-3300), Y(-2450)), (X(-2950), Y(-1080)), color=MUTED)

lx, ly = X(-1900), Y(-2380)
for i, (c, lab) in enumerate([(PIXEL, "pixel barrel + endcap"), (STRIP, "strip"), (OUTER, "outer tracker")]):
    add(f'<line x1="{lx:.1f}" y1="{ly + i*33 - 8:.1f}" x2="{lx+42:.1f}" y2="{ly + i*33 - 8:.1f}" '
        f'stroke="{c}" stroke-width="5" stroke-linecap="round"/>')
    txt(lx + 56, ly + i * 33, lab, 23, MUTED, "500")
add(f'<circle cx="{lx+21:.1f}" cy="{ly + 3*33 - 8:.1f}" r="7" fill="{TRACK_C}"/>')
txt(lx + 56, ly + 3 * 33, "one particle: tracker hits \u2192 calo shower", 23, MUTED, "500")

block(X(2900), Y(-2380), [("endcaps", 33, INK, "700"),
                          ("ECAL  |z| = 3.21\u20133.42 m", 23, MUTED, "500"),
                          ("HCAL  |z| = 3.65\u20135.33 m", 23, MUTED, "500")], gap=33)

# ---- ECAL/HCAL gap bracket, in the empty strip beyond the ECAL barrel end
gz = 3000.0
add(f'<line x1="{X(gz):.1f}" y1="{Y(1442):.1f}" x2="{X(gz):.1f}" y2="{Y(1649):.1f}" '
    f'stroke="{INK}" stroke-width="3"/>')
for rr in (1442, 1649):
    add(f'<line x1="{X(gz)-10:.1f}" y1="{Y(rr):.1f}" x2="{X(gz)+10:.1f}" y2="{Y(rr):.1f}" '
        f'stroke="{INK}" stroke-width="3"/>')
txt(X(gz) + 24, Y(1545) - 4, "physical gap", 23, INK, "700")
txt(X(gz) + 24, Y(1545) + 26, "cell log-E steps 3\u00d7", 23, MUTED, "500")

add("</svg>")
p = ROOT / "figures" / "detector_rz.svg"
p.write_text("\n".join(out))
print("wrote", p)
