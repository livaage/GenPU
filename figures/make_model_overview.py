"""One visual figure showing BOTH heads: transverse detector view + how each model builds its output.

Emits model_overview.svg (labelled) and model_overview_notext.svg (identical, zero text).
Geometry from genpu.detector_geometry.LAYER_GEOMETRY and calo_geometry.json; track arcs are real
circular trajectories for the field in genpu.helix.B_FIELD.
"""
import json, math, random, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from genpu.detector_geometry import LAYER_GEOMETRY          # noqa: E402
from genpu.helix import B_FIELD                              # noqa: E402

GEO = json.loads((ROOT / "calo_geometry.json").read_text())
FF = "DejaVu Sans, Helvetica, Arial, sans-serif"
INK, MUTED, LINE = "#12161c", "#5b6673", "#c9d1da"
BG, PANEL = "#ffffff", "#f7f9fb"
PIXEL, STRIP, OUTER = "#1f6feb", "#0e9488", "#7c5cd6"
ECAL_C, HCAL_C = "#e8871a", "#c2384a"
TRACK_C = "#12161c"

W, H = 1700, 880
CX, CY, S = 850.0, 470.0, 0.1378          # mm -> px for the transverse view

BARREL_R = sorted({g[0] for g in LAYER_GEOMETRY.values() if g[-1]})
PIX, STR, OUT = BARREL_R[:4], BARREL_R[4:8], BARREL_R[8:]
D = GEO["detectors"]
E_IN, E_OUT = D["10"]["r_p5"], D["10"]["r_p95"]
H_IN, H_OUT = D["13"]["r_p5"], D["13"]["r_p95"]


class Fig:
    def __init__(self, text=True):
        self.text = text
        self.o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
                  '<defs>']
        for n, c in [("m_ink", INK), ("m_mut", MUTED), ("m_or", ECAL_C), ("m_teal", STRIP),
                     ("m_pur", OUTER), ("m_blue", PIXEL)]:
            self.o.append(f'<marker id="{n}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
                          f'markerHeight="7" orient="auto-start-reverse"><path d="M0,1 L10,5 L0,9 z" fill="{c}"/></marker>')
        self.o += ['</defs>', f'<rect width="{W}" height="{H}" fill="{BG}"/>']

    def add(self, s): self.o.append(s)

    def t(self, x, y, s, size=22, fill=INK, weight="600", anchor="start"):
        if not self.text: return
        self.o.append(f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FF}" font-size="{size}" '
                      f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{s}</text>')

    def card(self, x, y, w, h, fill=PANEL, stroke=LINE, sw=2, rx=16):
        self.add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" '
                 f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')

    def circ(self, x, y, r, fill="none", stroke="none", sw=2, op=1.0, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.2f}" fill="{fill}" fill-opacity="{op}" '
                 f'stroke="{stroke}" stroke-width="{sw}"{d}/>')

    def path(self, d, col=MUTED, sw=3, mk=None, dash=None, fill="none", op=1.0):
        m = f' marker-end="url(#{mk})"' if mk else ""
        ds = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<path d="{d}" fill="{fill}" fill-opacity="{op}" stroke="{col}" stroke-width="{sw}"{m}{ds}/>')

    def arrow(self, x0, y0, x1, y1, col=MUTED, mk="m_mut", sw=3, dash=None):
        self.path(f"M{x0:.1f},{y0:.1f} L{x1:.1f},{y1:.1f}", col, sw, mk, dash)

    def ring(self, r_in, r_out, col, op=0.22):
        """filled annulus in the transverse view, via even-odd fill"""
        ri, ro = r_in * S, r_out * S
        d = (f"M{CX-ro:.1f},{CY:.1f} a{ro:.1f},{ro:.1f} 0 1,0 {2*ro:.1f},0 a{ro:.1f},{ro:.1f} 0 1,0 {-2*ro:.1f},0 Z "
             f"M{CX-ri:.1f},{CY:.1f} a{ri:.1f},{ri:.1f} 0 1,1 {2*ri:.1f},0 a{ri:.1f},{ri:.1f} 0 1,1 {-2*ri:.1f},0 Z")
        self.add(f'<path d="{d}" fill="{col}" fill-opacity="{op}" fill-rule="evenodd" '
                 f'stroke="{col}" stroke-width="1.5" stroke-opacity="0.5"/>')

    def save(self, name):
        self.add("</svg>")
        p = ROOT / "figures" / name
        p.write_text("\n".join(self.o))
        print("wrote", p)


# ---------------------------------------------------------------- track geometry
def arc_pts(phi0, R, q, r_max, dth=0.004):
    """circular trajectory from the origin: initial direction phi0, radius R mm, charge sign q."""
    cx, cy = R * math.cos(phi0 + q * math.pi / 2), R * math.sin(phi0 + q * math.pi / 2)
    psi0 = math.atan2(-cy, -cx)
    out, th = [], 0.0
    while th < math.pi:
        x, y = cx + R * math.cos(psi0 + q * th), cy + R * math.sin(psi0 + q * th)
        out.append((x, y, math.hypot(x, y)))
        if out[-1][2] > r_max:
            break
        th += dth
    return out


def cross_at(pts, rt):
    for i in range(1, len(pts)):
        if pts[i - 1][2] <= rt <= pts[i][2]:
            (x0, y0, r0), (x1, y1, r1) = pts[i - 1], pts[i]
            f = (rt - r0) / max(r1 - r0, 1e-9)
            return x0 + f * (x1 - x0), y0 + f * (y1 - y0)
    return None


def P(x, y): return CX + x * S, CY - y * S


def draw_centrepiece(f, rnd):
    f.ring(H_IN, H_OUT, HCAL_C, 0.16)
    f.ring(E_IN, E_OUT, ECAL_C, 0.30)
    f.circ(CX, CY, OUT[-1] * S + 6, "#9aa4b2", "none", 0, op=0.10)
    for rs, col in ((PIX, PIXEL), (STR, STRIP), (OUT, OUTER)):
        for r in rs:
            f.circ(CX, CY, r * S, "none", col, 2.0)
    f.circ(CX, CY, 5, INK)

    showers, marks = [], []
    for phi0, R, q in [(0.55, 3400, +1), (2.35, 2300, -1), (4.35, 5200, +1), (5.55, 2800, -1)]:
        pts = arc_pts(phi0, R, q, E_IN)
        d = " ".join(("M" if i == 0 else "L") + "%.1f,%.1f" % P(x, y) for i, (x, y, _) in enumerate(pts))
        f.path(d, TRACK_C, 2.6)
        for r in PIX + STR + OUT:
            c = cross_at(pts, r)
            if c:
                px, py = P(*c)
                f.circ(px, py, 4.0, TRACK_C)
                if r == STR[2]:                       # a mid-tracker layer, for the zoom callout
                    marks.append(c)
        entry = cross_at(pts, E_IN)
        if entry:
            showers.append(entry)
            ex, ey = P(*entry)
            f.path(f"M{ex-9:.1f},{ey-9:.1f} L{ex+9:.1f},{ey+9:.1f} M{ex-9:.1f},{ey+9:.1f} L{ex+9:.1f},{ey-9:.1f}",
                   INK, 2.4)                                     # helix anchor on the calo face
            phi_e = math.atan2(entry[1], entry[0])
            for _ in range(64):
                rr = E_IN + abs(rnd.gauss(0, 1)) * 165
                if rnd.random() < 0.22:                            # a tail into the HCAL
                    rr = H_IN + abs(rnd.gauss(0, 1)) * 130
                pp = phi_e + rnd.gauss(0, 0.048)
                if rr > 2000 or E_OUT < rr < H_IN: continue        # nothing lives in the physical gap
                sx, sy = P(rr * math.cos(pp), rr * math.sin(pp))
                col = ECAL_C if rr <= E_OUT else HCAL_C
                f.circ(sx, sy, rnd.uniform(2.2, 6.5), col, op=0.85)
    return showers, marks


# ---------------------------------------------------------------- left panel: tracker
def mini_layers(f, x, y, w, h, n_hits, ghost=True):
    """a small 'strip of layers' with the first n_hits placed"""
    f.card(x, y, w, h, "#ffffff", LINE, 1.6, 12)
    xs = [x + 22 + i * (w - 44) / 4.0 for i in range(5)]
    for i, xx in enumerate(xs):
        col = PIXEL if i < 2 else (STRIP if i < 4 else OUTER)
        f.path(f"M{xx:.1f},{y+18:.1f} L{xx:.1f},{y+h-18:.1f}", col, 3.2)
    ys = [y + h * 0.66, y + h * 0.56, y + h * 0.46, y + h * 0.38, y + h * 0.32]
    d = f"M{x+14:.1f},{y+h*0.74:.1f} " + " ".join("L%.1f,%.1f" % (xs[i], ys[i]) for i in range(n_hits))
    f.path(d, TRACK_C, 2.4)
    if ghost and n_hits < 5:
        f.path(f"M{xs[n_hits-1]:.1f},{ys[n_hits-1]:.1f} L{xs[n_hits]:.1f},{ys[n_hits]:.1f}",
               MUTED, 2.0, dash="5 5")
    for i in range(n_hits):
        f.circ(xs[i], ys[i], 5.5, TRACK_C)
    if n_hits < 5:
        f.circ(xs[n_hits], ys[n_hits], 5.5, "none", MUTED, 2.0, dash="3 3")


def left_panel(f):
    x0, y0, w, h = 40, 140, 480, 660
    f.card(x0, y0, w, h)
    f.t(x0 + 26, y0 + 44, "TRACKER", 27, PIXEL, "700")
    f.t(x0 + 26, y0 + 74, "one hit at a time, outward", 20, MUTED, "500")

    fw = 132
    for k in range(3):
        mini_layers(f, x0 + 22 + k * (fw + 20), y0 + 96, fw, 128, k + 1)
        if k < 2:
            f.arrow(x0 + 22 + k * (fw + 20) + fw + 3, y0 + 160, x0 + 22 + (k + 1) * (fw + 20) - 5, y0 + 160,
                    MUTED, "m_mut", 2.6)
    ax0, ax1 = x0 + 22 + fw / 2, x0 + 22 + 2 * (fw + 20) + fw / 2
    f.path(f"M{ax1:.1f},{y0+232:.1f} C{ax1:.1f},{y0+282:.1f} {ax0:.1f},{y0+282:.1f} {ax0:.1f},{y0+238:.1f}",
           STRIP, 2.6, "m_teal", "8 6")
    f.t((ax0 + ax1) / 2, y0 + 300, "each hit is fed back in", 19, STRIP, "600", "middle")

    # hierarchical anchor: layer -> module -> position inside it
    f.t(x0 + 26, y0 + 352, "each step: pick a MODULE,", 19, MUTED, "500")
    f.t(x0 + 26, y0 + 376, "then a point inside it", 19, MUTED, "500")
    acx, acy, r_arc = x0 + 132, y0 + 620, 230.0
    a0, a1 = math.radians(-19), math.radians(19)
    sx, sy = acx + r_arc * math.sin(a0), acy - r_arc * math.cos(a0)
    ex, ey = acx + r_arc * math.sin(a1), acy - r_arc * math.cos(a1)
    f.path(f"M{sx:.1f},{sy:.1f} A{r_arc:.1f},{r_arc:.1f} 0 0,1 {ex:.1f},{ey:.1f}", LINE, 16)
    hx = hy = None
    for i in range(7):
        ang = -18 + i * 6.0
        a = math.radians(ang)
        mx, my = acx + r_arc * math.sin(a), acy - r_arc * math.cos(a)
        hi = (i == 4)
        f.add(f'<rect x="{mx-13:.1f}" y="{my-8:.1f}" width="26" height="16" rx="3" '
              f'fill="{OUTER if hi else "#ffffff"}" stroke="{OUTER if hi else MUTED}" stroke-width="1.8" '
              f'transform="rotate({ang:.1f} {mx:.1f} {my:.1f})"/>')
        if hi: hx, hy = mx, my
    bx, by, bw, bh = x0 + 262, y0 + 400, 190, 124
    f.path(f"M{hx+12:.1f},{hy-10:.1f} L{bx:.1f},{by:.1f}", MUTED, 1.6, dash="4 4")
    f.path(f"M{hx+12:.1f},{hy+10:.1f} L{bx:.1f},{by+bh:.1f}", MUTED, 1.6, dash="4 4")
    f.card(bx, by, bw, bh, "#ffffff", OUTER, 2.4, 10)
    for gx in range(1, 5):
        f.path(f"M{bx+gx*bw/5:.1f},{by+8:.1f} L{bx+gx*bw/5:.1f},{by+bh-8:.1f}", LINE, 1.2)
    for gy in range(1, 3):
        f.path(f"M{bx+8:.1f},{by+gy*bh/3:.1f} L{bx+bw-8:.1f},{by+gy*bh/3:.1f}", LINE, 1.2)
    px, py = bx + bw * 0.58, by + bh * 0.44
    f.circ(px, py, 8, OUTER)
    f.arrow(bx + 14, py, px - 12, py, OUTER, "m_pur", 2.0, "4 3")
    f.arrow(px, by + bh - 14, px, py + 12, OUTER, "m_pur", 2.0, "4 3")
    f.t(bx + bw / 2, by + bh + 28, "offset inside the module", 18, MUTED, "500", "middle")
    f.t(x0 + 26, y0 + 636, "1 of 18,824 modules", 19, OUTER, "600")


# ---------------------------------------------------------------- right panel: calo
def cloud(f, x, y, w, t, rnd, energy=False):
    f.card(x, y, w, w, "#ffffff", LINE, 1.6, 12)
    base = [(rnd.gauss(0, 1), rnd.gauss(0, 1)) for _ in range(70)]
    tgt = [(rnd.gauss(0, 0.30) + 0.1, rnd.gauss(0, 0.20)) for _ in base]
    for (a, b), (c, d) in zip(base, tgt):
        u, v = a + (c - a) * t, b + (d - b) * t
        cx, cy = x + w / 2 + u * w * 0.26, y + w / 2 + v * w * 0.26
        if energy:
            e = rnd.random() ** 2.2
            f.circ(cx, cy, 2.2 + 7.5 * e, ECAL_C if e < 0.72 else HCAL_C, op=0.35 + 0.6 * e)
        else:
            f.circ(cx, cy, 2.4 + 2.4 * t, ECAL_C, op=0.32 + 0.55 * t)


def right_panel(f, rnd):
    x0, y0, w, h = 1180, 140, 480, 660
    f.card(x0, y0, w, h)
    f.t(x0 + 26, y0 + 44, "CALORIMETER", 27, ECAL_C, "700")
    f.t(x0 + 26, y0 + 74, "every cell at once, from noise", 20, MUTED, "500")

    fw = 132
    for k, t in enumerate((0.0, 0.5, 1.0)):
        xx = x0 + 22 + k * (fw + 20)
        cloud(f, xx, y0 + 96, fw, t, random.Random(4), False)
        if k < 2:
            f.arrow(xx + fw + 3, y0 + 162, xx + fw + 17, y0 + 162, ECAL_C, "m_or", 2.6)
    f.t(x0 + 22, y0 + 262, "noise", 19, MUTED, "500")
    f.t(x0 + 22 + fw + 20, y0 + 262, "halfway", 19, MUTED, "500")
    f.t(x0 + 22 + 2 * (fw + 20), y0 + 262, "shower shape", 19, MUTED, "500")

    f.arrow(x0 + 62, y0 + 288, x0 + 62, y0 + 322, MUTED, "m_mut", 2.6)
    f.t(x0 + 82, y0 + 314, "then an energy per cell", 19, MUTED, "500")
    cw = 250
    cloud(f, x0 + (w - cw) / 2, y0 + 336, cw, 1.0, random.Random(4), True)
    rx, ry = x0 + (w - cw) / 2 + cw + 16, y0 + 356
    for i in range(9):
        e = i / 8.0
        f.circ(rx + 12, ry + i * 22, 2.2 + 7.5 * e, ECAL_C if e < 0.72 else HCAL_C, op=0.35 + 0.6 * e)
    f.t(x0 + w / 2, y0 + 336 + cw + 34, "one point cloud — never snapped to cells", 19, MUTED, "500", "middle")


def build(text):
    rnd = random.Random(11)
    f = Fig(text)
    f.t(46, 54, "How the two heads build an event", 34, INK, "700")
    f.t(46, 88, f"one truth particle in → tracker hits, then a calorimeter shower  ·  B = {B_FIELD} T",
        21, MUTED, "500")
    showers, marks = draw_centrepiece(f, rnd)
    left_panel(f)
    right_panel(f, rnd)
    # zoom callouts from the transverse view into each panel
    left = min(marks, key=lambda c: c[0]) if marks else None
    if left:
        lx, ly = P(*left)
        f.circ(lx, ly, 58, "none", PIXEL, 2.0, dash="6 5")
        f.path(f"M{lx-58:.1f},{ly:.1f} L{520:.1f},{ly:.1f}", PIXEL, 2.0, dash="6 5")
    best = max(showers, key=lambda c: c[0]) if showers else None
    if best:
        sx, sy = P(*best)
        f.circ(sx, sy, 64, "none", ECAL_C, 2.0, dash="6 5")
        f.path(f"M{sx+64:.1f},{sy:.1f} L{1180:.1f},{sy:.1f}", ECAL_C, 2.0, dash="6 5")
    f.t(CX, CY + 342, "transverse view of the detector", 20, MUTED, "500", "middle")
    f.save("model_overview.svg" if text else "model_overview_notext.svg")


build(True)
build(False)
