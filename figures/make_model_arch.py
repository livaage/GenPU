"""Model-centric figure: what kind of model each head is, and where the heads sit.

The detector appears only as two small OUTPUT thumbnails; the architecture is the subject.
Emits model_arch.svg (labelled) and model_arch_notext.svg (identical drawing, zero text).

Architecture drawn from:
  genpu/conditioning.py            7 cont features + PDG embedding -> one shared vector c
  models/tracker_module_ar.py      causal transformer decoder, 4 layers/4 heads, d=128
                                   -> hierarchical module head (48 layers -> masked surface)
                                   -> local x,y,z 512-bin + time 64-bin heads
  flow/calo_flow.py                c -> GlobalHead (Gaussian mixture, K=4)
                                        PointCFM   (flow-matching velocity field, 50 Euler steps)
                                        EnergyHead (at-floor Bernoulli + mixture)
"""
import math, random, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

FF = "DejaVu Sans, Helvetica, Arial, sans-serif"
INK, MUTED, LINE = "#12161c", "#5b6673", "#c9d1da"
BG, PANEL, SOFT = "#ffffff", "#f7f9fb", "#eef2f6"
COND = "#0e9488"                      # shared conditioning
TRK, TRK2 = "#1f6feb", "#7c5cd6"      # tracker trunk / heads
CAL, CAL2, GOLD = "#e8871a", "#c2384a", "#b8860b"
W, H = 1820, 1080


class Fig:
    def __init__(self, text=True):
        self.text = text
        self.items = []                       # every label, for the editable-PPTX build
        self.o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">', '<defs>']
        for n, c in [("k_ink", INK), ("k_mut", MUTED), ("k_cond", COND), ("k_trk", TRK),
                     ("k_trk2", TRK2), ("k_cal", CAL), ("k_cal2", CAL2), ("k_gold", GOLD)]:
            self.o.append(f'<marker id="{n}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
                          f'orient="auto-start-reverse"><path d="M0,1 L10,5 L0,9 z" fill="{c}"/></marker>')
        self.o += ['</defs>', f'<rect width="{W}" height="{H}" fill="{BG}"/>']

    def add(self, s): self.o.append(s)

    def t(self, x, y, s, size=20, fill=INK, weight="600", anchor="start"):
        self.items.append(dict(x=x, y=y, text=s, size=size, fill=fill,
                               bold=(weight in ("700", "bold")), anchor=anchor))
        if self.text:
            self.o.append(f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FF}" font-size="{size}" '
                          f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{s}</text>')

    def box(self, x, y, w, h, fill=PANEL, stroke=LINE, sw=2, rx=14, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}" '
                 f'stroke="{stroke}" stroke-width="{sw}"{d}/>')

    def sq(self, x, y, s, fill, stroke="none", sw=0, rx=3, op=1.0):
        self.add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{s:.1f}" height="{s:.1f}" rx="{rx}" fill="{fill}" '
                 f'fill-opacity="{op}" stroke="{stroke}" stroke-width="{sw}"/>')

    def circ(self, x, y, r, fill="none", stroke="none", sw=2, op=1.0, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.2f}" fill="{fill}" fill-opacity="{op}" '
                 f'stroke="{stroke}" stroke-width="{sw}"{d}/>')

    def path(self, d, col=MUTED, sw=2.4, mk=None, dash=None, fill="none"):
        m = f' marker-end="url(#{mk})"' if mk else ""
        ds = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<path d="{d}" fill="{fill}" stroke="{col}" stroke-width="{sw}"{m}{ds}/>')

    def arrow(self, x0, y0, x1, y1, col=MUTED, mk="k_mut", sw=2.6, dash=None):
        self.path(f"M{x0:.1f},{y0:.1f} L{x1:.1f},{y1:.1f}", col, sw, mk, dash)

    def save(self, name):
        self.add("</svg>")
        p = ROOT / "figures" / name
        p.write_text("\n".join(self.o))
        print("wrote", p)


# ------------------------------------------------------------------ reusable model glyphs
def vec_col(f, x, y, n, col, s=22, gap=5, op=0.85):
    for i in range(n):
        f.sq(x, y + i * (s + gap), s, col, "none", 0, 4, op)
    return y + n * (s + gap) - gap


def mlp_glyph(f, x, y, w, h, col=COND, cols=(4, 5, 4)):
    xs = [x + i * w / (len(cols) - 1) for i in range(len(cols))]
    pts = []
    for ci, n in enumerate(cols):
        pts.append([(xs[ci], y + h * (j + 0.5) / n) for j in range(n)])
    for a, b in zip(pts, pts[1:]):
        for p in a:
            for q in b:
                f.path(f"M{p[0]:.1f},{p[1]:.1f} L{q[0]:.1f},{q[1]:.1f}", col, 0.7)
    for lay in pts:
        for p in lay:
            f.circ(p[0], p[1], 5, col)


def attn_mask(f, x, y, n=8, c=13, col=TRK):
    """lower-triangular attention mask — the visual signature of a causal transformer"""
    for r in range(n):
        for k in range(n):
            on = k <= r
            f.sq(x + k * c, y + r * c, c - 2, col if on else SOFT, "none", 0, 2, 0.85 if on else 1.0)


def block_stack(f, x, y, w, h, col, n=3, off=7):
    for i in range(n - 1, 0, -1):
        f.box(x + i * off, y - i * off, w, h, "#ffffff", col, 1.6, 12)
    f.box(x, y, w, h, "#ffffff", col, 2.6, 12)


def bars(f, x, y, w, h, n, hi, col, base=0.30):
    bw = w / n
    for i in range(n):
        v = h if i == hi else h * base * (0.45 + 0.55 * abs(math.sin(i * 1.7)))
        f.add(f'<rect x="{x + i*bw:.1f}" y="{y + h - v:.1f}" width="{bw*0.62:.1f}" height="{v:.1f}" rx="2" '
              f'fill="{col if i == hi else LINE}"/>')


def grid_sel(f, x, y, cols, rows, cell, hi, col):
    for r in range(rows):
        for c in range(cols):
            on = (r, c) == hi
            f.sq(x + c * cell, y + r * cell, cell - 3, col if on else SOFT, "none", 0, 2)


def mixture(f, x, y, w, h, col):
    comps = [(0.34, 0.055, 1.0), (0.55, 0.13, 0.42), (0.76, 0.20, 0.22)]
    for mu, sg, a in comps:
        p = [f"{x + u*w:.1f},{y + h - a*h*0.82*math.exp(-((u-mu)**2)/(2*sg*sg)):.1f}"
             for u in [i / 59 for i in range(60)]]
        f.path("M" + " L".join(p), col, 1.4, dash="4 3")
    p = []
    for i in range(80):
        u = i / 79
        v = sum(a * math.exp(-((u - mu) ** 2) / (2 * sg * sg)) for mu, sg, a in comps)
        p.append(f"{x + u*w:.1f},{y + h - min(v, 1.25) * h * 0.66:.1f}")
    f.path("M" + " L".join(p), col, 3.0)


def vfield(f, x, y, w, h, col, rnd):
    """learned velocity field + Euler trajectories: the visual signature of flow matching"""
    nx, ny = 7, 5
    for i in range(nx):
        for j in range(ny):
            px, py = x + (i + 0.5) * w / nx, y + (j + 0.5) * h / ny
            tx, ty = x + w * 0.72, y + h * 0.5
            dx, dy = tx - px, ty - py
            n = math.hypot(dx, dy) or 1
            dx, dy = dx / n * 13, dy / n * 13
            f.path(f"M{px:.1f},{py:.1f} L{px+dx:.1f},{py+dy:.1f}", col, 1.5, "k_cal")
    for s in (-1, 0, 1):
        p0 = (x + w * 0.12, y + h * (0.5 + 0.30 * s))
        p1 = (x + w * 0.72, y + h * (0.5 + 0.05 * s))
        pts = [(p0[0] + (p1[0] - p0[0]) * t, p0[1] + (p1[1] - p0[1]) * t + 26 * s * math.sin(math.pi * t))
               for t in [i / 8 for i in range(9)]]
        f.path("M" + " L".join(f"{a:.1f},{b:.1f}" for a, b in pts), col, 2.4)
        for a, b in pts:
            f.circ(a, b, 3.0, col)


def floor_spectrum(f, x, y, w, h, col):
    u0 = 0.22
    p = []
    for i in range(60):
        u = u0 + (1 - u0) * i / 59
        p.append(f"{x + u*w:.1f},{y + h - math.exp(-((u-0.48)**2)/0.05)*h*0.86:.1f}")
    xf = x + u0 * w
    f.path(f"M{xf:.1f},{y+h:.1f} L" + " L".join(p), col, 3.0)
    f.path(f"M{xf:.1f},{y+6:.1f} L{xf:.1f},{y+h:.1f}", col, 2.0, dash="6 5")


def cloud(f, cx, cy, n, sx, sy, rnd, energy=False, clip=None):
    for _ in range(n):
        px, py = cx + rnd.gauss(0, sx), cy + rnd.gauss(0, sy)
        if clip and not (clip[0] < px < clip[2] and clip[1] < py < clip[3]):
            continue
        if energy:
            e = rnd.random() ** 2.2
            f.circ(px, py, 2.0 + 7.0 * e, CAL if e < 0.72 else CAL2, op=0.4 + 0.55 * e)
        else:
            f.circ(px, py, rnd.uniform(2.0, 4.4), CAL, op=0.75)


# ------------------------------------------------------------------ the figure
def build(text):
    rnd = random.Random(5)
    f = Fig(text)
    LX, LY, LW, LH = 372, 128, 900, 400          # tracker lane
    LANE_X = LX
    CY0 = 570                                    # calo lane
    CALO_BUS_Y = CY0 + 78                        # conditioning bus above the three calo heads
    f.t(46, 54, "Two generative heads on one shared conditioning", 34, INK, "700")
    f.t(46, 88, "tracker: autoregressive transformer  ·  calorimeter: flow matching  ·  the detector is only the output",
        21, MUTED, "500")

    # ---------------- input + shared conditioning
    f.t(52, 150, "truth particle", 21, INK, "700")
    fy = 176
    names = ["log pT", "η", "log E", "q", "m", "vr", "vz", "PDG"]
    for i, nm in enumerate(names):
        f.sq(52, fy + i * 27, 22, COND, "none", 0, 4, 0.85 if i < 7 else 0.45)
        f.t(82, fy + i * 27 + 17, nm, 17, MUTED, "500")
    mlp_glyph(f, 150, fy + 20, 90, 150, COND)
    f.t(150, fy + 210, "MLP", 19, COND, "700")
    ce_y = fy + 8
    vec_col(f, 262, ce_y, 6, COND, 22, 5)
    f.t(273, ce_y - 12, "c", 22, COND, "700", "middle")
    f.arrow(126, fy + 95, 144, fy + 95, COND, "k_cond")
    f.arrow(246, fy + 95, 258, fy + 95, COND, "k_cond")

    bx = 320                                       # branch point
    f.path(f"M{296},{fy+95:.1f} L{bx},{fy+95:.1f}", COND, 3)
    f.circ(bx, fy + 95, 6, COND)
    f.path(f"M{bx},{fy+95:.1f} L{bx},{300} L{372},{300}", COND, 3, "k_cond")
    f.path(f"M{bx},{fy+95:.1f} L{bx},{CALO_BUS_Y} L{LANE_X + 34 + 2*(268+22) + 134:.1f},{CALO_BUS_Y}",
           COND, 3)

    # ================= TRACKER LANE
    f.box(LX, LY, LW, LH, PANEL, LINE, 2, 18)
    f.t(LX + 26, LY + 40, "TRACKER HEAD", 24, TRK, "700")
    f.t(LX + LW - 26, LY + 40, "autoregressive transformer", 21, MUTED, "500", "end")
    f.t(LX + LW - 26, LY + 66, "closest published model:  arXiv:2512.24254", 17, MUTED, "500", "end")

    for i in range(4):
        x = LX + 34 + i * 58
        dashed = (i == 3)
        f.box(x, LY + 62, 46, 40, "#ffffff", MUTED if dashed else TRK, 2, 8, "4 4" if dashed else None)
        if not dashed:
            f.circ(x + 23, LY + 82, 6, TRK)

    tb_x, tb_y, tb_w, tb_h = LX + 34, LY + 148, 240, 196
    block_stack(f, tb_x, tb_y, tb_w, tb_h, TRK)
    attn_mask(f, tb_x + 22, tb_y + 30, 8, 13, TRK)
    mlp_glyph(f, tb_x + 150, tb_y + 40, 66, 86, TRK, (3, 4, 3))
    f.t(tb_x, tb_y + tb_h + 24, "causal self-attention  ×4", 18, TRK, "600")
    f.arrow(LX + 34 + 3 * 58 + 23, LY + 104, tb_x + tb_w / 2, tb_y - 12, TRK, "k_trk", 2.4)
    f.arrow(372, 300, tb_x - 10, 300, COND, "k_cond", 3)

    hx = tb_x + tb_w + 46
    f.box(hx, tb_y - 10, 300, 92, "#ffffff", TRK2, 2.4, 12)
    bars(f, hx + 18, tb_y + 6, 118, 52, 16, 6, TRK2)
    f.arrow(hx + 146, tb_y + 34, hx + 168, tb_y + 34, TRK2, "k_trk2", 2.2)
    grid_sel(f, hx + 182, tb_y + 8, 5, 3, 17, (1, 2), TRK2)
    f.t(hx + 18, tb_y + 76, "layer (48) → surface", 18, TRK2, "600")

    f.box(hx, tb_y + 96, 300, 90, "#ffffff", TRK2, 2.4, 12)
    for k in range(3):
        bars(f, hx + 18 + k * 92, tb_y + 112, 76, 40, 12, 4 + k, TRK2, 0.22)
    f.t(hx + 18, tb_y + 174, "local x, y, z  ·  512 bins", 18, TRK2, "600")
    f.arrow(tb_x + tb_w + 8, tb_y + 40, hx - 6, tb_y + 40, TRK, "k_trk", 2.4)
    f.arrow(tb_x + tb_w + 8, tb_y + 140, hx - 6, tb_y + 140, TRK, "k_trk", 2.4)
    f.t(hx + 312, tb_y + 34, "2 heads", 19, TRK2, "700")

    f.path(f"M{hx+150:.1f},{tb_y-16:.1f} C{hx+150:.1f},{LY+112:.1f} {LX+150:.1f},{LY+112:.1f} "
           f"{LX+57:.1f},{LY+106:.1f}", TRK, 2.4, "k_trk", "8 6")
    f.t(LX + 34, LY + 130, "sampled hit fed back in", 18, TRK, "600")

    # ================= CALO LANE
    f.box(LX, CY0, LW, 400, PANEL, LINE, 2, 18)
    f.t(LX + 26, CY0 + 40, "CALORIMETER HEAD", 24, CAL, "700")
    f.t(LX + LW - 26, CY0 + 40, "conditional flow matching", 21, MUTED, "500", "end")
    f.t(LX + LW - 26, CY0 + 66, "closest published model:  CaloClouds II, arXiv:2309.05704", 17, MUTED, "500", "end")
    f.t(LX + 34, CY0 + 68, "3 heads, no sequence", 19, CAL, "700")

    hw, hh, hy = 268, 220, CY0 + 104
    for i, (col, lab) in enumerate([(GOLD, "how big"), (CAL, "where"), (CAL2, "how bright")]):
        x = LX + 34 + i * (hw + 22)
        f.box(x, hy, hw, hh, "#ffffff", col, 2.4, 12)
        f.t(x + 16, hy + 32, lab, 20, col, "700")
        if i == 0:
            mixture(f, x + 18, hy + 48, hw - 36, 104, GOLD)
            for k in range(4):
                f.sq(x + 18 + k * 30, hy + 166, 22, GOLD, "none", 0, 4, 0.8)
            f.t(x + 18, hy + 204, "mixture → total E, N points, core", 17, MUTED, "500")
        elif i == 1:
            vfield(f, x + 16, hy + 44, hw - 32, 112, CAL, rnd)
            f.t(x + 18, hy + 180, "velocity field, 50 Euler steps", 17, MUTED, "500")
            f.t(x + 18, hy + 204, "noise → all points at once", 17, MUTED, "500")
        else:
            floor_spectrum(f, x + 20, hy + 48, hw - 44, 104, CAL2)
            f.t(x + 18, hy + 180, "Gaussian mixture over log E,", 17, MUTED, "500")
            f.t(x + 18, hy + 204, "truncated at the 50 keV cut", 17, MUTED, "500")
        f.arrow(x + hw / 2, CALO_BUS_Y, x + hw / 2, hy - 6, COND, "k_cond", 2.6)
        if i:
            f.arrow(x - 24, hy + hh / 2, x - 4, hy + hh / 2, CAL2, "k_cal2", 3.0)

    # ================= OUTPUT THUMBNAILS (the detector, demoted)
    OX = 1310
    f.t(OX + 12, LY + 40, "output", 21, INK, "700")
    f.box(OX, LY + 58, 460, 320, "#ffffff", LINE, 2, 14)
    ccx, ccy = OX + 26, LY + 372
    A0, A1 = math.radians(-68), math.radians(-18)
    for r, col in [(96, TRK), (150, TRK), (206, TRK), (258, TRK2), (300, TRK2)]:
        f.path(f"M{ccx + r*math.cos(A0):.1f},{ccy + r*math.sin(A0):.1f} "
               f"A{r},{r} 0 0,1 {ccx + r*math.cos(A1):.1f},{ccy + r*math.sin(A1):.1f}", col, 3)
    tr = []
    for i, r in enumerate((96, 150, 206, 258, 300)):
        a = math.radians(-56 + i * 4.2)
        tr.append((ccx + r * math.cos(a), ccy + r * math.sin(a)))
    f.path("M%.1f,%.1f " % (ccx + 40 * math.cos(math.radians(-60)), ccy + 40 * math.sin(math.radians(-60)))
           + " ".join("L%.1f,%.1f" % t for t in tr), INK, 2.4)
    for t in tr:
        f.circ(t[0], t[1], 6, INK)
    f.t(OX + 20, LY + 96, "hits, inner → outer", 18, MUTED, "500")

    f.box(OX, CY0 + 58, 460, 320, "#ffffff", LINE, 2, 14)
    cloud(f, OX + 230, CY0 + 218, 150, 62, 42, rnd, True,
          clip=(OX + 18, CY0 + 110, OX + 442, CY0 + 366))
    f.t(OX + 20, CY0 + 96, "energy deposits — a point cloud,", 18, MUTED, "500")
    f.t(OX + 20, CY0 + 120, "NOT projected onto detector cells", 18, CAL2, "600")
    f.arrow(1272, LY + 218, OX - 8, LY + 218, TRK2, "k_trk2", 2.6)
    f.t(LX + 34, CY0 + 382, "each head also sees the one before it: globals → positions → energy",
        18, CAL2, "600")
    jy = CY0 + 352
    for i in range(3):
        cxm = LX + 34 + i * (hw + 22) + hw / 2
        f.path(f"M{cxm:.1f},{hy + hh:.1f} L{cxm:.1f},{jy:.1f} L{LX + LW - 70:.1f},{jy:.1f}", CAL2, 2.4)
    f.circ(LX + LW - 70, jy, 6, CAL2)
    f.arrow(LX + LW - 70, jy, OX - 8, jy, CAL2, "k_cal2", 2.6)

    f.t(46, H - 44, "flow matching: Lipman et al. arXiv:2210.02747  ·  OT-CFM: Tong et al. arXiv:2302.00482"
        "  ·  point-cloud calorimeter: CaloClouds arXiv:2305.04847, CaloClouds II arXiv:2309.05704",
        17, MUTED, "500")
    f.t(46, H - 20, "tracker sequence model: arXiv:2512.24254  ·  data + geometry: ColliderML / "
        "OpenDataDetector arXiv:2512.15230", 17, MUTED, "500")
    f.save("model_arch.svg" if text else "model_arch_notext.svg")
    if text:
        import json
        (ROOT / "figures" / "model_arch_text.json").write_text(
            json.dumps(dict(w=W, h=H, items=f.items), indent=1))
        print("wrote model_arch_text.json:", len(f.items), "labels")


build(True)
build(False)
