"""Architecture diagrams for slides: tracker AR head and calo flow head.

Contents checked against the code:
  conditioning  -> genpu/conditioning.py     CONT_FEATURES + PDG embedding, ONE PARTICLE per call
  particle list -> build_tracker_slice.py:117 (--population default "all")
                   build_calo_slice_v2.py     (no primary filter; groups by calo-incident ancestor,
                                               which is itself usually a secondary)
  tracker       -> models/tracker_module_ar.py (v3: hierarchical module token + local bins)
                   models/tracker_ar.py        (decoder 4 layers / 4 heads / d=128; 512 & 64 bins)
  calo          -> flow/calo_flow.py (GlobalHead mixture, PointCFM, EnergyHead) + calo_geom.py anchor
"""
import math, random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FF = "DejaVu Sans, Helvetica, Arial, sans-serif"

INK, MUTED, LINE = "#12161c", "#5b6673", "#c9d1da"
BG, PANEL = "#ffffff", "#f4f6f9"
BLUE, TEAL, PURPLE = "#1f6feb", "#0e9488", "#7c5cd6"
ORANGE, RED, GOLD = "#e8871a", "#c2384a", "#b8860b"
W = 1520


class Svg:
    def __init__(self, w, h):
        self.w, self.h, self.o = w, h, []
        self.o.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">')
        self.o.append('<defs>')
        for name, col in [("a_ink", INK), ("a_mut", MUTED), ("a_blue", BLUE), ("a_teal", TEAL),
                          ("a_pur", PURPLE), ("a_or", ORANGE), ("a_red", RED)]:
            self.o.append(f'<marker id="{name}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
                          f'markerHeight="7" orient="auto-start-reverse">'
                          f'<path d="M0,1 L10,5 L0,9 z" fill="{col}"/></marker>')
        self.o.append('</defs>')
        self.o.append(f'<rect width="{w}" height="{h}" fill="{BG}"/>')

    def add(self, s): self.o.append(s)

    def txt(self, x, y, s, size=22, fill=INK, weight="500", anchor="start", op=1.0):
        self.add(f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FF}" font-size="{size}" '
                 f'font-weight="{weight}" fill="{fill}" fill-opacity="{op}" text-anchor="{anchor}">{s}</text>')

    def card(self, x, y, w, h, fill=PANEL, stroke=LINE, sw=2, rx=16, op=1.0):
        self.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
                 f'fill-opacity="{op}" stroke="{stroke}" stroke-width="{sw}"/>')

    def tag(self, x, y, s, col, size=21):
        w = len(s) * size * 0.60 + 26
        self.add(f'<rect x="{x}" y="{y-size-7}" width="{w:.0f}" height="{size+16}" rx="{(size+16)/2:.0f}" '
                 f'fill="{col}" fill-opacity="0.14" stroke="{col}" stroke-width="1.6"/>')
        self.txt(x + w / 2, y, s, size, col, "700", "middle")
        return w

    def chip(self, x, y, s, col, size=18):
        w = len(s) * size * 0.60 + 20
        self.add(f'<rect x="{x}" y="{y-size-5}" width="{w:.0f}" height="{size+11}" rx="{(size+11)/2:.0f}" '
                 f'fill="{col}" fill-opacity="0.16"/>')
        self.txt(x + w / 2, y, s, size, col, "700", "middle")
        return w

    def arrow(self, x0, y0, x1, y1, col=MUTED, mk="a_mut", sw=3, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<path d="M{x0:.1f},{y0:.1f} L{x1:.1f},{y1:.1f}" stroke="{col}" stroke-width="{sw}" '
                 f'fill="none" marker-end="url(#{mk})"{d}/>')

    def path(self, d, col=MUTED, mk=None, sw=3, dash=None):
        m = f' marker-end="url(#{mk})"' if mk else ""
        ds = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<path d="{d}" stroke="{col}" stroke-width="{sw}" fill="none"{m}{ds}/>')

    def lines(self, x, y, rows, size=19, col=MUTED, weight="500", gap=26):
        for i, t in enumerate(rows):
            if t:
                self.txt(x, y + i * gap, t, size, col, weight)

    def save(self, name):
        self.add("</svg>")
        p = ROOT / "figures" / name
        p.write_text("\n".join(self.o))
        print("wrote", p)


PCARD_H = 278


def particle_card(s, x, y, w, accent, rows, callline, footnote):
    """The input is Geant's PARTICLE LIST — primaries and secondaries alike, each at its own true
    production vertex. Measured on shard 0 (10,000 events, cascade_graph): 855 particles/event,
    224 primary + 631 non-primary; non-primaries have median production radius 424 mm."""
    s.card(x, y, w, PCARD_H)
    s.txt(x + 24, y + 38, "TRUTH PARTICLE LIST", 24, INK, "700")
    s.txt(x + 24, y + 66, "855 per event — 224 from the generator, 631 made by Geant4", 19, MUTED, "500")
    for i, (chip, col, desc) in enumerate(rows):
        yy = y + 88 + i * 34
        s.card(x + 24, yy, w - 48, 28, "#ffffff", LINE, 1.4, 8)
        cw = s.chip(x + 36, yy + 21, chip, col)
        s.txt(x + 44 + cw, yy + 21, desc, 18, MUTED, "500")
    s.txt(x + 24, y + 206, callline, 19, MUTED, "500")
    s.txt(x + 24, y + 234, "log pT,  η,  log E,  q,  m,  vr,  vz   +   PDG class (17)", 21, accent, "700")
    s.txt(x + 24, y + 264, footnote, 18, MUTED, "500")


def footer(s, y):
    s.txt(46, y, "an event is the union of these per-particle responses — the tracker and calo heads are "
                 "independent and never see each other.", 19, MUTED, "500")
    s.txt(46, y + 26, "the secondary list with its true vertices is Geant's own output, so this is detector "
                      "RESPONSE given a truth particle list.", 19, MUTED, "500")


# ============================================================ TRACKER
def tracker():
    H = 1090
    s = Svg(W, H)
    s.txt(46, 52, "Tracker head", 38, INK, "700")
    s.txt(46, 88, "writes a track one hit at a time, inner → outer — like a language model whose words are detector modules",
          23, MUTED, "500")

    # --- row 1: inputs
    particle_card(s, 46, 118, 680, BLUE,
                  [("generator", INK, "handed to Geant4 by the generator  —  224 / event"),
                   ("Geant4-made", BLUE, "created during transport — 218/event leave hits"),
                   ("", MUTED, "NB: \u201csecondary\u201d here = made by Geant4, not the lifetime rule")],
                  "the head runs ONCE PER PARTICLE, on that particle alone:",
                  "Geant4-made: 60% of hits, a median of 1 each (generator: 11)")
    s.arrow(746, 190, 806, 190, MUTED, "a_mut")
    s.card(826, 118, 300, 144)
    s.txt(976, 158, "conditioning MLP", 24, INK, "700", "middle")
    s.txt(976, 188, "shared by both heads", 20, MUTED, "500", "middle")
    s.tag(976 - 78, 234, "c  —  128-d", BLUE)

    s.card(1166, 118, 308, PCARD_H, "#ffffff", LINE)
    s.txt(1190, 158, "why autoregressive?", 23, INK, "700")
    s.lines(1190, 194, ["a track is an ORDERED", "walk outward — hit k+1",
                        "depends on where hit k", "landed.", "",
                        "the model sees its own", "previous hits, so it cannot",
                        "emit a layer it has passed."], 19)

    # --- row 2: the decoder strip
    s.path(f"M{976},{262} L{976},{414}", BLUE, "a_blue", 3)
    s.card(46, 414, 1428, 280, PANEL, LINE)
    s.txt(70, 454, "causal transformer decoder", 26, INK, "700")
    s.txt(70, 482, "4 layers  ·  4 heads  ·  d = 128  ·  each step attends only to earlier hits", 20, MUTED, "500")

    toks = [("start\n(vertex)", PURPLE), ("hit 1", BLUE), ("hit 2", BLUE), ("hit 3", BLUE)]
    tx, tw, gap, TY = 96, 224, 66, 506
    for i, (lab, col) in enumerate(toks):
        x = tx + i * (tw + gap)
        s.card(x, TY, tw, 92, "#ffffff", col, 2.5, 14)
        ls = lab.split("\n")
        for j, ln in enumerate(ls):
            s.txt(x + tw / 2, TY + 54 + j * 28 - (14 if len(ls) > 1 else 0), ln, 23, col, "700", "middle")
        if i < len(toks) - 1:
            s.arrow(x + tw + 8, TY + 46, x + tw + gap - 10, TY + 46, MUTED, "a_mut")
    s.txt(tx + 4 * (tw + gap) - 20, TY + 54, "…  until n hits", 22, MUTED, "600")
    s.path(f"M{tx + 2*(tw+gap) + tw/2},{TY+92} C{tx + 2*(tw+gap) + tw/2},{TY+150} "
           f"{tx + 3*(tw+gap) + tw/2},{TY+150} {tx + 3*(tw+gap) + tw/2},{TY+100}", TEAL, "a_teal", 3, "9 7")
    s.txt(tx + 2.55 * (tw + gap) + tw / 2, TY + 168, "each sampled hit is fed back in", 20, TEAL, "600", "middle")

    # --- row 3: the two per-step heads
    HY = 726
    s.path(f"M{396},{694} L{396},{HY}", TEAL, "a_teal", 3)
    s.path(f"M{1124},{694} L{1124},{HY}", PURPLE, "a_pur", 3)
    s.txt(760, 716, "per step, two heads", 20, MUTED, "600", "middle")

    s.card(46, HY, 700, 268, "#ffffff", TEAL, 2.5)
    s.txt(72, HY + 44, "1.  WHICH module?", 26, TEAL, "700")
    s.txt(72, HY + 74, "discrete — pins the hit's global position", 20, MUTED, "500")
    for i in range(48):
        x = 76 + i * 8.0
        hi = (i == 21)
        s.add(f'<rect x="{x:.1f}" y="{HY + 94 + (0 if hi else 7)}" width="5" height="{38 if hi else 24}" '
              f'rx="2" fill="{TEAL if hi else LINE}"/>')
    s.txt(76, HY + 154, "layer class  (48)", 20, INK, "600")
    s.arrow(478, HY + 114, 524, HY + 114, TEAL, "a_teal", 3)
    for r in range(3):
        for c in range(6):
            hi = (r, c) == (1, 3)
            s.add(f'<rect x="{552 + c*26}" y="{HY + 90 + r*22}" width="20" height="16" rx="3" '
                  f'fill="{TEAL if hi else LINE}"/>')
    s.txt(552, HY + 154, "surface", 20, INK, "600")
    s.txt(72, HY + 190, "= 1 of 18,824 modules, predicted hierarchically", 21, MUTED, "500")
    s.txt(72, HY + 216, "surface logits masked to the ones that exist there", 19, MUTED, "500")
    s.txt(72, HY + 242, "(v1 predicted the layer only — residuals drifted outward)", 19, MUTED, "500")

    s.card(774, HY, 700, 268, "#ffffff", PURPLE, 2.5)
    s.txt(800, HY + 44, "2.  WHERE inside it?", 26, PURPLE, "700")
    s.txt(800, HY + 74, "continuous — places the hit within the module", 20, MUTED, "500")
    s.add(f'<path d="M816,{HY+110} L960,{HY+92} L1000,{HY+136} L856,{HY+156} z" fill="{PURPLE}" '
          f'fill-opacity="0.16" stroke="{PURPLE}" stroke-width="2.5"/>')
    s.add(f'<circle cx="925" cy="{HY+125}" r="8" fill="{PURPLE}"/>')
    s.txt(1024, HY + 118, "local x, y, z  —  512 bins each", 21, INK, "600")
    s.txt(1024, HY + 148, "time  —  64 bins", 21, INK, "600")
    s.txt(800, HY + 198, "standardised per module, so the residual is bounded", 21, MUTED, "500")
    s.txt(800, HY + 226, "by the module size (≲ 50 mm) by construction.", 21, MUTED, "500")

    footer(s, H - 52)
    s.save("tracker_arch.svg")


# ============================================================ CALO
def calo():
    H = 1084
    s = Svg(W, H)
    rnd = random.Random(7)
    s.txt(46, 52, "Calorimeter head", 38, INK, "700")
    s.txt(46, 88, "a shower is a CLOUD, not a sequence — sample how big it is, then flow a cloud of cells into shape",
          23, MUTED, "500")

    # --- row 1: inputs
    particle_card(s, 46, 118, 680, ORANGE,
                  [("generator", INK, "handed to Geant4 by the generator"),
                   ("Geant4-made", ORANGE, "created in tracker material during transport"),
                   ("in-calo", MUTED, "shower product — lifted to its incident ancestor")],
                  "runs ONCE PER CALO-INCIDENT PARTICLE — 167 / event, 30% primary:",
                  "after the lift, 46% of calo energy comes from a Geant4-made one")
    s.arrow(746, 166, 806, 166, MUTED, "a_mut")
    s.card(826, 118, 300, 96)
    s.txt(976, 152, "conditioning MLP", 23, INK, "700", "middle")
    s.txt(976, 186, "c  —  128-d", 22, ORANGE, "700", "middle")
    s.arrow(746, 278, 806, 278, MUTED, "a_mut")
    s.card(826, 230, 300, 96, "#fff8ef", ORANGE, 2)
    s.txt(976, 264, "helix anchor", 23, ORANGE, "700", "middle")
    s.txt(976, 294, "truth track → calo face", 19, MUTED, "500", "middle")

    s.card(1166, 118, 308, PCARD_H, "#ffffff", LINE)
    s.txt(1190, 158, "why an anchor?", 23, INK, "700")
    s.lines(1190, 194, ["the shower core sits ~1.5 m", "from the particle's original",
                        "direction — that offset is", "just the B-field bend.", "",
                        "extrapolate it exactly and", "let the model learn only the",
                        "RESIDUAL around it."], 19)

    # --- row 2: three heads
    y0, ch = 452, 392
    cw, cx = 452, [46, 534, 1022]

    # (1) GlobalHead
    s.card(cx[0], y0, cw, ch, "#ffffff", GOLD, 2.5)
    s.txt(cx[0] + 24, y0 + 44, "1.  how big?", 26, GOLD, "700")
    s.txt(cx[0] + 24, y0 + 74, "GlobalHead — Gaussian mixture (K=4)", 20, MUTED, "500")
    s.txt(cx[0] + 24, y0 + 102, "conditioned on c + the helix anchor", 19, ORANGE, "600")
    for i, (lab, val) in enumerate([("total log E", "shower energy"), ("log N", "how many cells"),
                                    ("core η, φ", "offset from the anchor")]):
        yy = y0 + 140 + i * 62
        s.card(cx[0] + 24, yy - 30, cw - 48, 48, "#fffaf0", GOLD, 1.8, 10)
        s.txt(cx[0] + 42, yy, lab, 22, INK, "700")
        s.txt(cx[0] + cw - 42, yy, val, 19, MUTED, "500", "end")
    bx, by = cx[0] + 24, y0 + 338
    pts = []
    for i in range(80):
        u = i / 79
        v = 0.9 * math.exp(-((u - 0.42) ** 2) / 0.006) + 0.35 * math.exp(-((u - 0.66) ** 2) / 0.05)
        pts.append(f"{bx + u*(cw-48):.1f},{by - v*46:.1f}")
    s.path("M" + " L".join(pts), GOLD, None, 3)
    s.txt(bx + (cw - 48) / 2, by + 24, "peak + heavy tail → a mixture, not one Gaussian", 18, MUTED, "500", "middle")

    # (2) PointCFM
    s.card(cx[1], y0, cw, ch, "#ffffff", ORANGE, 2.5)
    s.txt(cx[1] + 24, y0 + 44, "2.  where do the cells go?", 26, ORANGE, "700")
    s.txt(cx[1] + 24, y0 + 74, "PointCFM — conditional flow matching", 20, MUTED, "500")
    pw, py = 116, y0 + 104
    base = [(rnd.gauss(0, 1), rnd.gauss(0, 1)) for _ in range(60)]
    targ = [(rnd.gauss(0, 0.32) + 0.15, rnd.gauss(0, 0.22)) for _ in base]
    for k, (t, lab) in enumerate([(0.0, "t = 0"), (0.5, "t = ½"), (1.0, "t = 1")]):
        px = cx[1] + 24 + k * (pw + 34)
        s.card(px, py, pw, pw, PANEL, LINE, 1.6, 12)
        for (a, b), (c, d) in zip(base, targ):
            u, v = a + (c - a) * t, b + (d - b) * t
            s.add(f'<circle cx="{px + pw/2 + u*pw*0.26:.1f}" cy="{py + pw/2 + v*pw*0.26:.1f}" '
                  f'r="{2.6 + 2.2*t:.1f}" fill="{ORANGE}" fill-opacity="{0.35 + 0.55*t:.2f}"/>')
        s.txt(px + pw / 2, py + pw + 26, lab, 20, MUTED, "600", "middle")
        if k < 2:
            s.arrow(px + pw + 6, py + pw / 2, px + pw + 28, py + pw / 2, ORANGE, "a_or", 2.6)
    s.txt(cx[1] + 24, y0 + 274, "Gaussian noise → shower cloud,", 20, INK, "600")
    s.txt(cx[1] + 24, y0 + 300, "in 50 Euler steps", 20, INK, "600")
    s.txt(cx[1] + 24, y0 + 334, "per cell:  Δη,  Δφ,  depth", 22, ORANGE, "700")
    s.txt(cx[1] + 24, y0 + 360, "i.i.d. given (c, globals) — no ordering", 19, MUTED, "500")

    # (3) EnergyHead
    s.card(cx[2], y0, cw, ch, "#ffffff", RED, 2.5)
    s.txt(cx[2] + 24, y0 + 44, "3.  how much energy each?", 26, RED, "700")
    s.txt(cx[2] + 24, y0 + 74, "EnergyHead — per cell log E", 20, MUTED, "500")
    hx, hy, hw, hh = cx[2] + 34, y0 + 108, cw - 68, 128
    s.card(hx, hy, hw, hh, PANEL, LINE, 1.6, 12)
    u0 = 0.20
    hp = []
    for i in range(60):
        u = u0 + (1 - u0) * i / 59
        v = math.exp(-((u - 0.46) ** 2) / 0.05)
        hp.append(f"{hx + 26 + u*(hw-44):.1f},{hy + hh - 8 - v*88:.1f}")
    xf = hx + 26 + u0 * (hw - 44)
    s.path(f"M{xf:.1f},{hy + hh - 8:.1f} L" + " L".join(hp), RED, None, 3)
    s.add(f'<line x1="{xf:.1f}" y1="{hy+10}" x2="{xf:.1f}" y2="{hy+hh-8}" stroke="{RED}" '
          f'stroke-width="2" stroke-dasharray="7 6"/>')
    s.txt(hx + 6, hy + hh + 26, "50 keV cut", 18, RED, "700")
    s.txt(hx + hw, hy + hh + 26, "log E →", 18, MUTED, "600", "end")
    s.txt(cx[2] + 24, y0 + 300, "at-floor Bernoulli + mixture", 22, RED, "700")
    s.txt(cx[2] + 24, y0 + 330, "zero-suppression truncates the", 19, MUTED, "500")
    s.txt(cx[2] + 24, y0 + 352, "spectrum; the edge is modelled", 19, MUTED, "500")
    s.txt(cx[2] + 24, y0 + 374, "separately from the bulk.", 19, MUTED, "500")

    # buses: c -> all three heads, anchor -> GlobalHead only
    for i in range(3):
        s.path(f"M{cx[i] + cw/2 - 30},{426} L{cx[i] + cw/2 - 30},{y0}", MUTED, "a_mut", 2.6)
    s.path(f"M{1126},{166} L{1146},{166} L{1146},{426} L{cx[0] + cw/2 - 30},{426} "
           f"M{1146},{426} L{cx[2] + cw/2 - 30},{426}", MUTED, None, 2.6)
    s.txt(1156, 420, "c", 21, MUTED, "700")
    s.path(f"M{976},{326} L{976},{408} L{cx[0] + cw/2 + 40},{408} L{cx[0] + cw/2 + 40},{y0}",
           ORANGE, "a_or", 2.6)
    s.txt(748, 402, "anchor → head 1 only", 18, ORANGE, "600")

    # --- row 3: output
    oy = y0 + ch + 34
    s.card(46, oy, 1428, 106, PANEL, LINE)
    s.txt(74, oy + 44, "shower", 26, INK, "700")
    s.txt(74, oy + 76, "N cells", 20, MUTED, "500")
    for i in range(46):
        s.add(f'<circle cx="{400 + rnd.gauss(0, 52):.1f}" cy="{oy + 54 + rnd.gauss(0, 20):.1f}" '
              f'r="{rnd.uniform(3,9):.1f}" fill="{ORANGE}" fill-opacity="0.85"/>')
    s.txt(560, oy + 46, "= anchor  +  core offset  +  per-cell (Δη, Δφ, depth),", 22, INK, "600")
    s.txt(560, oy + 76, "each carrying its own log E — a point cloud, never projected onto cells", 21, MUTED, "500")

    footer(s, H - 52)
    s.save("calo_arch.svg")


tracker()
calo()
