# 2026-08-27 — the calo is TWO detectors; the depth gradient is mostly physics, the depth AXIS is broken

- **Commit / branch**: `32d0cab` / `flow-response`
- **Job**: 13033802 (`jobs/calo_section_split.sh`, CPU only — no model, no generation)
- **New**: `scripts/calo_section_split.py`
- **Artifacts**: `plots/calo/metrics/section_split_{ms,ele}.json`
- Follows [one-point](2026-08-27-calo-energy-position-coupling-and-the-floor-that-isnt.md) ·
  [coherence](2026-08-27-calo-coherence-is-hadron-only.md)

## Hypothesis

The one-point entry reported rho(logE, depth) = +0.542 for protons as the longitudinal shower
profile. But the ColliderML calorimeter is an EM section plus a hadronic section, and cells in the
two are not comparable. Measured from `calo_geometry.json` + raw `calo_hits` (1500 events, 8.0M cells):

| region | det | depth p1..p99 (mm) | `<logE>` | layer pitch | E frac |
|---|---|---|---|---|---|
| barrel ECAL | 10 | -6 .. 228 | -7.97 | 5.050 mm | 0.161 |
| endcap ECAL | 9, 11 | -10 .. 227 | -8.07 | 5.050 mm | 0.559 |
| barrel HCAL | 13 | **388** .. 1446 | -6.86 | 51.000 mm | 0.008 |
| endcap HCAL | 12, 14 | **435** .. 2220 | -6.89 | 51.000 mm | 0.272 |

ECAL is 71.5% of energy, HCAL 28.5%, with a PHYSICAL GAP between them (barrel 228->388, endcap
227->435). Mean cell energy **steps ~3x** across it, because an HCAL cell integrates ~10x more
material. Deep cells are hot for a reason that has nothing to do with the shower, and
deep-penetrating species put more cells past the boundary — so the gradient may be geometry.

## Change

`scripts/calo_section_split.py` recomputes the SAME within-shower Spearman four ways: all cells;
each shower's ECAL cells only; its HCAL cells only; and all cells with log-E centred WITHIN ITS
SECTION first (removes exactly the step, keeps the full depth range). `point_layer >= 0` marks
endcap cells exactly; depth then separates sections with a guard band over the physical gap, and
band cells are DROPPED and counted rather than guessed.

## Result — the step is real (+1.12 logE = 3.06x), the guard band caught 0.0% of cells

| class | %HCAL | step | rho all | ECAL | HCAL | **rho step-rm** | % was geometry |
|---|---|---|---|---|---|---|---|
| p | 10.9% | +1.03 | +0.5421 | **+0.5179** | +0.2415 | **+0.5187** | **4%** |
| mu+ / mu- | ~19% | +1.05 | +0.457 / +0.440 | +0.284 / +0.255 | +0.385 / +0.385 | **+0.384 / +0.364** | 16% / 17% |
| pi+ / pi- | ~15% | +1.22 | +0.271 / +0.226 | +0.170 / +0.119 | +0.178 / +0.143 | **+0.200 / +0.148** | 26% / 35% |
| K+ / K- | ~17% | +1.19 | +0.290 / +0.185 | +0.167 / +0.079 | +0.135 / +0.077 | +0.197 / +0.105 | 32% / 43% |
| gamma | 8.3% | +1.06 | -0.1006 | -0.0846 | -0.2351 | **-0.1103** | -10% |
| e- / e+ | 2.5% | +0.91 | -0.013 / -0.009 | -0.037 / -0.032 | -0.062 / -0.064 | **-0.034 / -0.030** | -163% / -235% |
| pbar | 13.4% | +1.15 | +0.1163 | **+0.0013** | +0.0225 | +0.0309 | **73%** |
| nbar | 15.5% | +1.10 | -0.0066 | -0.0848 | -0.0302 | **-0.0845** | sign flip |
| n | 22.2% | +1.08 | +0.1292 | +0.0688 | +0.0536 | +0.0696 | 46% |
| **ALL pooled** | 10.3% | +1.12 | +0.0668 | +0.0250 | +0.0613 | +0.0316 | **53%** |

1. **The headline SURVIVES.** Protons keep +0.519 of +0.542, and it holds within the ECAL ALONE — a
   proton's interaction length far exceeds a radiation length, so its shower is still building
   through the EM section. Pions and kaons keep 57-74%, muons 83%.
2. **The POOLED number was the misleading one** (53% geometry). The all-species view understates
   twice over: species cancel AND the step contaminates.
3. **For e± and gamma the raw rho was TOO SMALL.** Their few hot HCAL cells dragged a genuine
   negative ECAL gradient toward zero: e± -0.013 -> **-0.034**, gamma -0.101 -> **-0.110**.
4. **The p / pbar asymmetry resolves.** Proton +0.518 within ECAL, antiproton **+0.001** — 73% of the
   antiproton's apparent gradient was pure sampling step. An antiproton annihilates on contact and
   dumps ~2 GeV promptly; a proton must interact and cascade. `nbar` shows the same flip. The raw
   statistic conflated matter and antimatter; the section split separates them.

## The DATA CONTRACT defect this exposed

`build_calo_slice_v2.depth_and_layer` takes ONE `(R0, Z0)` from `load_front_face()`
(`calo_geom.py:32`) for every detector, so:

- **Depth is ambiguous.** Barrel and endcap overlap on the same axis but transition at DIFFERENT
  depths (barrel HCAL starts 388, endcap 435). A cell at depth 400 mm is barrel HCAL or endcap
  dead-gap. At depth 150-250 mm the population is 12% barrel / 88% endcap.
- **`point_layer` is ambiguous too.** Layers are indexed per detector with no offset — 0..47 for
  dets 9/11, 0..35 for 12/14 — so **layer 20 could be EM endcap or hadronic endcap**.
- The model therefore must reproduce a 3x step whose LOCATION depends on a variable it is never
  given: barrel-vs-endcap is per-shower conditioning for the GlobalHead only, and `EnergyHead`
  never sees it.

**The calorimeter is also not phi-symmetric** (recorded in the one-point entry): endcap occupancy
modulates rms/mean 0.153 with a clean n=80 harmonic; barrel median r swings 22 mm with phi.

## Verdict — KEPT. The gradient is physics; the AXIS is broken and must be rebuilt

The model does have to learn the profile, but the 3x step is deterministic geometry it should be
GIVEN, and currently cannot be. Same class of defect as the tracker's v1 -> v3 fix, where replacing
a global continuous residual with (module token + local coords) took the matched pion gate
0.9996 -> 0.80.

## Next

- **Slice change, and it PRECEDES the head work**: store `(section token, depth-within-section)`
  instead of one global continuous depth. The detector id is in the source and the `calo_rz` sidecar
  already carries (r, z), so this is a rebuild, not a new data pass.
- Give `point_layer` a per-detector OFFSET so a layer index identifies its detector uniquely.
- Only then the head work: floor truncation + simplex fractions, decompose; then position +
  PDG-conditioned attention, decompose again.
