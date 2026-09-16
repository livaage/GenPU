# PIPELINE — what the code ACTUALLY does

_Derived by reading the code, not the plan. Last verified 2026-08-24 against commit `6b18b19`._

**Why this file exists.** On 2026-08-24 a single session found four wrong recorded assumptions, two
contaminated results, and a fundamental design divergence — every one of them a gap between what a
document said and what the code did. `STATUS.md` records *what we tried*; `pileup_generator_plan.md`
records *what we intended*. Neither records *what runs*. This does.

**Rule: if this file and any other document disagree, this file is the one to re-verify first, and
whichever is wrong gets fixed the same day.** Every claim below should be checkable against a named
file and line.

---

## 0. One-line summary

Truth particles in → per-particle detector response out. **No Stage-1 particle generator**: every
result to date conditions on real (Geant) particles, including all secondaries at their true
vertices. Two independent heads: tracker (tokenised autoregressive) and calo (flow-matching point
cloud). They do not talk to each other.

---

## 1. Raw source — CERN/ColliderML-Release-1

`/scratch/gpfs/IOJALVO/lv7805/genpu_cache/datasets/CERN___collider_ml-release-1/`
Per-EVENT list columns, IPC stream format. Load with `genpu.data.load_shard(subset, i)`.

| subset | columns | on disk? |
|---|---|---|
| `particles` | event_id, particle_id, pdg_id, mass, energy, charge, vx, vy, vz, time, px, py, pz, perigee_d0, perigee_z0, vertex_primary, parent_id, primary | yes |
| `tracker_hits` | event_id, x, y, z, true_x/y/z, time, particle_id, detector, volume_id, layer_id, surface_id | yes |
| `calo_hits` | event_id, detector, total_energy, **x, y, z**, contrib_particle_ids, contrib_energies, contrib_times | yes |
| `tracks` | — | **NOT DOWNLOADED** (reco tracks; the deferred ACTS validation needs it) |

**Joins**: by `event_id`, never row index. The subsets are NOT in the same row order — measured
agreement 0.000. `genpu.data.build_event_index` is the correct tool.

**Not in the source**: the Geant4 **creation process** (conversion / brem / hadronic / decay). Any
cascade model must infer it from (parent pdg, daughter pdg set).

---

## 2. `preprocessing.py` — the input boundary, and where information is lost

Per event, attributes hits to particles by `particle_id` (`searchsorted`, not a positional join).

### What it keeps

| output | contents |
|---|---|
| `particle_features` (N,6) | log_pt, eta, phi, **pdg_class (17-way collapse)**, charge, mass |
| `particle_aux` (N,6) | primary, parent_id, vx, vy, vz, energy |
| `tracker_hits_flat` (M,5) | **layer_class (0..47)**, r, phi, z, time |
| `calo_hits_flat` (M,5) | eta, phi, log(contrib_energy), contrib_frac, detector |
| `tracker_offsets`, `calo_offsets`, `event_ids`, `particle_ids` | CSR layout + keys |

### What it DROPS — read this before assuming a quantity exists

- **calo depth.** `(x,y,z)` → `(eta,phi)`; the radial/longitudinal coordinate is discarded at
  line ~212. `detector` (6 values) is the only surviving depth proxy. **This is why the calo model
  is 2D** — see §5.
- **tracker `surface_id`** and `true_x/y/z`. `(volume,layer)` is collapsed to 48 `layer_class`
  values, so module identity is gone (the v3 surface-local work had to rebuild it separately via
  `module_geometry.py`).
- **particles with no hits.** `visible_mask = (n_tracker_hits > 0) | (n_calo_hits > 0)`. Drops
  **24.2%** of particles and **2,586 of 10,000 events** entirely. Consequence: the parent→child
  graph CANNOT be rebuilt from stage2 — 16.1% of visible secondaries have an invisible direct
  parent and 48.6% of chains to a primary cross an invisible node. Use the cascade graph (§3).
- **pdg identity beyond 17 classes.** Everything unusual becomes class 16 — 6% of particles,
  9,561 distinct masses up to 27.9 GeV (nuclei). `mass` is the only surviving discriminator there.

### Row order within a particle is NOT reproducible in the existing shards

`preprocessing.py` grouped hits with `np.argsort(pid)`, which defaults to quicksort and is **not
stable**, so ties (hits sharing a particle_id) came out in arbitrary order. Fixed to
`kind="stable"` on 2026-08-24, but **the shards on disk were written with the unstable sort**, so:

- nothing may assume a within-particle row order in stage2 (it is arbitrary anyway — see §4);
- anything aligning new data to `calo_hits_flat` must match **by value**, not by row index.
  `build_calo_depth.py` lexsorts on (eta, phi, logE) and pairs the blocks up.

It never mattered before because no consumer depended on order — but it means the existing shards
cannot be byte-reproduced, only reproduced as sets.

### Units of the stored calo record

`calo_hits_flat` rows are **(cell, particle) CONTRIBUTION pairs, not cells.** A cell with 3
contributors appears 3 times, each carrying that particle's share. Mean 1.206 contributors per cell;
89.3% of cells have one. This is why nine of the ten calo gate features are partition-dependent (§6).

---

## 3. Derived artifacts

| artifact | built by | contents |
|---|---|---|
| `shard_XXXX_stage2.npz` (4 shards: **0,1,2,5**) | `preprocessing.py` | above |
| `shard_XXXX_graph.npz` | `build_cascade_graph.py` | **every RAW particle** (visible or not): 17 source columns + n_tracker_hits, n_calo_hits, calo_energy_sum, r_innermost, r_outermost, depth, root_primary_id, visible |
| `shard_XXXX_pids.npz` | same | join key, stage2 row order |
| `shard_XXXX_calo_rz.npz` | `build_calo_depth.py` | per-contribution `(r, z)` of the cell, row-aligned with `calo_hits_flat` — the longitudinal coordinate preprocessing drops |
| `tracker_slice/*.npz` | `build_tracker_slice.py` | cont (S,7), pdg, hits (P,5), offsets |
| `calo_slice/*.npz` (**v1**) | `build_calo_slice.py` | cont (S,7), glob (S,4), points_flat **(P,3)**, offsets, anchor, anchor_mode |
| `shard_XXXX_calo_rz.npz` | `build_calo_depth.py` | per-contribution `(r, z)`, row-aligned to `calo_hits_flat` |
| `calo_slice/*_v2.npz` (**v2**) | `build_calo_slice_v2.py` | as v1 **plus** points_flat **(P,4)** with depth at index 2 and **energy at index -1**, `point_layer` (int16, -1 = barrel), `point_depth_local`/`point_region`/`point_det` (2026-08-27), `n_src`, `event_id`, `p_phi`, `E_true` |
| `gate_features/*.npz` | `calo_metrics.py --dump_features` | PAIRED per-event gate features `Xr`/`Xg` (E,12), `event_id`, `n_src`, `feature_names` — row k is the same event on both sides |

**Shards are 0, 1, 2, 5** — 3 and 4 were never preprocessed.

**v2 column order is (d_eta, d_phi, depth, log_e).** Energy is the LAST column, never index 2 — index 2 is DEPTH. Two scripts have already been caught reading `[:, 2]` as energy (`train_calo_flow.py`, fixed 2026-08-24; `calo_metrics.py --logE_max_slice`, fixed 2026-08-25), and neither failure was loud: one bounds the energy head by a millimetre, the other by a depth. Read the last column.

**THE CALORIMETER IS TWO DETECTORS, and `depth` cannot say which** (measured 2026-08-27, job
13033802, `scripts/calo_section_split.py`):

| region | det | depth p1..p99 (mm) | `<logE>` | layer pitch | E frac | cells (e±) |
|---|---|---|---|---|---|---|
| barrel ECAL | 10 | -6 .. 228 | -7.97 | 5.050 mm | 0.161 | 17.3% |
| endcap ECAL | 9, 11 | -10 .. 227 | -8.07 | 5.050 mm | 0.559 | 80.3% |
| barrel HCAL | 13 | **388** .. 1446 | -6.86 | 51.000 mm | 0.008 | ~0% |
| endcap HCAL | 12, 14 | **435** .. 2220 | -6.89 | 51.000 mm | 0.272 | 2.4% |

ECAL is **71.5%** of energy, HCAL **28.5%**, with a PHYSICAL GAP between them. Mean cell log-E
**steps by +1.12 (3.06x)** across it, because an HCAL cell integrates ~10x more material. `depth`
(points_flat index 2) is measured from a SINGLE front face (`calo_geom.load_front_face`), so barrel
and endcap **overlap on it while transitioning at different depths (388 vs 435 mm)** — a cell at
depth 400 mm is barrel HCAL or endcap dead-gap, and at depth 150-250 mm the population is 12%
barrel / 88% endcap. The model must reproduce a 3x step whose location depends on a variable it is
never given.

**Added 2026-08-27, ALONGSIDE the existing columns** (`points_flat` is byte-identical — verified:
`max|depth_global - depth_local| = 0.000000 mm` over 775,818 ECAL cells, and HCAL differs by one
exact constant per detector, 389.396 barrel / 435.000 endcap):
- `point_depth_local` (P,) float32 — depth from the cell's **own** detector front face
- `point_region` (P,) int8 — 0 barrel-ECAL / 1 endcap-ECAL / 2 barrel-HCAL / 3 endcap-HCAL
- `point_det` (P,) int8 — raw detector id (9,10,11,12,13,14)
- `point_layer` is now **globally unique** (per-detector base offsets; det 9 -> 0..47, 11 -> 48..95,
  12 -> 96..131, 14 -> 132..167). Before this every endcap detector numbered from 0, so **layer 20
  was EM endcap OR hadronic endcap** — two detectors whose cells differ 3x in mean energy. Nothing
  read the field, so the renumbering breaks no recorded result.

**THE ZERO-SUPPRESSION FLOOR IS A TRUNCATION, NOT A PILE** (measured 2026-08-27 from raw
`calo_hits.total_energy`: min>0 = 5.0001e-05 GeV, frac below 5e-5 = **0.00000**, 1,493,002 distinct
values in 1,528,778 cells). The 50 keV cutoff is real and applies to the **cell total**, but nothing
accumulates AT it: **zero** cells sit at log(5e-5) in either a v1 or a v2 slice. Per-shower
ATTRIBUTED energy is a *share* of the cell total, so shared cells fall below the threshold — a
sub-floor tail of 1.06% (v1) / 0.39% (v2). What the slice shows at the threshold is a **density
step** (~17x), the truncation edge of the unshared population.

Consequence for `EnergyHead`: it models this region as an at-floor Bernoulli pinning cells to
exactly `log_floor` (`calo_flow.py:229`), i.e. it places ~0.56% of its cells at a value the data
never contains. And the metric and the mechanism measure different populations — the gate's
`frac_near_floor` band (`logE < LF+0.5`) holds **3.79%** of e± cells while the head's trained band
(`|logE-LF|<0.05`) holds **0.30%**, so **92% of what the gate scores is drawn by the MIXTURE, not
the floor logit**. `--floor_n_buckets` only ever touched the logit (`calo_flow.py:205`).

**THE CALORIMETER IS NOT phi-SYMMETRIC**, contrary to the stated justification for omitting phi,
vx, vy from the `cont` contract. Endcap occupancy modulates rms/mean **0.153** with a clean **n=80**
harmonic (module segmentation; min-bias pileup is phi-uniform on average, so this is geometry);
barrel median r swings **22 mm** with phi (stave polygon). Harmless while the model emits continuous
offsets and produces no cell-level geometry — **binding** once gap #1's cell projection exists.

**`n_src` counts DISTINCT DEPOSITORS merged into the shower** (1 = nothing merged). Until 2026-08-25 it counted deduped CELLS, so every build reported "merged sources/shower" exactly equal to cells/shower. No model or metric reads it; slices built before that date carry the wrong value in this field.

---

## 4. Slice builders — every one filters to n >= 1

- `build_tracker_slice.py` — `--min_hits` default 1. **Sorts hits r-ascending here**, NOT in
  preprocessing (stage2 order is arbitrary: Spearman(index, r) = 0.01).
- `build_calo_slice.py` — `>= 1` calo deposit, one pdg class. Groups by **direct depositing
  particle**, so 64.8% of "showers" are fragments born inside the calorimeter (§6).
- `build_count_slice_stage2.py` — `keep = nh >= 1`. Computes `d0` **geometrically**
  (`vx*sin(phi) - vy*cos(phi)`) because the source `perigee_d0` is non-finite for 56% of charged
  particles.

Shared conditioning: `CONT_FEATURES = [log_pt, eta, log_E, charge, mass, vr, vz]`
(`genpu/conditioning.py`). phi excluded — response is phi-symmetric.

---

## 5. Models — what they actually generate

| model | file | generates |
|---|---|---|
| Tracker v1/v3 | `models/tracker_ar.py` | layer_class + standardised (r,phi,z,time) residual bins, inner→outer |
| Tracker v2 | `models/tracker_state_ar.py` | + direction state, STOP head |
| Count head | `models/count_head.py` | **categorical over 1..48** — structurally cannot emit 0 |
| Calo flow | `flow/calo_flow.py` | GlobalHead (total_logE, log_n, core_eta, core_phi) + PointFlow + EnergyHead |
| Incidence | `train_incidence_head.py` | P(tracker trace), P(calo trace) — shared trunk, 2 Bernoullis |

**THE TRACKER HEADS ARE FACTORISED GIVEN THE STEP'S HIDDEN STATE, AND ONE OF THEM SHOULD NOT BE**
(measured 2026-09-07, login node, `tracker_slice/surface_pion.npz`, 16,400,726 hits / 18,824 modules).
In `tracker_module_ar.generate` every head is a linear map of the SAME vector `h`: layer, surface,
x, y, z, time are drawn independently, and the only coupling is the surface head's additive -inf
validity mask on the sampled layer (`_masked_surface_logits`, line 123). The module never reaches
the coordinate heads; it enters afterwards, deterministically, as the frame the standardised
residual is un-standardised into (`resid * mod_std[module] + mod_mean[module]`).

Two of the three things that could go wrong measure clean, one does not:

| | measured |
|---|---|
| module identity explains the local offset MEAN | **eta^2 = 0.02-0.03%** — nothing (between/sampling ratio 1.2-1.3x) |
| within-module correlation between x, y, z | **rho = 0.001** — the factorisation across coordinates is fine |
| module identity sets the local offset WIDTH | **local z is degenerate in 39% of modules holding 49.5% of hits** |

The z case is real but **physically negligible — do NOT "fix" it.** Per-module std of the
standardised local z: p10 = **0.00**, median 0.98; hits in those modules are at `|z_res| < 0.05`
**100.0%** of the time against **5.1%** elsewhere, and the split is exactly barrel vs endcap
(**46 of 47** layer classes are pure). But the standardised residual is multiplied by
`module_std[:,2]`, and for **100.0% of endcap modules that value sits at the 0.05 mm std_floor**
(`build_module_geometry.py --std_floor`), because an endcap module is a planar sensor at fixed z.
So a *completely* wrong local-z bin displaces an endcap hit by at most **3 x 0.05 = 0.15 mm** —
against a median module scale of 26 mm in x/y and well below hit resolution. The model is
predicting a coordinate that carries no information, and the un-standardisation throws the error
away by construction. Masking the z head on the sampled layer would buy ~0.15 mm and cost a
special case in the generation path; **the uniform code path is worth more.**
(For contrast, where local z IS free — barrel modules, median std_z **20.8 mm** — it is not
degenerate, so the head is doing real work exactly where it matters. 20.8% of barrel modules are
also at the z floor.)

**`--pos_transform quantile` is unusable on a 3-D calo slice** (read 2026-09-07): the trainer fits
`qt_pos_x = np.quantile(pts[:, :2], ...)` on TWO columns (`train_calo_flow.py:179`) while `std_pos`
calls `_qt_fwd` over `x.shape[1]` = 3 columns (`calo_flow.py:531`), so it raises IndexError on the
depth column. Loud, not silent — but it means the transform the trainer's own help text calls
"recommended" is available only to the 2-D v1 models.

**Calo cells within a shower are drawn i.i.d. given the conditioning** (`PointCFM.sample`,
`calo_flow.py:114`: one `randn` per cell, a velocity field that sees only that cell). Consequences,
all measured on `multispecies_v2_s0` over 200k showers:
- after projection onto real cells, **0.0357 of generated points collide** (two in one channel)
  against a real floor of **0.0003** — real showers are sets of DISTINCT cells and cannot collide;
- the collisions are core-localised (0.118 within r/width < 0.25, 0.000 beyond 5 widths);
- they are NOT a density excess: within-shower NN spacing is **2.14x real** (0.00674 vs 0.00314,
  2-D statistic, see caveat in the 2026-09-16 entry). Tuning core density does not address it.

**Set attention and joint energy (A-prime, 2026-09-16) — opt-in, no trained model yet.**
`train_calo_flow.py --point_attn L` makes cells of a shower attend to each other
(`SetAttnBlock`, size-bucketed by `SetBatcher`); the flow time `t` is shared per shower, and training
switches from random POINTS to whole-SHOWER batches (size classes, drawn in proportion to cell count).
`--attn_self_only` is the matched control: same network, attention masked to the diagonal.
`--joint_energy` appends standardised log-E as a flow coordinate and stops training `EnergyHead`;
at generation `logE` comes from the flow, is bounded above by `logE_max_pdg` as before, and cells
below the 50 keV threshold are **DROPPED** (`n_zero_suppressed` in the output), never clamped —
clamping to the floor is what sank the original energy-in-flow model. With `--joint_energy`,
`--energy_pos` and the EnergyHead region/position inputs are unused. `from_checkpoint` infers all of
this from the state dict; every earlier checkpoint loads unchanged (verified on `electron_v2_s0`).

**Cell count per shower is BOUNDED per species** (2026-09-15). `sample_showers` uses
`min(max_cells, n_max_pdg[pdg])`; `n_max_pdg` is the largest shower each class produced in training,
written by `train_calo_flow.py` (same contract as `logE_max_pdg`). Checkpoints older than 2026-09-15
lack the buffer (-> inf), so for them `max_cells` alone governs, and **its default is still 128**.
At 128 the sampler destroys 6.87% of all cells on the multispecies slice, hadron-selectively (pbar
21.3%, mu±/pi0 0%), and `partition=True` repacks their energy into the survivors. Pass
`--max_cells 4096` when evaluating any pre-2026-09-15 checkpoint. `sample_showers` returns `n_trunc`.

**Calo dimensionality is set by the SLICE** (2026-08-24). `CaloFlow(pos_dim=...)` — 2 for a v1
(P,3) slice, 3 for a v2 (P,4) slice — and `from_checkpoint` infers it from `PointCFM`'s output
layer, so old checkpoints stay loadable. Before this the model was 2D unconditionally, which is why
the plan's depth metrics were never computable.

**ENERGY IS THE LAST POINT COLUMN, always.** Index 2 is `logE` in v1 and **DEPTH** in v2. A hardcoded
`pts[:, 2]` bounded the energy head with a depth value in millimetres (`logE_max: 2220.00`) without
erroring — see the entry for 2026-08-24. Read `pts[:, -1]`.

---

## 6. Metrics — what they measure, and what they cannot

`calo_metrics.py` event gate: pools an event's cells and computes 10 features. Partition-invariance:

| feature | invariant? |
|---|---|
| `log_totE` | **yes** |
| `n_cells`, `logE_mean/std/max/p90`, `frac_near_floor`, `cells_per_src`, `width_mean/std` | **no** — computed over contributions, or per-shower |

So any change to how particles are partitioned moves 9 of 10 features regardless of model quality.

**`--real_slice` (2026-08-24)** makes the real reference come from a v2 slice, so shower attribution
has ONE implementation shared by training and evaluation. Without it a v2-trained model is scored
against v1-attributed real showers, 64.8% of which are fragments. The anchor is then read FROM the
slice rather than recomputed — recomputing from stage2 rows uses the DEPOSITOR, not the
calo-incident ancestor, silently undoing re-attribution.

**`--pdg_class` with `--real_slice` (2026-08-25)** now filters and compacts the slice's CSR arrays.
Before that it was silently ignored on the `--real_slice` path (`sel = np.arange(len(cont))`), which
was harmless while every v2 slice was single-species and would have turned every per-species number
on a multispecies slice into an all-species one.

**`--dump_features` (2026-08-25)** writes the paired per-event feature matrices for
`calo_gate_diagnose.py`. That script decomposes the composite gate into the part explained by the
MARGINALS (permute each column within its class) and the part explained by the JOINT (rank-transform
each column within its class), with a quadratic-logistic probe splitting the joint part into the
correlation matrix and everything beyond it, plus a linear-logistic control that must return 0.5.
**A composite AUC far above every marginal does not by itself imply a joint defect** — k features
each slightly off aggregate to roughly `Phi(sqrt(sum_i d_i^2)/sqrt(2))` with no joint error at all,
and the script prints that prediction next to the measurement. Measured on e± v2: full 0.825/0.833,
marginals 0.601/0.617 (aggregation predicts 0.627), joint 0.700/0.724.

**`d_phi` is WRAPPED on the generated side (fixed 2026-09-15).** `g_dp` was `gen_phi - p_phi`
unwrapped while the real side has always been wrapped (`build_calo_slice_v2.py:292`). Not a no-op:
the helix anchor moves pion CORES by ~1.5 rad, so core+pos can exceed pi. **Every width/d_phi number
before 2026-09-15 is affected** — `width_std` AUC 0.5016 was really 0.6953 (never at chance),
`width_mean` 0.5618 -> 0.5100, `shower_width` W/sigma 0.0344 -> 0.0505.

**`--max_cells` (2026-09-15)** sets the sampler's cell-count cap (default 128, see §5) and the JSON
reports `n_trunc_frac_gen`. Real rate above 128 on the multispecies slice is 2.10% of showers.

**`--snap_cells` (2026-09-15)** projects generated points onto real ODD cells and MERGES points that
land in the same cell within a shower (`calo_cells.snap_and_merge`), as a separate deterministic step
after the continuous sample. Under the flag: depth observables are DROPPED (a snapped depth is a layer
index, not the continuous coordinate the real side carries), so `event_gate_auc_depth` is null and the
decomposition's `full` has 10 features instead of 12 — **compare `event_gate_auc`, not `full`,
across the flag.** The JSON carries `cell_projection` (merged fraction, energy ratio, co-occupancy).
First result: `event_gate_auc` 0.9357 -> 0.9458, cost entirely in energy features.

**The composite gate saturates.** Fixing the 128 cap moved five marginals toward chance and both the
marginal (0.842 -> 0.743) and copula (0.845 -> 0.776) halves, while `event_gate_auc` stayed flat
(0.933 -> 0.936). Near 0.95 the classifier has redundant paths; report the decomposition, not only the
composite, when judging a change.

**Depth observables (2026-08-24)**, present only when both sides have depth (guarded, with a warning
otherwise): `event_gate_auc_depth` (+ energy-weighted `depth_mean`/`depth_std`), `cell_depth` and
`shower_depth` Wassersteins, and a **longitudinal profile** — the acceptance metric
`pileup_generator_plan.md:356` names.

---

## 7. KNOWN GAPS — specified but NOT built

1. ~~**Calo depth / projection-to-cells.**~~ **BUILT.** Depth since 2026-08-24 (v2 slice, 3-D model).
   Projection since 2026-09-15: `src/genpu/calo_cells.py` (`etaphidepth_to_xyz`, `snap_cells`,
   `cell_ids`, `snap_and_merge`), constants from the ODD XML
   (`github.com/OpenDataDetector/OpenDataDetector`: 16-fold, phase 11.25 deg, cells 5.1/30 mm,
   48/36 layers at 5.050/51.0 mm, sensitive inset 2.40/47.5 mm). Real cells snap to themselves
   0.99698 within 10 um; round trip from (eta, phi, depth) recovers the same cell id 0.9925 (the
   residual is mostly the |eta| < 1.60 barrel/endcap guess, 0.9950 correct). Wired in as
   `calo_metrics.py --snap_cells`. **Still missing:** boundary assignment (HCAL endcap ~2.5% and
   barrels 0.2-0.5% of cells land in the neighbouring face), and the layer-wise energy-fraction
   metric the plan names (line 356) — the projection makes it computable but nobody has written it.
2. **Cell-level metric.** ~~Blocked on (1)~~ **UNBLOCKED 2026-09-15** — `src/genpu/calo_cells.py`
   snaps every calo point to a real cell centre: **99.698%** of 4.29M real cell-hits land within
   10 um, ECAL endcap ids bijective (ratio 1.0000).
   > **RETRACTED 2026-09-15 — the barrel claim below is WRONG.** Radius is the wrong coordinate: a
   > barrel stave is a FLAT plate, so depth is the PERPENDICULAR distance to the stave plane, and
   > radius blends depth with the across-stave position. In the stave frame all three coordinates are
   > exactly discrete — det 10 R(z) 1.0000 @5.1, R(along) 0.9979 @5.1, R(perp) 0.9998 @5.050 -> 48
   > layers; det 13 the same at 30/30/51.0 -> 36 layers. The barrel is ~18% of ECAL cells / 17.3% of
   > energy and was written off on a coordinate choice.
   > [entry](experiment-memory/2026-09-15-calo-snap-to-cell-built-and-barrel-gap2-RETRACTED.md)

   ~~and **exact cell identity is unreachable for barrel EM**~~:
   `calo_hits` carries no layer id, and barrel `r` cannot recover the layer because staves tile a
   cylinder so cells from different layers overlap in radius (max gap 0.985 mm over a 106 mm span).
   Endcaps are exact — dets 9/11 have exactly 48 layers at 5.050 mm, dets 12/14 exactly 36 at
   51.000 mm, 100% of cells on the grid, ~83% of deposited energy.

   **2a. THE CELL VOCABULARY IS ~2e7 — a discrete cell-ID head is not on the table** (measured
   2026-09-07, login node, raw `calo_hits` shard 0, first 4,000 events). 5,458 cell-hits/event;
   **11,831,448 DISTINCT cell positions** seen, still growing at 2,355 new cells/event at event
   4,000 (43% of every event's cells are still unseen). Uniform-occupancy coupon-collector puts the
   total near **2e7**, and occupancy is very non-uniform so that is a LOWER bound.
   - Positions ARE discrete, not energy-weighted continuous: quantising 0.1 mm -> 5 mm merges only
     2.4% of them (4,141,073 -> 4,041,381 over 1,000 events), and 10 mm merges 50% — the signature
     of a ~5-10 mm transverse pitch. Longitudinally the endcaps are exact (det 9/11: **48** distinct
     z; det 12/14: **36**), the barrel is not (det 10: 599 distinct |z|).
   - So the plan's stated reason for going continuous — "a cell-ID vocabulary couples the model to
     one geometry" (`pileup_generator_plan.md:108`) — is the WEAK argument for this project: we have
     exactly one geometry and are not transferring. The decisive argument is the one nobody made:
     **~2e7 classes vs the tracker's 18,824 modules**, against ~5 cells/shower of signal per
     particle. Even hierarchically factored, the bottom level is ~1e5-1e6 wide.
   - The surviving a-priori argument for continuous output is unchanged and independent of this:
     reconstructed energy is a SUM over cells on a steeply-falling spectrum, so per-cell
     quantisation error accumulates into a resolution bias rather than averaging out.
3. **Tracker incidence** — the head exists (`train_incidence_head.py`, AUC 0.9993/0.9890, ECE ~5e-4,
   3-way joint reproduced) but is not wired into any generation path.
3e. **THE CALO'S DEPTH AXIS IS DISCRETE WHERE MOST OF THE ENERGY IS, AND THE FLOW EMITS IT
   CONTINUOUS** (measured 2026-09-07, login node, `calo_slice/electron_v2_h5.npz`, 13,054,567 cells).
   This is the calorimeter analogue of the tracker's degenerate local-z, and unlike that one it is
   NOT negligible.

   | region | cells | energy | distinct depths | median off-grid | on a plane (<0.1 mm) |
   |---|---|---|---|---|---|
   | endcap ECAL | 79.6% | 76.0% | **48** (5.050 mm pitch) | **0.000 mm** | **100.0%** |
   | endcap HCAL | 2.4% | 6.7% | **36** (51.000 mm pitch) | **0.000 mm** | **100.0%** |
   | barrel ECAL | 18.0% | 17.3% | 2,737 | 1.251 mm | 4.6% |
   | barrel HCAL | 0.0% | 0.0% | — | — | — |

   So for **~82% of cells / ~83% of energy** (e±; species mix shifts this) depth is a ladder of 48 or
   36 planes, while `PointCFM` emits a continuous coordinate and **nothing snaps it** —
   `sample_showers` feeds the raw sampled depth to `depth_to_region_local` and out to the metrics.
   Generated cells therefore sit between planes, at depths the detector cannot produce. The barrel is
   genuinely continuous (staves tile a cylinder), so the right treatment differs by region — the same
   split as gap #2.

   NOT measured: how far off-grid generated depths actually land, which needs sampling from a
   checkpoint (GPU job). Note the existing depth metrics would barely see it: a Wasserstein between a
   comb and a comb smeared by < pitch/2 is small, so `cell_depth` W can look healthy while every
   generated cell is off-plane. Cheapest fix if it matters: snap endcap depths to the nearest plane
   at generation — deterministic, and the region assignment it needs is already computed there.

3b. **Depth is not used for CELL IDENTITY.** The v2 slice carries depth and layer, but the gate is
   still contribution-level: re-attribution deduplicates WITHIN a shower, while two incident
   particles landing in the same cell still double-count (3.5% at PU0, far more at M3's mu).
3c. ~~**v2 exists for e± ONLY.**~~ RESOLVED 2026-08-25 — `multispecies_v2.npz` covers all 17 classes.
3d. ~~**The energy head is blind to cell position AND to detector section.**~~ **RESOLVED in the
   code** — `EnergyHead.__init__` now takes `pos_dim` / `n_regions`, `_out` concatenates the PER-CELL
   `pos_feat` (`POS_FEAT_NAMES = (depth_local, r_over_width)`) and a 4-way region embedding, and
   `sample_showers` derives both at generation from the sampled depth plus the cell's own eta via
   `calo_geom.depth_to_region_local` (no new sampled quantity). Enabled by `train_calo_flow.py
   --energy_pos`. The defect it fixes was real: with per-shower-only inputs every cell of a shower
   got a bit-identical vector, so rho(logE, depth) was **0.000** generated against +0.542 (p),
   +0.46 (mu±), +0.27 (pi±), -0.101 (gamma) real. **Still open: whether a TRAINED checkpoint with
   `--energy_pos` reproduces those correlations** — the wiring exists, the measurement does not.
4. **Stage-1 / cascade generator.** Architecture settled (one recursive model on
   `[log_E, eta, vr, vz, pdg]`, ~3-4 batched levels) but not built.

   **4a. THE CONDITIONING GIVES AWAY GEANT'S ANSWER — accepted for now, must be fixed before any
   fast-simulation claim** (scoped explicitly 2026-09-07; the simplification is deliberate, this
   entry is the record of what it costs).
   - Every head conditions on the **truth particle list including secondaries at their true
     production vertices**: `CONT_FEATURES = [log_pt, eta, log_E, charge, mass, vr, vz]`
     (`genpu/conditioning.py`), and tracker v1 additionally seeds the sequence with a `(vr, vz)`
     token (`tracker_ar.py:_vertex_embed`).
   - **"Secondary" here means `primary == 0` in the source `particles` table** — nothing else.
     `vertex_primary` is **1 for every particle in the file** (PU0 = one interaction vertex), so it
     cannot be the discriminator, and the cascade graph's `depth > 0` is *derived from* the same
     flag, so it is not independent corroboration. Primaries are beamline particles (92.9% at
     vr < 1 mm, median vr 0.02 mm); non-primaries are Geant4's own products (median vr 424 mm).

     **LABEL IT PRECISELY — this is NOT the standard collider definition.** Ours is a *simulation*
     flag: primary = handed to Geant4 by the generator, secondary = created by Geant4 during
     transport. The analysis definition ([ATLAS min-bias, arXiv:1602.01633]) is a *lifetime* rule:
     a primary charged particle has tau > 300 ps and comes either directly from the pp collision or
     from a decay of a directly-produced particle with tau < 30 ps; secondaries are material
     interactions, photon conversions and decays of tau > 30 ps particles. Say **"Geant4-made"** or
     **"produced during transport"**, not bare "secondary", and always name the population the
     percentage is over.
     - The two definitions *mostly* agree here, which is why the numbers are still meaningful:
       K0_S / Lambda / K0_L are present and 83-88% flagged primary, so **Geant does their decays and
       their daughters land on our secondary side — the same side ATLAS puts them on**. The
       divergence is only the tau < 30 ps parents (pi0, eta, rho, omega) whose daughters ATLAS calls
       prompt: **20.3/event = 3.22% of our secondaries, carrying 0.61% of tracker hits** (measured
       2026-09-07).
     - Our definition also has no charge or fiducial cut, and it counts calorimeter shower products
       as particles — two categories that simply do not exist in a tracking analysis.

   - **EXTERNAL CONTROL — apply the ATLAS selection and we land within a factor 2** (measured
     2026-09-07; charged, pT > 500 MeV, |eta| < 2.5, >= 7 silicon hits, geometric d0):

     | | secondary fraction |
     |---|---|
     | ours, no d0 cut | 20.1% |
     | ours, \|d0\| < 1.5 mm (the ATLAS cut) | **4.4%** |
     | ATLAS 13 TeV min-bias, same selection | **(2.3 +- 0.6)%** |

     The residual factor ~2 is expected: ours are truth particles, not reconstructed tracks (real
     reconstruction loses large-d0 secondaries to seeding and quality cuts), the definitions differ
     as above, and the detectors differ. Note ATLAS scales its *simulated* secondaries UP by
     1.38 +- 0.14 to match data, i.e. Geant under-predicts them. Cross-check on the same selection:
     our primaries leave **13.8 silicon hits (median 13)**, against ATLAS's 11-14 across eta and the
     ODD paper's ">= 11 measurements within |eta| < 3" — the geometry is behaving normally.
     ODD passive material is up to **~2 X0 and ~0.5 nuclear interaction lengths** at high |eta|
     (ODD tracking-system paper, Figs. 9-10), so a large Geant4-made population is expected.
   - **Re-measured 2026-09-07** (login node, `cascade_graph/shard_0000_graph.npz`, 10,000 events,
     8,553,348 raw particles = 855/event: 224 primary, 631 non-primary):

     | quantity | measured | previously recorded |
     |---|---|---|
     | particles that are primary | **26.2%** | ~27% ✓ |
     | tracker hits from non-primaries | **59.7%** | 61.4% ✗ |
     | calo energy raw-attributed to non-primaries | **83.5%** | 84.4% ✗ |
     | calo energy conditioned on a non-primary, **after v2 re-attribution** | **45.7%** | never measured |

   - **The 84% figure counts particles the calo head never conditions on.** 58.4% of all calo energy
     is deposited by particles *born inside the calorimeter* (299/event, 48% of all non-primaries) —
     shower products, which `build_calo_slice_v2.py` lifts to the calo-incident ancestor
     (`g_inside = (gvr >= R0) | (gaz >= Z0)`, line 155). After that lift the calo head sees **167
     incident particles/event, 30.1% of them primary**, and **45.7%** of the energy is conditioned on
     a Geant secondary (median production radius 319 mm). Quote 83.5% only for the **v1** slice,
     which groups by the direct depositor.
   - The tracker has no re-attribution — hits are grouped by the depositing particle — so **59.7%**
     is the right number there. **But do not quote 59.7% to a tracking audience without the pT
     qualifier** (measured 2026-09-07, same shard): hits from particles with **pT > 100 MeV are
     39.6%** secondary, **pT > 400 MeV → 22.8%**, **pT > 1 GeV → 6.8%**. The pT > 400 MeV cut keeps
     only 27.9% of all hits, i.e. **72% of every event's tracker hits come from sub-400-MeV
     particles**. A primary that leaves hits leaves **11** of them (median 11 — the canonical ODD/ITk
     silicon count); a secondary that leaves hits leaves a **median of 1** (mean 3.31, 58.8% leave
     exactly one). Secondaries dominate the hit COUNT because they are numerous and mostly deposit
     once, not because they dominate reconstructible tracks.
   - **Two further reasons the raw fractions look higher than a physicist expects.** (i) The primary
     denominator is the FULL generator record over all eta: 224/event, of which **89/event have
     |eta| > 4** and never reach the detector, and only **44/event leave a tracker hit**. (ii) **47%
     of "secondaries" (299/event) are born INSIDE the calorimeter** — EM shower products, e+/e- ratio
     **0.95** (pair production), which nobody would call event secondaries. Tracker-born secondaries
     are 332/event with e+/e- **0.25** (delta rays + Compton, plus ~42 genuine conversion e+/event).
     It is NOT a Geant production-cut artifact: at **E > 100 MeV the list is still 68.7% secondary**,
     and only 23.8% of secondaries are below 100 MeV. So for a secondary born at vr = 420 mm we hand the model the fact
     that a material interaction of that kind happened at that point — which is Geant4's OUTPUT,
     not something derivable from a Pythia event.
   - **Consequence for the write-up**: in its present form this is not a Geant4 replacement. It is
     the hit-generation / shower-development step CONDITIONED ON Geant4's particle list. The gate
     numbers are honest for that object and only that object. Do not report them as fast-simulation
     performance.
   - Two further inflations pointing the same way: slices filter to `n >= 1` on both subsystems
     while only **40.4%** of particles leave a tracker hit and 68.7% a calo deposit — the incidence
     heads that would model this exist (gap #3) and are **not wired into any generation path**; and
     the calo gate double-counts cells shared by two incident particles (3.5% at PU0, worse at M3's
     mu, gap #3b).
   - **The fix is gap #4 itself** (primaries-in with a learned cascade) plus wiring gap #3. Until
     both land, the honest framing is "detector response given a truth particle list".
4b. **Sub-threshold smearing at the energy floor — ACCEPTED, deferred 2026-09-07** (deliberate;
   this entry exists so it does not become an invisible assumption). The at-floor Bernoulli was
   DELETED on evidence (job 13557402): it pinned cells to a value that occurs **0 times** in
   13,054,567 held-out e± cells, and removing it left the gate flat (0.811 -> 0.817, one seed) while
   IMPROVING the per-cell energy fit (`cell_logE` W/sigma 0.028 -> **0.0244**, best on record).
   What remains is that a Gaussian mixture cannot represent the **x19 density step** at the 50 keV
   threshold, so it puts **1.03%** of cells below a hard cut where the data has **0.387%** — a
   0.65 pp excess, x2.7. The indicated fix (normalise the mixture on `[log_floor, inf)`) is NOT
   built. Judged not worth the compute against the joint/copula defect, which is ~0.70 and the
   bigger half of the gate.
   **Revisit if**: gap #1's cell projection lands (a generated cell below threshold is unphysical
   once cells are real objects and zero suppression is applied downstream), or `frac_near_floor`
   resurfaces as a leading gate discriminator — it is still 0.73-0.74 on the POOLED model, where
   this has never been measured.
5. **M3 event assembly** — superposition, noise process, mu conditioning. Not started.
6. **M4 reco-level (ACTS)** — deferred; also needs the `tracks` subset, not downloaded.

---

## 8. Maintaining this file

Update it in the same commit as any change to a data contract, a model output, or a metric
definition. When a docstring and the code disagree, fix the docstring and note it here — several
were stale as of 2026-08-24 (`build_calo_slice.py` documents `cond (S,4)` / `glob (S,2)`; the real
shapes are `cont (S,7)` / `glob (S,4)`).
