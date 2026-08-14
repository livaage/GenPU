# 2026-08-13 — Calo species census + energy-head audit (root cause of the pion 0.809)

- **Branch:** flow-response. Scripts added: `scripts/calo_species_census.py`,
  `scripts/calo_energy_diag.py`, `scripts/calo_incidence_probe.py`.

## 1. Species census — which particles the calo actually needs (held-out shard 5)
6.50M particles, **4.46M (68.7%) deposit >=1 calo cell**, 47.96M cells, 25.64 TeV… (25,643 GeV)
deposited. Share of deposited ENERGY / CELLS, by PDG class:

| class | energy | cells | deposit rate | median cells/shower |
|---|---|---|---|---|
| e- | 24.4% | 28.7% | 50.4% | 9 |
| e+ | 23.6% | 28.0% | 94.5% | 9 |
| pi- | 14.9% | 14.1% | 78.6% | 9 |
| pi+ | 13.7% | 13.3% | 78.1% | 8 |
| p | 13.2% | 6.9% | 84.7% | 3 |
| gamma | **2.3%** | **2.9%** | 69.4% | 4 |
| K+ K- n pbar K0L mu± nbar pi0 K0S other | ~8% | ~6% | 32-100% | 2-24 |

- **e± dominate the calorimeter: 48% of energy, 57% of cells — and were not modelled at all.**
  The photon, the one species with a good gate (0.557), is **2.3% of the energy**.
- Cumulative: **e± + pi± + p + gamma = 92% of energy, 94% of cells.** The ~8% tail still needs
  covering for a full-event generator, so nothing can be dropped — only pooled.
- Grouping: EM (e±,γ,π0) 50.4% E / 59.7% cells; charged hadrons 45.1% / 37.1%; neutral hadrons
  2.6% / 1.3%; muons 1.2% / 1.6%.
- e- deposits only 50% of the time vs e+ 94.5% — e- is full of soft tracker secondaries that never
  reach the calo (median production radius 541 mm for non-depositing e-), e+ come from conversions
  and pi0 decay. Species-dependent incidence, not a bug.

**Decision (user, 2026-08-13): ONE PDG-conditioned head**, not per-species checkpoints.
`ParticleConditioning` already embeds all 17 classes. Slice built:
`calo_slice/multispecies_v1.npz` — shards 0-2, `--max_per_class 400000` (new flag), 2.65M
showers, per-class counts e-/e+/pi+/pi-/p/gamma 400k each, then n 147k, K+ 86k, other 77k,
pi0 66k, K- 60k, mu+ 60k, K0L 47k, mu- 46k, pbar 35k, nbar 27k, K0S 1.2k.

## 2. Energy audit — WHY the pion gate is 0.809 (root cause, measured)
`scripts/calo_energy_diag.py` on the pion and photon slices + trained checkpoints:

- **The per-shower energy scale is not a function of the particle.** Per-shower mean cell log-E:
  **R² = 0.049 from the 7 CONT_FEATURES** (pion; photon 0.057) but **R² = 0.853 once
  (total_logE, log_n) are known** (photon 0.935). The old `EnergyHead` conditioned on the particle
  ONLY (a deliberate choice, documented in its docstring), so 28% of the cell-log-E variance is
  shower-level structure it emitted as INDEPENDENT per-cell noise → pion per-shower energy
  resolution 1.29-1.76 vs real 0.91-1.24.
- **Both mixture heads are unbounded above.** Pion gen max cell log-E **+0.28 vs real -1.76**
  (a 1.3 GeV cell where the hardest real cell is 0.17 GeV): 167 cells in 4.66M, enough to put
  ⟨E_reco/E_true⟩ at 4.7 vs 0.016 in the lowest energy bin. The photon's *GlobalHead total* is
  worse: sampled total reached **1.2e9 × E_true** (median and W/σ 0.0175 are fine — pure tail).
- **The sampled total was being thrown away.** `GlobalHead` generates `total_logE`, but generation
  built each shower's total as the SUM of independently drawn cells. As a total estimator the
  global is 4x better: **W/σ on log total 0.017 (global) vs 0.073 (sum-of-cells)**.
- Oracle check: rescaling generated cells to the *real* total reproduces real resolution exactly
  (1.238 → 1.238 in the lowest bin), confirming the total is the whole defect and positions are fine.

## 3. Missing head: does a particle deposit AT ALL?
Every calo slice/model/gate so far is conditioned on `n_cells >= 1` — i.e. P(shower | deposits),
never P(deposits | particle) — but only 68.7% of particles deposit and the rate is species-dependent
(e- 50%, e+ 95%, n 35%, K0L/pi0/nbar 100%). A full-event calo generator cannot pick which particles
shower. `calo_incidence_probe.py`: **logistic regression on the 7 CONT_FEATURES alone gets
AUC 0.897 / acc 0.905**, so a Bernoulli incidence head on the standard contract is easily learnable.
This is the calo analogue of the tracker's count head and is required for an honest full-event
calo gate.

## 4. Fix attempt 1 — glob-conditioned energy head + partition (2x2 ablation)
Code: `EnergyHead(glob_dim=2)` conditioned on the standardised (total_logE, log_n); sampling clamped
at the training `logE_max`; `CaloFlow.sample_showers()` (new single generation path) optionally
renormalises cells to the sampled total, pinning at-floor cells and reporting a
`scale_saturated` fraction; `--qt_total` extends the bounded quantile inverse to total_logE.
Metrics gained median/IQR response + `respmax` (the mean response was tail-dominated).

Photon, held-out shard 5: **A (baseline, sum-of-cells) gate 0.557 → C (fixed ckpt + partition)
0.809.** The tail was fixed (worst resp_max 240 → 11.6; lowest-bin resolution 33.7 → 2.91) and the
total improved (logEreco W/σ 0.027 → 0.019), but **cell_logE W/σ degraded 0.016 → 0.084** and that
dominates the gate (5 of its 8 features are cell-log-E statistics).

Cause of the regression: a hard rescale of n i.i.d. sampled cells injects a common log-shift of
scatter ~σ/√n_above_floor — worst exactly where n is small (photon ~4 cells/shower). Pion 2x2
(B = partition alone, D = conditioning alone) pending; **D (glob conditioning, no rescale) is the
variant expected to work**, since a cell that knows the total needs no post-hoc correction.
If it isn't enough, the next construction is sequential/stick-breaking energies (cell k takes a
fraction of the remaining budget, energy-ordered), which makes totals exact with no common-factor
noise.

## Reproduce
```
python scripts/calo_species_census.py --shard 5
python scripts/calo_energy_diag.py --slice .../calo_slice/pion_ctr_v2.npz \
    --ckpt .../checkpoints/calo_flow/pion_qtd_v2/checkpoint_040000.pt --tag pion
python scripts/calo_incidence_probe.py --shard 5 --n 150000
sbatch jobs/calo_energy_fix_pion.sh   # train + 2x2
sbatch jobs/build_calo_multispecies.sh
```
