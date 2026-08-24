# 2026-08-24 — incidence head WORKS: calibrated, and two Bernoullis reproduce the 3-way joint exactly

- **Commit / branch**: `0bcfac3` / `flow-response`
- **Jobs**: 12880027 (FAILED, NaN), 12880126 (result). Ckpt
  `checkpoints/incidence/incidence_s0.pt`. Train shards 0-2 (25.7M particles), held out shard 5
  (8.59M).
- Fills the hole named in [the visibility census](2026-08-24-subsystem-visibility-census.md):
  every slice filters to >=1 hit, so both subsystems modelled P(response | particle, n>=1) and
  neither could answer P(response | particle).

## Result

| | AUC | ECE | rate real | mean-pred | **sampled** |
|---|---|---|---|---|---|
| tracker | 0.9993 | 0.0005 | 0.3063 | 0.3062 | **0.3062** |
| calo | 0.9890 | 0.0008 | 0.5190 | 0.5191 | **0.5191** |

**Two Bernoullis on a shared trunk are SUFFICIENT** — the sampled 3-way joint matches to 4 decimals:

| outcome | real | sampled |
|---|---|---|
| trk-only | 0.2368 | 0.2369 |
| calo-only | 0.4496 | 0.4498 |
| **both** | **0.0694** | **0.0693** |
| neither | 0.2441 | 0.2440 |

No joint categorical needed; the shared trunk carries the dependence and the subsystems stay
architecturally separate. (Base rates are over ALL raw particles; dividing by the 0.7559 visible
fraction returns the census numbers 0.313 / 0.595 / 0.092 exactly.)

## Two things that went wrong first, both worth keeping

**(1) NaN for 20,000 steps, silently.** `perigee_d0`/`perigee_z0` are NON-FINITE for 41.9% of
particles — and those are **100% CHARGED** (56% of charged particles), the soft ones where the helix
perigee diverges; neutrals actually have it. Job 12880027 trained to NaN and still wrote a
checkpoint reporting ECE 0.0000 that samples "neither" for 100% of particles. The builder now
refuses to train on non-finite features and names the column.
**This also retracts an earlier claim of mine** that the source `perigee_d0` is "the real thing" and
a free upgrade over `build_count_slice_stage2.py`'s geometric `vx*sin(phi) - vy*cos(phi)`. The
geometric one is right precisely because it is defined for all particles. We use it, and carry the
source perigee only as (sanitised value, defined-flag).
Also: `|vz|` reaches 1.79e6 mm, so it enters as `log1p`, never raw.

**(2) AUC 0.9993 looked like leakage and is not.** Checked three ways:
- `d0_defined` single-feature AUC is 0.682, agreement with the label only 0.314 — not a proxy.
- `log_E` alone reaches ~0.9999 on tracker-born photons, which looked impossible for a conversion
  coin flip. Explanation: **corr(log E, |eta|) = 0.719** — energy is largely a proxy for
  FORWARDNESS, and acceptance is mostly geometric.
- The graph is not misaligned: P(tracker hit) over (|eta|, pT) is physical — central soft charged
  **0.997**, falling to **~0.28** at |eta| 3-5 regardless of pT.

**Framing correction**: earlier notes described the irreducible residual as "the ~5% conversion coin
flip". The high AUC does NOT mean the model predicts the coin flip — it means it RANKS well, mostly
on geometry. What shows it is doing the right thing is the calibration (ECE ~5e-4) plus the joint
reproduction, i.e. correct PROBABILITIES that sample to the right rates.

## Next

- Unblocks the honest full-event gate on BOTH subsystems, and is a prerequisite for the cell-level
  calo metric (it decides which particles to generate for).
- Not yet replicated (one seed). Cheap to add, and this project's standing rule is 2 seeds for any
  gate claim — though calibration and joint reproduction are far more stable statistics than a gate
  AUC, so the risk is lower here.
- Not yet wired into any generation path.
