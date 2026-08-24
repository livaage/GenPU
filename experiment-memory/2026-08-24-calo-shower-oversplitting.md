# 2026-08-24 — 65% of calo "showers" are FRAGMENTS born inside the calorimeter; candidate mechanism for the standing e± problem

- **Commit / branch**: `99879a3` / `flow-response` (working tree)
- **Job**: none — login-node probe, offsets + `particle_aux` only.
- **Measurement only. No retrain. The hypothesis below is UNTESTED.**
- Follows the [subsystem visibility census](2026-08-24-subsystem-visibility-census.md) and
  [scoping_secondary_cascade.md](../scoping_secondary_cascade.md), whose Option C already called
  the calo re-attribution a "latent simplification ... not urgent". This says it may not be latent.

## Hypothesis

`scoping_secondary_cascade.md` notes in passing that the current calo training "over-splits the
shower into per-secondary deposits". `build_calo_slice.py` groups by `calo_offsets`, i.e. **per
direct depositing particle**. If a large share of those depositors were created *inside* the
calorimeter by an earlier particle's shower, then the model is being trained on shower FRAGMENTS
presented as showers, with "incident kinematics" that are really mid-shower kinematics.

## Result

Shard 0. A depositor is a shower fragment if its production vertex is past the calo front face
(`calo_geometry.json`: barrel r > 1259.20 mm, endcap |z| > 3212.5 mm).

- **64.8% of calo-depositing particles were born INSIDE the calorimeter**, carrying **54.2% of all
  calo cells**. Cells/shower: born-outside 13.98, born-inside 8.99.

| species | showers | **% born inside** | **% of its cells** | current gate (8-feat) |
|---|---|---|---|---|
| **γ** | 315,436 | **0.1%** | 0.0% | **0.557** (best on record) |
| π+ / π− | 408k / 426k | 55.5% / 56.6% | 34.8% / 37.2% | 0.809 |
| **e− / e+** | 1.31M / 1.25M | **74.2% / 74.3%** | **67.8% / 67.8%** | 0.888 ms / 0.773 dedicated |
| p | 517,228 | 87.3% | 70.8% | 0.749 |
| n | 49,203 | 7.2% | 2.5% | — |
| K0L / π0 | 15k / 22k | 0.1% / 0.0% | 0.0% | — |

**The species ordering of calo gate difficulty tracks the over-splitting fraction.** Photon is the
one species essentially never split (0.1%) and has by far the best gate; e± are the most split and
the most stubborn. This supplies a MECHANISM for a line STATUS currently attributes to intrinsic
difficulty: *"e± are intrinsically harder than photons (9 cells/shower vs 4, **wide production-radius
spread**)"* — the wide production-radius spread IS this, e± born at varying depths inside the calo.

**Counter-evidence, stated plainly**: the proton breaks monotonicity (87.3% split, yet gate 0.749,
better than e±). It deposits a median 3 cells, so it is plausibly a different regime, but the
correlation is not clean and this is not proof.

## Verdict — measurement KEPT, hypothesis UNTESTED

No retrain was run. What is established: the over-splitting is real and large (65% of showers,
54% of cells), and it is species-ordered in a way that matches the calo difficulty ordering. What
is NOT established: that fixing it moves any gate.

## Next — the test is cheap and is on the Option C path anyway

1. **BLOCKED ON**: stage2 does not store the particle's own id, so the parent chain cannot be walked
   downstream (see the census entry). Storing `particle_ids` in `preprocessing.py` and re-running
   preprocessing is the prerequisite.
2. Re-attribute each calo cell to its **calo-incident ancestor** (walk `parent_id` up until the
   production vertex is outside the front face), rebuild the e± slice, retrain, gate.
3. **Two seeds minimum** — the 2026-08-24 replicate established e± gate8 seed spread ~0.11, which is
   larger than most deltas this project has acted on.
4. This is half of Option C (calo-inclusive) and would be needed for Pythia-input generation
   regardless, so it is not a detour even if the gate does not move.
5. If it DOES move the gate, every calo result since 2026-08-13 was measured through a confounded
   training target, and the frac_near_floor / n=1 thread should be re-read in that light.
