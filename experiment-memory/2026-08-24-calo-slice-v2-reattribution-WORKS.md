# 2026-08-24 — calo slice v2: re-attribution moves the anchor from 73% fallback to 77% real face crossings

- **Commit / branch**: `6dfe8dd` / `flow-response`
- **Jobs**: 12884742 (semantics bug), 12887279 (timeout), 12891198 (timeout), 12893810 (result)
- **Artifacts**: `calo_slice/electron_v2_smoke.npz`, `electron_v2_noreattr_smoke.npz` (shard 0, e±)

## Result — the headline prediction confirmed

| anchor branch | control (v1 grouping) | **re-attributed** |
|---|---|---|
| barrel | 0.114 | 0.192 |
| endcap | 0.151 | 0.574 |
| **turning (fallback)** | **0.734** | **0.234** |

**Face crossings 0.265 -> 0.766.** The anchor is worth 15-24x on face branches vs ~2.5x on the
turning fallback, so it now does real work for three quarters of showers instead of a quarter. This
was the falsifiable test of whether the ancestor walk finds genuinely calo-incident particles.

| | control | re-attributed |
|---|---|---|
| showers (shard 0, e±) | 2,563,096 | 659,721 |
| cells/shower | 10.62 (med 9) | **19.82** (med 12) |
| dedup ratio | **1.000x** | 1.176x |
| layer defined | 0.831 | 0.824 |
| depth p50 / p95 | 51.2 / **1149.0** mm | 35.4 / **146.4** mm |

Two independent physics checks pass:
- **Control dedup is exactly 1.000x** — a single depositor's cells contain no duplicates, as they
  must not. Deduplication only bites when merging, confirming the merge is doing what it should.
- **Depth profiles separate correctly.** Re-attributed e± showers reach p95 = 146 mm; the control
  reaches 1149 mm, because control "e± showers" include e± FRAGMENTS deep inside hadronic showers
  while genuine e± incident particles make shallow EM showers. The re-attributed population is
  physically coherent in a way the control is not.

## A design error worth recording

v1 semantics select depositors BY species. But an e± fragment usually belongs to a photon or pion
shower, so "showers containing e± fragments" and "showers whose INCIDENT particle is e±" are
different sets — only the second is a meaningful fast-sim target. The first implementation selected
depositors by class then lifted, producing groups keyed by mixed-species ancestors while labelled
with the ancestor's pdg. Caught because the predicted shower ratio (~0.51) came out at 0.386.
**Writing the prediction down before running is what caught it.**

Consequence: **every v1 species slice has this character.** `electron_anchor` contains e± fragments,
most belonging to photon or pion showers, so "the e± calo head" has never modelled electron showers
in the sense the name implies.

## Caveats

- The two arms cover DIFFERENT cell sets (27.2M vs 15.4M): v1 selects cells deposited BY e±, v2
  selects cells from descendants of e± INCIDENT particles. Inherent to the correction, but totals
  are not comparable between arms.
- depth p5 = **-5.1 mm**, i.e. slightly before the front face: endcap layers start at 3202.4 mm but
  `load_front_face` returns 3212.5 (it took the p5 of the cell distribution). ~10 mm offset,
  harmless in the residual frame, worth correcting.
- One shard, e± only. Not yet built for other species or the held-out shard.

## Performance — three failures, all mine

1. Per-shower `np.unique(axis=0)` over ~1M showers: did not finish.
2. `core_anchor` called per shower on size-1 arrays (~660k numpy calls).
3. **The real killer: `np.load` on an `.npz` returns a LAZY NpzFile, and every `g["k"]` access
   decompresses the WHOLE array.** Three such accesses sat inside the per-shower loop, so the loop
   did not complete 100k of 660k showers in an hour. Materialising the arrays once took it to
   **23 seconds** — a ~150x speedup.

The only reason (3) was diagnosed is progress output added after the first timeout. **A long silent
loop is indistinguishable from a hung one**; instrument before submitting, not after.

## Next

- Fix the front-face constant (3202.4 for endcap).
- Build the production slices: shards 0/1/2 + held-out 5, e± and pion.
- `calo_flow.py` PointFlow 2D -> 3D positions to consume the depth column.
