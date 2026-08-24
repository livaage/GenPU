"""Sidecar the calo LONGITUDINAL coordinate onto existing stage2 shards.

preprocessing.py collapses calo cell (x,y,z) -> (eta,phi) and keeps only `detector` as a depth
proxy, so the calo model is 2D and the plan's acceptance metrics (layer-wise energy fractions,
longitudinal profiles) were never computable — see PIPELINE.md §2 and §7.

WHY A SIDECAR, NOT A RE-PREPROCESS. The existing calo data is INCOMPLETE, not WRONG: every column
it stores is correct. Regenerating the shards would risk silently changing the data every logged
result was measured on, for no gain. Same reasoning as shard_XXXX_pids.npz. We recompute (r, z) via
the verified code path and ASSERT the recomputed (eta, phi, logE, frac, detector) match the existing
`calo_hits_flat` row-for-row before writing anything.

WHAT IS STORED: raw per-contribution `(r, z)`, NOT a derived depth. Reducing coordinates at the
input boundary is what caused this problem; downstream picks the convention. For reference, the
measured structure (job 12881108) is: barrel dets {10,13} depth = r - 1259.2, endcap {9,11,12,14}
depth = |z| - 3212.5; endcaps carry 48 (EM) / 36 (hadronic) discrete layers while the barrel is
effectively continuous (2,526 distinct depths at 0.08 mm).

Output: `shard_XXXX_calo_rz.npz` with `calo_rz` (M, 2) aligned row-for-row with `calo_hits_flat`.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from genpu.data import load_shard, build_event_index, explode_list_columns  # noqa: E402
from genpu.preprocessing import (process_event_vectorized, _empty_tracker,  # noqa: E402
                                 _empty_calo)

CH_ETA, CH_PHI, CH_LOGE, CH_FRAC, CH_DET = range(5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--preproc_dir", default="/scratch/gpfs/IOJALVO/lv7805/genpu_data/preprocessed")
    ap.add_argument("--out_dir", default="/scratch/gpfs/IOJALVO/lv7805/genpu_data/calo_depth")
    ap.add_argument("--max_events", type=int, default=0, help="smoke test; output tagged _smoke")
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)

    print(f"[shard {a.shard}] loading raw tables", flush=True)
    t, tr, ca = (load_shard("particles", a.shard), load_shard("tracker_hits", a.shard),
                 load_shard("calo_hits", a.shard))
    pi, ti, ci = build_event_index(t), build_event_index(tr), build_event_index(ca)

    s2 = np.load(Path(a.preproc_dir) / f"shard_{a.shard:04d}_stage2.npz")
    ch, coff, ev_ex = s2["calo_hits_flat"], s2["calo_offsets"], s2["event_ids"]
    rows_of = {int(e): np.where(ev_ex == e)[0] for e in np.unique(ev_ex)}
    print(f"[shard {a.shard}] stage2 {ch.shape[0]:,} calo contributions / {len(rows_of):,} events",
          flush=True)

    rz = np.full((ch.shape[0], 2), np.nan, np.float32)
    ev_list = sorted(pi)[:a.max_events] if a.max_events else sorted(pi)
    n_ok = 0
    for k, eid in enumerate(ev_list):
        if eid not in rows_of:
            continue
        res = process_event_vectorized(
            explode_list_columns(t, pi[eid]),
            explode_list_columns(tr, ti[eid]) if eid in ti else _empty_tracker(),
            explode_list_columns(ca, ci[eid]) if eid in ci else _empty_calo())
        vis = np.where(res["visible_mask"])[0]
        rows = rows_of[eid]
        if len(vis) != len(rows):
            raise SystemExit(f"event {eid}: {len(vis)} visible vs {len(rows)} stage2 rows")
        for p_local, p_row in zip(vis, rows):
            lo, hi = int(coff[p_row]), int(coff[p_row + 1])
            mine = res["calo_hits"][p_local]
            if hi - lo != len(mine):
                raise SystemExit(f"event {eid} particle row {p_row}: {hi-lo} stored vs {len(mine)}")
            if len(mine):
                # MATCH BY VALUE, NOT ROW INDEX. preprocessing used a non-stable np.argsort, so the
                # within-particle order of the stored rows is arbitrary and not reproducible — the
                # recomputed block is the same SET in a different order (verified job 12881313).
                # lexsort both on (eta, phi, logE) and pair them up; cells are effectively unique in
                # (eta, phi), so this is a bijection.
                st, mi = ch[lo:hi], mine[:, [CH_ETA, CH_PHI, CH_LOGE, CH_FRAC, CH_DET]]
                o_st = np.lexsort((st[:, CH_LOGE], st[:, CH_PHI], st[:, CH_ETA]))
                o_mi = np.lexsort((mi[:, CH_LOGE], mi[:, CH_PHI], mi[:, CH_ETA]))
                if not np.allclose(mi[o_mi], st[o_st], atol=1e-4, equal_nan=True):
                    raise SystemExit(f"event {eid} particle row {p_row}: calo hits differ as a SET "
                                     f"— refusing to write a misaligned sidecar")
                blk = np.empty((hi - lo, 2), np.float32)
                blk[o_st] = res["calo_rz"][p_local][o_mi]
                rz[lo:hi] = blk
        n_ok += 1
        if k % 1000 == 0:
            print(f"  event {k}/{len(ev_list)}", flush=True)

    if not a.max_events:
        miss = np.isnan(rz[:, 0]).mean()
        if miss > 0:
            raise SystemExit(f"{miss:.4f} of rows never filled — incomplete, refusing to write")
    sfx = "_smoke" if a.max_events else ""
    np.savez_compressed(out / f"shard_{a.shard:04d}_calo_rz{sfx}.npz", calo_rz=rz)
    r, z = rz[:, 0], rz[:, 1]
    ok = np.isfinite(r)
    print(f"\n[shard {a.shard}] events validated {n_ok:,}   rows filled {ok.mean():.4f}")
    print(f"  r  p5/50/95: {np.percentile(r[ok],5):.1f} / {np.percentile(r[ok],50):.1f} / "
          f"{np.percentile(r[ok],95):.1f} mm")
    print(f"  |z| p5/50/95: {np.percentile(np.abs(z[ok]),5):.1f} / "
          f"{np.percentile(np.abs(z[ok]),50):.1f} / {np.percentile(np.abs(z[ok]),95):.1f} mm")
    print(f"wrote {out}/shard_{a.shard:04d}_calo_rz{sfx}.npz")


if __name__ == "__main__":
    main()
