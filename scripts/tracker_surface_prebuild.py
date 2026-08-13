"""Pre-build viability tests for a paper-style (arXiv:2512.24254) surface-local tracker
representation on ColliderML. Read-only, one source shard, bounded slice.

(A) p99-tail breakdown: within-module spatial spread, broken down by volume_id, to see which
    volumes cause the ~500mm p99 tail (candidates for a per-surface local frame).
(B) module-transition learnability: build the real inner->outer module sequence per particle and
    measure H(next module | current module). Low conditional entropy = geometry constrains the
    next surface = the AR can predict it. This is an UPPER bound on the AR's uncertainty (it also
    sees full history + particle kinematics), so if THIS is small the sequence is learnable.
"""
from __future__ import annotations
import glob
import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc

SHARD_GLOB = ("/scratch/gpfs/IOJALVO/lv7805/genpu_cache/datasets/"
              "CERN___collider_ml-release-1/pileup_only_pu0_tracker_hits/0.0.0/*/*.arrow")
N_EVENTS = 3000


def entropy_bits(counts):
    c = counts[counts > 0].astype(np.float64)
    p = c / c.sum()
    return float(-(p * np.log2(p)).sum())


def main():
    shard = sorted(glob.glob(SHARD_GLOB))[0]
    print("shard:", shard.split("/")[-1], "| events:", N_EVENTS)
    tbl = ipc.open_stream(pa.memory_map(shard, "r")).read_all()
    sub = tbl.slice(0, min(N_EVENTS, tbl.num_rows))

    def flat(c):
        col = sub.column(c).combine_chunks()
        return col.flatten().to_numpy(zero_copy_only=False)

    x, y, z = flat("x").astype(np.float64), flat("y").astype(np.float64), flat("z").astype(np.float64)
    r = np.hypot(x, y)
    vol = flat("volume_id").astype(np.int64)
    lay = flat("layer_id").astype(np.int64)
    surf = flat("surface_id").astype(np.int64)
    pid = flat("particle_id").astype(np.int64)
    # per-event lengths -> event index per hit (for grouping tracks within an event)
    lengths = np.array([len(v) for v in sub.column("x").to_pylist()], dtype=np.int64)
    ev_idx = np.repeat(np.arange(len(lengths)), lengths)
    n = len(x)

    # dense global module id over the (volume,layer,surface) composite
    key = np.stack([vol, lay, surf], axis=1)
    _, module = np.unique(key, axis=0, return_inverse=True)
    n_modules = module.max() + 1
    print(f"hits={n:,}  modules={n_modules:,}  (uniform H = {np.log2(n_modules):.1f} bits)")

    # ---------- (A) within-module spread, broken down by volume ----------
    print("\n" + "=" * 74)
    print("(A) WITHIN-MODULE SPATIAL SPREAD by volume_id  (groups >=20 hits)")
    print(f"  {'vol':>3s} {'#mods':>7s} {'hits':>9s} | {'std_r med':>9s} {'p90':>7s} {'p99':>7s} "
          f"| {'inplane med':>11s} {'p90':>7s} {'p99':>7s}")
    order = np.lexsort((r, module))
    ms = module[order]; xs, ys, zs, rs, vs = x[order], y[order], z[order], r[order], vol[order]
    bnd = np.where(np.diff(ms) != 0)[0] + 1
    starts = np.concatenate([[0], bnd]); ends = np.concatenate([bnd, [len(ms)]])
    m_vol, m_sr, m_ip, m_sz, m_n = [], [], [], [], []
    for s, e in zip(starts, ends):
        if e - s < 20:
            continue
        sx, sy = xs[s:e].std(), ys[s:e].std()
        m_vol.append(vs[s]); m_sr.append(rs[s:e].std()); m_sz.append(zs[s:e].std())
        m_ip.append(np.hypot(sx, sy)); m_n.append(e - s)
    m_vol = np.array(m_vol); m_sr = np.array(m_sr); m_ip = np.array(m_ip)
    m_sz = np.array(m_sz); m_n = np.array(m_n)
    for v in np.unique(m_vol):
        k = m_vol == v
        sr, ip = m_sr[k], m_ip[k]
        pr = np.percentile(sr, [50, 90, 99]); pi = np.percentile(ip, [50, 90, 99])
        print(f"  {v:>3d} {k.sum():>7d} {int(m_n[k].sum()):>9d} | {pr[0]:>9.2f} {pr[1]:>7.2f} {pr[2]:>7.2f} "
              f"| {pi[0]:>11.2f} {pi[1]:>7.2f} {pi[2]:>7.2f}")
    # what fraction of wide modules (in-plane std >100mm) is in which volume?
    wide = m_ip > 100
    print(f"\n  wide modules (in-plane std >100mm): {wide.sum()}/{len(m_ip)} = {wide.mean()*100:.1f}%")
    if wide.any():
        for v in np.unique(m_vol[wide]):
            print(f"    vol {v}: {(m_vol[wide]==v).sum()} wide modules "
                  f"({(m_vol[wide]==v).sum()/wide.sum()*100:.0f}% of wide)")
    print("  -> volumes with small spread can use raw (x-x̄, z-z̄) local coords; wide ones need a fitted in-plane frame.")

    # ---------- (B) module-transition learnability ----------
    print("\n" + "=" * 74)
    print("(B) MODULE-TRANSITION ENTROPY (real inner->outer sequences)")
    # sort by (event, particle, r) so consecutive same-(event,particle) rows are inner->outer
    o2 = np.lexsort((r, pid, ev_idx))
    e2, p2, m2 = ev_idx[o2], pid[o2], module[o2]
    same = (e2[:-1] == e2[1:]) & (p2[:-1] == p2[1:])   # adjacent pair in the same track
    cur, nxt = m2[:-1][same], m2[1:][same]
    n_trans = len(cur)
    n_tracks = len(np.unique(np.stack([e2, p2], axis=1), axis=0))
    print(f"  tracks={n_tracks:,}  transitions={n_trans:,}  (mean {n_trans/max(n_tracks,1):.2f} steps/track)")
    self_loop = (cur == nxt).mean()
    print(f"  self-transitions (next module == current, i.e. 2 hits same surface): {self_loop*100:.1f}%")

    # marginal H(next)
    nxt_counts = np.bincount(nxt, minlength=n_modules)
    H_next = entropy_bits(nxt_counts)
    # conditional H(next | current): weighted avg over current modules
    pair = cur.astype(np.int64) * n_modules + nxt.astype(np.int64)
    upair, pcnt = np.unique(pair, return_counts=True)
    ucur = upair // n_modules
    cur_tot = np.bincount(cur, minlength=n_modules).astype(np.float64)
    # per-current entropy and top-1 successor prob
    H_cond = 0.0; top1_acc = 0.0
    # group pair counts by current
    order_c = np.argsort(ucur, kind="stable")
    ucur_s, pcnt_s = ucur[order_c], pcnt[order_c]
    b = np.where(np.diff(ucur_s) != 0)[0] + 1
    st = np.concatenate([[0], b]); en = np.concatenate([b, [len(ucur_s)]])
    total = cur_tot.sum()
    for s, e in zip(st, en):
        c = pcnt_s[s:e].astype(np.float64)
        tot = c.sum()
        p = c / tot
        H_cond += (tot / total) * (-(p * np.log2(p)).sum())
        top1_acc += c.max() / total   # best-guess accuracy if you always pick top successor
    print(f"  H(next)            = {H_next:6.2f} bits   (eff. {2**H_next:8.1f} modules)")
    print(f"  H(next | current)  = {H_cond:6.2f} bits   (eff. {2**H_cond:8.1f} modules)")
    print(f"  info gain from knowing current module: {H_next - H_cond:.2f} bits")
    print(f"  top-1 next-module accuracy (greedy, given current only): {top1_acc*100:.1f}%")
    print(f"  distinct successors per current module (mean): "
          f"{len(upair)/max((cur_tot>0).sum(),1):.1f}")
    print("\n  -> H(next|current) is an UPPER bound on the AR's uncertainty (it also sees history +")
    print("     kinematics). Small eff-modules / high top-1 => the surface sequence is learnable.")


if __name__ == "__main__":
    main()
