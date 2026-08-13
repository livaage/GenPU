"""CPU smoke + correctness test for the v3 surface-local tracker (tracker_module_ar).
Uses the real module_geometry.npz + surface_pion.npz slice. No GPU, no training."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from genpu.module_geometry import ModuleGeometry
from genpu.models.tracker_module_ar import TrackerModuleARModel

DATA = "/scratch/gpfs/IOJALVO/lv7805/genpu_data"


def main():
    torch.manual_seed(0)
    mg = ModuleGeometry(f"{DATA}/module_geometry.npz")
    hy = mg.build_hierarchy()
    print(f"modules={mg.n_modules}  max_surf={hy['max_surf']}  "
          f"n_surf/layer: min={hy['n_surf_per_layer'].min()} max={hy['n_surf_per_layer'].max()}")

    # --- hierarchy invariants ---
    v = hy["valid_mask"]; l2m = hy["local_to_module"]
    assert int(v.sum()) == mg.n_modules, "valid slots must equal module count"
    for m in range(0, mg.n_modules, 991):   # spot-check round-trip
        c, s = hy["layer_of_module"][m], hy["local_of_module"][m]
        assert l2m[c, s] == m and v[c, s]
    # every valid slot maps back to a module whose layer matches
    cc, ss = np.nonzero(v)
    assert (hy["layer_of_module"][l2m[cc, ss]] == cc).all()
    print("hierarchy round-trip OK")

    # --- load a batch from the real slice ---
    d = np.load(f"{DATA}/tracker_slice/surface_pion.npz")
    cont_all, hits, off = d["cont"], d["hits"], d["offsets"]
    cmean, cstd = d["cont_mean"], d["cont_std"]
    B, M = 16, 12
    sel = [i for i in range(len(off) - 1) if 1 <= off[i + 1] - off[i] <= M][:B]
    mod = torch.zeros(B, M, dtype=torch.long); contin = torch.zeros(B, M, 4)
    nh = torch.zeros(B, dtype=torch.long); msk = torch.zeros(B, M, dtype=torch.bool)
    condc = torch.zeros(B, 7)
    for bi, i in enumerate(sel):
        a, b = off[i], off[i + 1]; n = b - a
        blk = hits[a:b]
        mod[bi, :n] = torch.as_tensor(blk[:, 0].astype(np.int64))
        contin[bi, :n] = torch.as_tensor(blk[:, 1:5])
        nh[bi] = n; msk[bi, :n] = True
        condc[bi] = torch.as_tensor((cont_all[i] - cmean) / cstd)
    print(f"batch: B={B} max_hits={M} n_hits={nh.tolist()}")

    # cond embedding is external in the real pipeline; here fake a cond_dim=7 pass-through
    model = TrackerModuleARModel(hy, mg.means, mg.stds, cond_dim=7, model_dim=128,
                                 n_layers=2, n_heads=4, max_hits=M)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params/1e6:.2f}M")

    # --- forward + loss + backward ---
    out = model.forward(mod, contin, condc, msk)
    for k, t in out.items():
        assert torch.isfinite(t).all() or k == "surface_logits", k
    losses = model.loss(mod, contin, condc, msk, nh)
    print("losses:", {k: round(float(v), 3) for k, v in losses.items()})
    losses["tracker_ar_loss"].backward()
    g = sum(p.grad.abs().sum() for p in model.parameters() if p.grad is not None)
    assert torch.isfinite(g) and g > 0, "no gradient flow"
    # masked surface logits: invalid slots must be -inf-ish for the true layer
    tl = model._layer_of_module[mod]
    sl = out["surface_logits"]
    invalid = ~torch.as_tensor(v)[tl]
    assert (sl[invalid] < -1e8).all(), "invalid surface slots not masked"
    print("forward/loss/backward OK; surface masking OK")

    # --- generation invariants ---
    model.eval()
    phys, gmod = model.generate(condc, nh)
    print(f"generate: hits {tuple(phys.shape)} modules {tuple(gmod.shape)}")
    for bi in range(B):
        k = int(nh[bi])
        gm = gmod[bi, :k]
        assert (gm >= 0).all() and (gm < mg.n_modules).all(), "generated module out of range"
        # generated module's layer must be a real module (no invalid-slot fallback used)
        gl = model._layer_of_module[gm]
        back = torch.as_tensor(l2m)[gl, model._local_of_module[gm]]
        assert (back == gm).all(), "generated module not a clean (layer,slot) round-trip"
    # physical positions finite and within detector scale
    r = torch.hypot(phys[..., 0], phys[..., 1])
    print(f"generated |r| max {r.max():.0f}mm  |z| max {phys[...,2].abs().max():.0f}mm  (detector ~ r<1500, |z|<3000)")
    assert torch.isfinite(phys).all()
    print("generation invariants OK")
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
