"""Tracker MODULE geometry — surface-granular analogue of detector_geometry.py, for the
paper-style (arXiv:2512.24254) surface-local hit representation.

A module = (volume_id, layer_id, surface_id) composite. The generative model predicts a
module_index (categorical over ~18k modules) then per-module-standardized LOCAL Cartesian
residuals for (x, y, z, time): local = (physical - MODULE_MEANS[m]) / MODULE_STDS[m], which
is small & bounded because within-module spread is <=~50 mm (see scripts/tracker_surface_prebuild.py).

Data is produced by scripts/build_module_geometry.py -> module_geometry.npz. Path via
GENPU_MODULE_GEOMETRY, else <GENPU_DATA_DIR-ish scratch>/module_geometry.npz.
"""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np

_DEFAULT = "/scratch/gpfs/IOJALVO/lv7805/genpu_data/module_geometry.npz"


def module_composite(vol, layer, surface):
    """Collision-free int64 module key. MUST match build_module_geometry.module_composite."""
    return (np.asarray(vol).astype(np.int64) << 48) | \
           (np.asarray(layer).astype(np.int64) << 32) | np.asarray(surface).astype(np.int64)


class ModuleGeometry:
    """Loaded module vocabulary + per-module local frame."""

    def __init__(self, path: str | Path | None = None):
        path = Path(path or os.environ.get("GENPU_MODULE_GEOMETRY", _DEFAULT))
        if not path.exists():
            raise FileNotFoundError(
                f"module_geometry.npz not found at {path}. Build it first:\n"
                f"  python scripts/build_module_geometry.py --shards 0 1 2 3 4")
        d = np.load(path)
        self.composite = d["module_composite"].astype(np.int64)   # (M,) sorted
        self.key = d["module_key"].astype(np.int64)               # (M,3) [vol,layer,surface]
        self.means = d["module_mean"].astype(np.float32)          # (M,4) [x,y,z,time]
        self.stds = d["module_std"].astype(np.float32)            # (M,4)
        self.count = d["module_count"].astype(np.int64)           # (M,)
        self.layer_class = d["module_layer_class"].astype(np.int32)  # (M,)
        self.n_modules = len(self.composite)

    def to_index(self, vol, layer, surface):
        """(vol,layer,surface) arrays -> module_index (M,), or -1 if the module is unseen."""
        comp = module_composite(vol, layer, surface)
        pos = np.searchsorted(self.composite, comp)
        pos = np.clip(pos, 0, self.n_modules - 1)
        hit = self.composite[pos] == comp
        return np.where(hit, pos, -1).astype(np.int64)

    def local_residual(self, module_index, xyzt):
        """Physical (N,4) [x,y,z,time] -> per-module standardized local residual (N,4)."""
        m = self.means[module_index]
        s = self.stds[module_index]
        return (xyzt - m) / s

    def to_physical(self, module_index, resid):
        """Inverse of local_residual: standardized local residual -> physical (N,4)."""
        return resid * self.stds[module_index] + self.means[module_index]

    def build_hierarchy(self, n_layers: int = 48):
        """Factor the flat module vocab into (layer_class, surface-slot-within-layer) for the
        hierarchical head. Returns a dict of numpy arrays:
          layer_of_module   (M,)            int32  parent layer_class of each module
          local_of_module   (M,)            int32  module's slot 0..(n_surf[layer]-1) within its layer
          n_surf_per_layer  (n_layers,)     int32  # surfaces in each layer
          max_surf          int                    max over layers (surface-head width)
          local_to_module   (n_layers,max_surf) int64  (layer,slot)->global module index, -1 if invalid
          valid_mask        (n_layers,max_surf) bool   True where a slot maps to a real module
        Slots are assigned in ascending composite order within each layer (stable, reproducible)."""
        layer_of = self.layer_class.astype(np.int32)
        local_of = np.full(self.n_modules, -1, dtype=np.int32)
        n_surf = np.zeros(n_layers, dtype=np.int32)
        for c in range(n_layers):
            idx = np.nonzero(layer_of == c)[0]
            idx = idx[np.argsort(self.composite[idx], kind="stable")]
            local_of[idx] = np.arange(len(idx), dtype=np.int32)
            n_surf[c] = len(idx)
        max_surf = int(n_surf.max())
        local_to_module = np.full((n_layers, max_surf), -1, dtype=np.int64)
        valid_mask = np.zeros((n_layers, max_surf), dtype=bool)
        for m in range(self.n_modules):
            local_to_module[layer_of[m], local_of[m]] = m
            valid_mask[layer_of[m], local_of[m]] = True
        return {"layer_of_module": layer_of, "local_of_module": local_of,
                "n_surf_per_layer": n_surf, "max_surf": max_surf,
                "local_to_module": local_to_module, "valid_mask": valid_mask}
