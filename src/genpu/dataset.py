"""PyTorch Dataset classes for GenPU training.

Reads preprocessed .npz shards (flat arrays + offsets format).
Padding to fixed sizes happens here at __getitem__ / collate time.

Stage 2 (ParticleHitDataset): one sample = one visible particle + its hits.
Stage 1 (EventParticleSetDataset): one sample = one event's visible particle set.
"""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class ParticleHitDataset(Dataset):
    """Stage 2 dataset: per-particle samples with tracker + calo hits.

    Each sample is:
        particle_features: (6,) float32 — log_pt, eta, phi, pdg_class, charge, mass
        tracker_hits: (n_trk, 6) float32 — r, phi, z, time, volume_id, layer_id
        calo_hits: (n_cal, 5) float32 — eta, phi, log_energy, contrib_frac, detector
        n_tracker: int
        n_calo: int

    Collation pads hits to the max in the batch (see collate_fn).
    """

    def __init__(
        self,
        data_dir: str | Path,
        shard_indices: Optional[list[int]] = None,
        max_tracker_hits: int = 32,
        max_calo_hits: int = 64,
    ):
        """
        Parameters
        ----------
        data_dir : path to preprocessed directory
        shard_indices : which shards to include (default: all found)
        max_tracker_hits : clip hits beyond this count
        max_calo_hits : clip hits beyond this count
        """
        self.max_tracker_hits = max_tracker_hits
        self.max_calo_hits = max_calo_hits

        data_dir = Path(data_dir)
        if shard_indices is not None:
            paths = [data_dir / f"shard_{i:04d}_stage2.npz" for i in shard_indices]
            paths = [p for p in paths if p.exists()]
        else:
            paths = sorted(data_dir.glob("shard_*_stage2.npz"))

        if not paths:
            raise FileNotFoundError(f"No stage2 shards found in {data_dir}")

        # Load all shards into memory (flat arrays are compact)
        self.particle_features = []
        self.tracker_hits_flat = []
        self.tracker_offsets = []
        self.calo_hits_flat = []
        self.calo_offsets = []

        cumulative_trk = 0
        cumulative_cal = 0
        cumulative_particles = 0

        for path in paths:
            d = np.load(path)
            n = len(d["particle_features"])
            self.particle_features.append(d["particle_features"])

            # Shift offsets by cumulative hit counts
            trk_off = d["tracker_offsets"].astype(np.int64) + cumulative_trk
            cal_off = d["calo_offsets"].astype(np.int64) + cumulative_cal
            self.tracker_offsets.append(trk_off[:-1])  # drop last (it's the total)
            self.calo_offsets.append(cal_off[:-1])

            self.tracker_hits_flat.append(d["tracker_hits_flat"])
            self.calo_hits_flat.append(d["calo_hits_flat"])

            cumulative_trk += len(d["tracker_hits_flat"])
            cumulative_cal += len(d["calo_hits_flat"])
            cumulative_particles += n

        self.particle_features = np.concatenate(self.particle_features)
        self.tracker_hits_flat = np.concatenate(self.tracker_hits_flat) if cumulative_trk > 0 else np.empty((0, 6), dtype=np.float32)
        self.calo_hits_flat = np.concatenate(self.calo_hits_flat) if cumulative_cal > 0 else np.empty((0, 5), dtype=np.float32)

        # Build global offset arrays (one entry per particle)
        trk_offsets = np.concatenate(self.tracker_offsets)
        cal_offsets = np.concatenate(self.calo_offsets)
        # Append sentinel for the last particle
        self._trk_offsets = np.append(trk_offsets, cumulative_trk).astype(np.int64)
        self._cal_offsets = np.append(cal_offsets, cumulative_cal).astype(np.int64)

        self._len = len(self.particle_features)

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        pf = torch.from_numpy(self.particle_features[idx].copy())

        # Tracker hits for this particle
        trk_lo = self._trk_offsets[idx]
        trk_hi = self._trk_offsets[idx + 1]
        n_trk = min(int(trk_hi - trk_lo), self.max_tracker_hits)
        trk = torch.from_numpy(self.tracker_hits_flat[trk_lo:trk_lo + n_trk].copy())

        # Calo hits for this particle
        cal_lo = self._cal_offsets[idx]
        cal_hi = self._cal_offsets[idx + 1]
        n_cal = min(int(cal_hi - cal_lo), self.max_calo_hits)
        cal = torch.from_numpy(self.calo_hits_flat[cal_lo:cal_lo + n_cal].copy())

        return {
            "particle_features": pf,
            "tracker_hits": trk,
            "calo_hits": cal,
            "n_tracker": n_trk,
            "n_calo": n_cal,
        }

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
        """Pad hits to max in batch, produce masks."""
        bsz = len(batch)
        pf = torch.stack([b["particle_features"] for b in batch])
        n_trk = [b["n_tracker"] for b in batch]
        n_cal = [b["n_calo"] for b in batch]

        max_trk = max(n_trk) if max(n_trk) > 0 else 1
        max_cal = max(n_cal) if max(n_cal) > 0 else 1

        trk_padded = torch.zeros(bsz, max_trk, 6)
        trk_mask = torch.zeros(bsz, max_trk, dtype=torch.bool)
        cal_padded = torch.zeros(bsz, max_cal, 5)
        cal_mask = torch.zeros(bsz, max_cal, dtype=torch.bool)

        for i, b in enumerate(batch):
            nt = b["n_tracker"]
            if nt > 0:
                trk_padded[i, :nt] = b["tracker_hits"]
                trk_mask[i, :nt] = True
            nc = b["n_calo"]
            if nc > 0:
                cal_padded[i, :nc] = b["calo_hits"]
                cal_mask[i, :nc] = True

        return {
            "particle_features": pf,
            "tracker_hits": trk_padded,
            "tracker_mask": trk_mask,
            "calo_hits": cal_padded,
            "calo_mask": cal_mask,
            "n_tracker": torch.tensor(n_trk, dtype=torch.long),
            "n_calo": torch.tensor(n_cal, dtype=torch.long),
        }


class EventParticleSetDataset(Dataset):
    """Stage 1 dataset: per-event samples of visible particle sets.

    Each sample is a variable-size set of particle feature vectors.
    Collation pads to the largest set in the batch.
    """

    def __init__(
        self,
        data_dir: str | Path,
        shard_indices: Optional[list[int]] = None,
        max_particles: int = 2048,
    ):
        self.max_particles = max_particles
        data_dir = Path(data_dir)

        if shard_indices is not None:
            paths = [data_dir / f"shard_{i:04d}_stage1.npz" for i in shard_indices]
            paths = [p for p in paths if p.exists()]
        else:
            paths = sorted(data_dir.glob("shard_*_stage1.npz"))

        if not paths:
            raise FileNotFoundError(f"No stage1 shards found in {data_dir}")

        self.particle_features = []
        self.offsets = []
        cumulative = 0

        for path in paths:
            d = np.load(path)
            self.particle_features.append(d["particle_features"])
            off = d["offsets"].astype(np.int64) + cumulative
            # Each event is offsets[i]:offsets[i+1]
            # Store start for each event; we'll append the total at the end
            n_events = len(d["event_ids"])
            self.offsets.append(off[:n_events])
            cumulative += len(d["particle_features"])

        self.particle_features = np.concatenate(self.particle_features)
        self.offsets = np.concatenate(self.offsets)
        # Sentinel
        self._offsets = np.append(self.offsets, cumulative).astype(np.int64)
        self._len = len(self._offsets) - 1

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        lo = self._offsets[idx]
        hi = self._offsets[idx + 1]
        n = min(int(hi - lo), self.max_particles)
        features = torch.from_numpy(self.particle_features[lo:lo + n].copy())
        return {
            "particle_set": features,
            "n_particles": n,
        }

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
        """Pad particle sets to max in batch."""
        bsz = len(batch)
        n_parts = [b["n_particles"] for b in batch]
        max_n = max(n_parts)
        d = batch[0]["particle_set"].shape[-1]

        padded = torch.zeros(bsz, max_n, d)
        mask = torch.zeros(bsz, max_n, dtype=torch.bool)

        for i, b in enumerate(batch):
            n = b["n_particles"]
            padded[i, :n] = b["particle_set"]
            mask[i, :n] = True

        return {
            "particle_set": padded,
            "mask": mask,
            "n_particles": torch.tensor(n_parts, dtype=torch.long),
        }
