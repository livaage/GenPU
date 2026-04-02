#!/usr/bin/env python
"""Preprocess a single Arrow shard into training-ready .npz files.

Usage:
    python preprocess_shard.py SHARD_IDX [--output-dir DIR]

Called by the SLURM array job (submit_preprocess.slurm) or directly for testing.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from genpu.preprocessing import process_shard


def main():
    parser = argparse.ArgumentParser(description="Preprocess one Arrow shard")
    parser.add_argument("shard_idx", type=int, help="Shard index (0-999)")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/scratch/gpfs/IOJALVO/genpu_data/preprocessed",
        help="Output directory for .npz files",
    )
    args = parser.parse_args()

    print(f"Processing shard {args.shard_idx}...")
    t0 = time.time()

    summary = process_shard(
        shard_idx=args.shard_idx,
        output_dir=args.output_dir,
    )

    elapsed = time.time() - t0
    summary["elapsed_seconds"] = round(elapsed, 1)
    print(f"Done in {elapsed:.1f}s")
    print(json.dumps(summary, indent=2))

    # Save summary alongside the data
    summary_path = Path(args.output_dir) / f"shard_{args.shard_idx:04d}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
