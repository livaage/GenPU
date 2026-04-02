#!/usr/bin/env python
"""Export ColliderML HF cache to per-event Parquet files for offline use.

Run this on a node with internet/HF cache access. The resulting files can be
loaded directly with pyarrow on GPU nodes without network.

Output structure:
    /scratch/gpfs/IOJALVO/genpu_data/
        particles/event_{event_id}.parquet
        tracker_hits/event_{event_id}.parquet
        calo_hits/event_{event_id}.parquet
"""

import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from genpu.data import load_from_hf

OUTPUT_DIR = Path("/scratch/gpfs/IOJALVO/genpu_data")
SUBSETS = ["particles", "tracker_hits", "calo_hits"]

for name in SUBSETS:
    subset_dir = OUTPUT_DIR / name
    subset_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {name} from HF cache...")
    table = load_from_hf(name)
    n_rows = table.num_rows
    print(f"  {n_rows} events, {len(table.column_names)} columns")

    for i in tqdm(range(n_rows), desc=f"Exporting {name}"):
        event_id = table.column("event_id")[i].as_py()
        out_path = subset_dir / f"event_{event_id}.parquet"
        if out_path.exists():
            continue
        row = table.slice(i, 1)
        pq.write_table(row, out_path)

    n_files = len(list(subset_dir.glob("*.parquet")))
    print(f"  {n_files} files in {subset_dir}/")

print("\nDone.")
