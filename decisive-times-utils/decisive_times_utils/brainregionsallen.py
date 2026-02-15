"""
Export Allen brain regions (acronyms + names + hierarchy) to Parquet.

Creates a compact lookup table you can ship with your GUI for:
- autocomplete / validation
- a "Region Help" searchable table
- parent/child expansion logic

Requirements:
  pip install iblatlas pandas pyarrow

Run:
  python export_allen_regions_to_parquet.py --out allen_brain_regions.parquet
Optional:
  python export_allen_regions_to_parquet.py --out allen_brain_regions.parquet --with-desc-count
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

try:
    from iblatlas.regions import BrainRegions
except ImportError as e:
    print("ERROR: iblatlas is not installed.")
    print("Install with: pip install iblatlas")
    raise


def export_allen_regions(
    out_path: str | Path,
    compression: str = "snappy",
    include_mappings: bool = True,
    include_descendant_count: bool = False,
) -> pd.DataFrame:
    """
    Build a DataFrame of Allen brain regions and save it as Parquet.

    Columns (base):
      - acronym
      - name
      - region_id
      - depth (hierarchy level)

    Optional columns:
      - beryl_acronym, cosmos_acronym, swanson_acronym (if available in mappings)
      - n_descendants (how many descendants incl. itself)
    """
    br = BrainRegions()  # default brainmap = 'Allen'

    df = pd.DataFrame(
        {
            "acronym": pd.Series(br.acronym, dtype="string"),
            "name": pd.Series(br.name, dtype="string"),
            "region_id": pd.Series(br.id, dtype="Int64"),
            "depth": pd.Series(getattr(br, "level", [None] * len(br.id)), dtype="Int64"),
        }
    )

    # Optional: add mappings (Allen -> Beryl/Cosmos/Swanson) if available
    if include_mappings:
        # BrainRegions has a mappings dict; keys often include 'Beryl', 'Cosmos', 'Swanson'
        mappings = getattr(br, "mappings", {})
        for target in ["Beryl", "Cosmos", "Swanson"]:
            if isinstance(mappings, dict) and target in mappings:
                try:
                    mapped_ids = br.remap(br.id, target_map=target)
                    df[f"{target.lower()}_acronym"] = pd.Series(
                        br.id2acronym(mapped_ids), dtype="string"
                    )
                except Exception:
                    # If remap fails for this target, skip silently
                    pass

    # Optional: descendant counts (can take a little time, but still fine for ~1300 regions)
    if include_descendant_count:
        # Precompute descendant counts efficiently
        # descendants() returns ids; we count length for each region
        counts = []
        for rid in df["region_id"].astype("Int64"):
            if pd.isna(rid):
                counts.append(pd.NA)
                continue
            try:
                desc = br.descendants([int(rid)])
                counts.append(int(len(desc)))
            except Exception:
                counts.append(pd.NA)
        df["n_descendants"] = pd.Series(counts, dtype="Int64")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Save as parquet (pyarrow recommended)
    df.to_parquet(out_path, index=False, compression=compression)

    return df


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export Allen brain regions (acronyms/names/hierarchy) to a Parquet file."
    )
    parser.add_argument(
        "--out",
        type=str,
        default="allen_brain_regions.parquet",
        help="Output parquet path",
    )
    parser.add_argument(
        "--compression",
        type=str,
        default="snappy",
        choices=["snappy", "gzip", "zstd", "none"],
        help="Parquet compression codec",
    )
    parser.add_argument(
        "--no-mappings",
        action="store_true",
        help="Do not include Beryl/Cosmos/Swanson mapped acronyms",
    )
    parser.add_argument(
        "--with-desc-count",
        action="store_true",
        help="Include n_descendants column (may be slightly slower)",
    )
    args = parser.parse_args()

    compression = None if args.compression == "none" else args.compression

    df = export_allen_regions(
        out_path=args.out,
        compression=compression or "snappy",
        include_mappings=not args.no_mappings,
        include_descendant_count=args.with_desc_count,
    )

    print(f"Saved {len(df)} regions to: {args.out}")
    print("Columns:", ", ".join(df.columns))
    print("\nExample rows:")
    print(df.head(10).to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
