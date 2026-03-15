"""
Filter a full manifest CSV to only include cases from a split file.

Usage:
    python Scripts/filter_manifest_by_split.py \
        --manifest manifests/ctrate_900_manifest.csv \
        --split_file manifests/split_seed42/test.txt \
        --dataset ctrate \
        --out manifests/ctrate_test.csv
"""
from __future__ import annotations

import argparse
import pandas as pd
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--split_file", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True, help="Dataset prefix in split file (ctrate/radgenome)")
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    # Read split file, filter to this dataset
    with open(args.split_file) as f:
        lines = [l.strip() for l in f if l.strip()]
    case_ids = [l.split("/", 1)[1] for l in lines if l.startswith(args.dataset + "/")]
    print(f"Split file: {len(case_ids)} cases for dataset '{args.dataset}'")

    df = pd.read_csv(args.manifest)
    filtered = df[df["case_id"].isin(case_ids)].reset_index(drop=True)
    print(f"Filtered: {len(filtered)} / {len(df)} rows")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(args.out, index=False)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
