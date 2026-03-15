"""
Split 900 cases (450 ctrate + 450 radgenome) into train/val/test (60/20/20).
Stratified by dataset, deterministic seed.

Output:
  manifests/split_seed42/
    train.txt   (540 lines: 270 ctrate + 270 radgenome)
    val.txt     (180 lines: 90 ctrate + 90 radgenome)
    test.txt    (180 lines: 90 ctrate + 90 radgenome)

Each line: dataset/case_id  (e.g. ctrate/train_10123_a_1)

Usage:
    python Scripts/split_train_val_test.py \
        --cases_dir outputs/stage0_5_llama_450/cases \
        --out_dir manifests/split_seed42 \
        --seed 42
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List, Tuple


def stratified_split(
    items: List[str],
    ratios: Tuple[float, float, float],
    seed: int,
) -> Tuple[List[str], List[str], List[str]]:
    """Split items into train/val/test by ratios, deterministic."""
    rng = random.Random(seed)
    shuffled = list(items)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])

    train = shuffled[:n_train]
    val = shuffled[n_train : n_train + n_val]
    test = shuffled[n_train + n_val :]
    return train, val, test


def main() -> None:
    parser = argparse.ArgumentParser(description="Split cases into train/val/test.")
    parser.add_argument("--cases_dir", type=str, default="outputs/stage0_5_llama_450/cases")
    parser.add_argument("--out_dir", type=str, default="manifests/split_seed42")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--ratios",
        type=float,
        nargs=3,
        default=[0.6, 0.2, 0.2],
        help="train/val/test ratios (must sum to 1.0)",
    )
    args = parser.parse_args()

    cases_root = Path(args.cases_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ratios = tuple(args.ratios)
    assert abs(sum(ratios) - 1.0) < 1e-6, f"Ratios must sum to 1.0, got {sum(ratios)}"

    # Collect cases per dataset
    datasets = sorted([d.name for d in cases_root.iterdir() if d.is_dir()])
    print(f"Datasets: {datasets}")

    all_train: List[str] = []
    all_val: List[str] = []
    all_test: List[str] = []

    for ds in datasets:
        ds_dir = cases_root / ds
        case_ids = sorted([c.name for c in ds_dir.iterdir() if c.is_dir()])
        print(f"  {ds}: {len(case_ids)} cases")

        train, val, test = stratified_split(case_ids, ratios, args.seed)
        print(f"    train={len(train)}, val={len(val)}, test={len(test)}")

        all_train.extend(f"{ds}/{cid}" for cid in train)
        all_val.extend(f"{ds}/{cid}" for cid in val)
        all_test.extend(f"{ds}/{cid}" for cid in test)

    # Sort for reproducibility
    all_train.sort()
    all_val.sort()
    all_test.sort()

    for split_name, split_list in [("train", all_train), ("val", all_val), ("test", all_test)]:
        out_file = out_dir / f"{split_name}.txt"
        with out_file.open("w") as f:
            for item in split_list:
                f.write(item + "\n")
        print(f"Wrote {out_file} ({len(split_list)} cases)")

    print(f"\nSummary: train={len(all_train)}, val={len(all_val)}, test={len(all_test)}")


if __name__ == "__main__":
    main()
