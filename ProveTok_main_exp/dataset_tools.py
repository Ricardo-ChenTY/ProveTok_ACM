from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass
class MiniSubsetSpec:
    input_csv: str
    output_csv: str
    n_samples: int = 450
    strata_cols: Sequence[str] = ()
    seed: int = 42


def _stratified_sample(df: pd.DataFrame, n: int, strata_cols: Sequence[str], seed: int) -> pd.DataFrame:
    if n >= len(df):
        return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    if not strata_cols:
        return df.sample(n=n, random_state=seed).reset_index(drop=True)

    grouped = df.groupby(list(strata_cols), dropna=False, sort=False)
    total = len(df)
    frames: List[pd.DataFrame] = []
    allocated = 0
    for _, g in grouped:
        frac = len(g) / total
        k = int(np.floor(frac * n))
        k = min(k, len(g))
        if k > 0:
            frames.append(g.sample(n=k, random_state=seed))
            allocated += k
    remainder = n - allocated
    if remainder > 0:
        used_idx = pd.Index([])
        if frames:
            used_idx = pd.concat(frames).index
        rest = df.drop(index=used_idx, errors="ignore")
        if len(rest) < remainder:
            remainder = len(rest)
        if remainder > 0:
            frames.append(rest.sample(n=remainder, random_state=seed))

    out = pd.concat(frames).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out.head(n).reset_index(drop=True)


def build_mini_subset(spec: MiniSubsetSpec) -> pd.DataFrame:
    df = pd.read_csv(spec.input_csv)
    mini = _stratified_sample(df, n=spec.n_samples, strata_cols=spec.strata_cols, seed=spec.seed)
    out_path = Path(spec.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mini.to_csv(out_path, index=False)
    return mini


def build_ctrate_radgenome_minis(
    ctrate_csv: str,
    radgenome_csv: str,
    out_dir: str,
    seed: int = 42,
    ctrate_strata: Sequence[str] = ("split",),
    radgenome_strata: Sequence[str] = ("split",),
) -> List[str]:
    out = []
    out_base = Path(out_dir)
    out_base.mkdir(parents=True, exist_ok=True)

    ct_out = out_base / "ctrate_mini450.csv"
    rg_out = out_base / "radgenome_mini450.csv"
    build_mini_subset(
        MiniSubsetSpec(
            input_csv=ctrate_csv,
            output_csv=str(ct_out),
            n_samples=450,
            strata_cols=ctrate_strata,
            seed=seed,
        )
    )
    build_mini_subset(
        MiniSubsetSpec(
            input_csv=radgenome_csv,
            output_csv=str(rg_out),
            n_samples=450,
            strata_cols=radgenome_strata,
            seed=seed,
        )
    )
    out.extend([str(ct_out), str(rg_out)])
    return out
