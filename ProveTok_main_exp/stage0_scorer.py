from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class DeterministicArtifactScorer:
    """
    Stage 0 precompute hook. Current implementation caches deterministic
    gradient magnitude and robust intensity stats; Stage 2 can consume it if needed.
    """

    cache_dir: Optional[str] = None

    def __post_init__(self) -> None:
        self._cache_dir = Path(self.cache_dir) if self.cache_dir else None
        if self._cache_dir is not None:
            self._cache_dir.mkdir(parents=True, exist_ok=True)

    def score(self, volume: object, case_id: Optional[str] = None) -> Dict[str, Any]:
        vol = np.asarray(volume, dtype=np.float32)
        if vol.ndim != 3:
            raise ValueError(f"Expected 3D volume [D,H,W], got {vol.shape}")

        if self._cache_dir is not None and case_id:
            npz = self._cache_dir / f"{case_id}.npz"
            if npz.exists():
                loaded = np.load(npz)
                return {k: loaded[k] for k in loaded.files}

        gz, gy, gx = np.gradient(vol, edge_order=1)
        grad_mag = np.sqrt(gx * gx + gy * gy + gz * gz)
        med = float(np.median(vol))
        mad = float(np.median(np.abs(vol - med))) + 1e-6
        state = {
            "grad_mag": grad_mag.astype(np.float32),
            "median": np.float32(med),
            "mad": np.float32(mad),
        }

        if self._cache_dir is not None and case_id:
            np.savez_compressed(self._cache_dir / f"{case_id}.npz", **state)
        return state
