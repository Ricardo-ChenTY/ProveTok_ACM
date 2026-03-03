from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np

from .config import SplitConfig
from .math_utils import clamp


@dataclass(frozen=True)
class CellBounds:
    z0: int
    z1: int
    y0: int
    y1: int
    x0: int
    x1: int

    def shape(self) -> Tuple[int, int, int]:
        return (self.z1 - self.z0, self.y1 - self.y0, self.x1 - self.x0)


@dataclass
class ArtifactComponents:
    snr_inv: float
    streak: float
    outlier: float
    a_i: float = 0.0


def _minmax_norm(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    vmin = min(values)
    vmax = max(values)
    if vmax <= vmin:
        return [0.5 for _ in values]
    scale = vmax - vmin
    return [(v - vmin) / scale for v in values]


def _cell_crop(volume: np.ndarray, b: CellBounds) -> np.ndarray:
    return volume[b.z0 : b.z1, b.y0 : b.y1, b.x0 : b.x1]


def _streak_score(crop: np.ndarray) -> float:
    # Stripe-like artifacts tend to have directional oscillation. Use axis anisotropy proxy.
    gx, gy, gz = np.gradient(crop.astype(np.float32), edge_order=1)
    v = np.array([np.mean(np.abs(gx)), np.mean(np.abs(gy)), np.mean(np.abs(gz))], dtype=np.float32)
    return float(np.max(v) - np.min(v))


def _outlier_score(crop: np.ndarray, eps: float) -> float:
    flat = crop.reshape(-1).astype(np.float32)
    med = float(np.median(flat))
    mad = float(np.median(np.abs(flat - med))) + eps
    robust_z = np.abs(flat - med) / (1.4826 * mad)
    return float(np.mean(robust_z > 3.5))


def compute_artifact_components(
    volume: np.ndarray,
    cell_bounds: Sequence[CellBounds],
    cfg: SplitConfig,
) -> List[ArtifactComponents]:
    raw: List[ArtifactComponents] = []
    for b in cell_bounds:
        crop = _cell_crop(volume, b)
        mu = float(np.mean(crop))
        sigma = float(np.std(crop))
        snr_inv = sigma / (abs(mu) + cfg.epsilon)
        streak = _streak_score(crop)
        outlier = _outlier_score(crop, cfg.epsilon)
        raw.append(ArtifactComponents(snr_inv=snr_inv, streak=streak, outlier=outlier))

    snr_n = _minmax_norm([x.snr_inv for x in raw])
    streak_n = _minmax_norm([x.streak for x in raw])
    outlier_n = _minmax_norm([x.outlier for x in raw])

    out: List[ArtifactComponents] = []
    for comp, ns, nt, no in zip(raw, snr_n, streak_n, outlier_n):
        a_i = clamp(cfg.w_snr * ns + cfg.w_streak * nt + cfg.w_outlier * no, 0.0, 1.0)
        out.append(ArtifactComponents(snr_inv=comp.snr_inv, streak=comp.streak, outlier=comp.outlier, a_i=a_i))
    return out
