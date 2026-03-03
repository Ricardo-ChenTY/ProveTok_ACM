from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .config import SplitConfig
from .math_utils import quantile_rank
from .stage0_2 import boundary_context_blend, importance_score, soft_gate
from .stage0_artifacts import CellBounds, compute_artifact_components
from .types import BBox3D, EvidenceToken


@dataclass
class OctreeCell:
    bounds: CellBounds
    level: int
    feature: np.ndarray
    a_i: float = 0.0
    h_i_raw: float = 0.0
    p_i_raw: float = 0.0
    h_i_q: float = 0.0
    p_i_q: float = 0.0
    g_i: float = 0.0
    score: float = 0.0


def _axis_splits(length: int, parts: int) -> List[Tuple[int, int]]:
    edges = [int(round(i * length / parts)) for i in range(parts + 1)]
    return [(edges[i], edges[i + 1]) for i in range(parts) if edges[i + 1] > edges[i]]


def _build_initial_grid(shape: Tuple[int, int, int], init_depth: int) -> List[CellBounds]:
    d, h, w = shape
    parts = 2 ** init_depth
    z_bins = _axis_splits(d, parts)
    y_bins = _axis_splits(h, parts)
    x_bins = _axis_splits(w, parts)
    out: List[CellBounds] = []
    for z0, z1 in z_bins:
        for y0, y1 in y_bins:
            for x0, x1 in x_bins:
                out.append(CellBounds(z0=z0, z1=z1, y0=y0, y1=y1, x0=x0, x1=x1))
    return out


def _split_bounds_8(b: CellBounds) -> List[CellBounds]:
    zm = (b.z0 + b.z1) // 2
    ym = (b.y0 + b.y1) // 2
    xm = (b.x0 + b.x1) // 2
    z_ranges = [(b.z0, zm), (zm, b.z1)]
    y_ranges = [(b.y0, ym), (ym, b.y1)]
    x_ranges = [(b.x0, xm), (xm, b.x1)]
    out: List[CellBounds] = []
    for z0, z1 in z_ranges:
        for y0, y1 in y_ranges:
            for x0, x1 in x_ranges:
                if z1 > z0 and y1 > y0 and x1 > x0:
                    out.append(CellBounds(z0=z0, z1=z1, y0=y0, y1=y1, x0=x0, x1=x1))
    return out


def _cell_voxels(b: CellBounds) -> int:
    dz, dy, dx = b.shape()
    return dz * dy * dx


def _crop_feature(encoded: np.ndarray, b: CellBounds, vol_shape: Tuple[int, int, int]) -> np.ndarray:
    if encoded.ndim == 3:
        encoded = encoded[np.newaxis, ...]
    if encoded.ndim != 4:
        raise ValueError(f"Expected encoded feature shape [C,D,H,W], got {encoded.shape}")
    c, fd, fh, fw = encoded.shape
    vd, vh, vw = vol_shape

    z0 = int(np.floor(b.z0 * fd / vd))
    z1 = int(np.ceil(b.z1 * fd / vd))
    y0 = int(np.floor(b.y0 * fh / vh))
    y1 = int(np.ceil(b.y1 * fh / vh))
    x0 = int(np.floor(b.x0 * fw / vw))
    x1 = int(np.ceil(b.x1 * fw / vw))

    z0, z1 = max(0, z0), min(fd, max(z0 + 1, z1))
    y0, y1 = max(0, y0), min(fh, max(y0 + 1, y1))
    x0, x1 = max(0, x0), min(fw, max(x0 + 1, x1))
    return encoded[:, z0:z1, y0:y1, x0:x1]


def _pooled_feature(encoded: np.ndarray, b: CellBounds, vol_shape: Tuple[int, int, int]) -> np.ndarray:
    crop = _crop_feature(encoded, b, vol_shape)
    return np.mean(crop, axis=(1, 2, 3))


def _uncertainty_h(crop_feat: np.ndarray, cfg: SplitConfig) -> float:
    eps = cfg.epsilon
    c = crop_feat.shape[0]
    channel_energy = np.mean(np.abs(crop_feat), axis=(1, 2, 3)) + eps
    probs = channel_energy / np.sum(channel_energy)
    ent = float(-np.sum(probs * np.log(probs + eps)) / max(np.log(max(c, 2)), eps))
    var = float(np.var(crop_feat))
    fmap = np.mean(crop_feat, axis=0)
    if min(fmap.shape) < 2:
        boundary = 0.0
    else:
        gz, gy, gx = np.gradient(fmap.astype(np.float32), edge_order=1)
        boundary = float(np.mean(np.sqrt(gz * gz + gy * gy + gx * gx)))
    return cfg.w_entropy * ent + cfg.w_variance * var + cfg.w_boundary * boundary


def _semantic_prior_p(crop_feat: np.ndarray) -> float:
    return float(np.mean(np.abs(crop_feat)))


def _bbox_iou(a: CellBounds, b: CellBounds) -> float:
    az0, az1, ay0, ay1, ax0, ax1 = a.z0, a.z1, a.y0, a.y1, a.x0, a.x1
    bz0, bz1, by0, by1, bx0, bx1 = b.z0, b.z1, b.y0, b.y1, b.x0, b.x1
    iz0, iz1 = max(az0, bz0), min(az1, bz1)
    iy0, iy1 = max(ay0, by0), min(ay1, by1)
    ix0, ix1 = max(ax0, bx0), min(ax1, bx1)
    inter = max(0, iz1 - iz0) * max(0, iy1 - iy0) * max(0, ix1 - ix0)
    if inter <= 0:
        return 0.0
    va = max(0, az1 - az0) * max(0, ay1 - ay0) * max(0, ax1 - ax0)
    vb = max(0, bz1 - bz0) * max(0, by1 - by0) * max(0, bx1 - bx0)
    union = va + vb - inter
    return 0.0 if union <= 0 else inter / union


def _nms_cells(cells: Sequence[OctreeCell], iou_threshold: float, top_b: int) -> List[OctreeCell]:
    if iou_threshold >= 1.0:
        return list(cells[:top_b])
    kept: List[OctreeCell] = []
    for c in cells:
        if len(kept) >= top_b:
            break
        if all(_bbox_iou(c.bounds, k.bounds) < iou_threshold for k in kept):
            kept.append(c)
    if len(kept) < top_b:
        remaining = [c for c in cells if c not in kept]
        kept.extend(remaining[: top_b - len(kept)])
    return kept[:top_b]


def _are_face_neighbors(a: CellBounds, b: CellBounds) -> bool:
    def overlap_1d(p0: int, p1: int, q0: int, q1: int) -> bool:
        return min(p1, q1) > max(p0, q0)

    touch_x = (a.x1 == b.x0 or b.x1 == a.x0) and overlap_1d(a.y0, a.y1, b.y0, b.y1) and overlap_1d(a.z0, a.z1, b.z0, b.z1)
    touch_y = (a.y1 == b.y0 or b.y1 == a.y0) and overlap_1d(a.x0, a.x1, b.x0, b.x1) and overlap_1d(a.z0, a.z1, b.z0, b.z1)
    touch_z = (a.z1 == b.z0 or b.z1 == a.z0) and overlap_1d(a.x0, a.x1, b.x0, b.x1) and overlap_1d(a.y0, a.y1, b.y0, b.y1)
    return touch_x or touch_y or touch_z


class AdaptiveOctreeSplitter:
    """
    Stage 2 executable implementation under CP equations:
    Eq(1) artifact risk, Eq(2) soft gate, Eq(4) quantile rank, Eq(5) score, Eq(6-7) boundary blend.
    """

    def __init__(self, cfg: SplitConfig) -> None:
        self.cfg = cfg

    def _recompute_scores(self, volume: np.ndarray, encoded: np.ndarray, cells: List[OctreeCell]) -> None:
        bounds = [c.bounds for c in cells]
        artifact = compute_artifact_components(volume, bounds, self.cfg)
        h_raw: List[float] = []
        p_raw: List[float] = []
        for i, c in enumerate(cells):
            c.a_i = artifact[i].a_i
            crop_feat = _crop_feature(encoded, c.bounds, volume.shape)
            c.h_i_raw = _uncertainty_h(crop_feat, self.cfg)
            c.p_i_raw = _semantic_prior_p(crop_feat)
            h_raw.append(c.h_i_raw)
            p_raw.append(c.p_i_raw)

        # Per-level quantile rank (CP Eq(4), frozen per round).
        by_level: Dict[int, List[int]] = {}
        for idx, c in enumerate(cells):
            by_level.setdefault(c.level, []).append(idx)
        h_q = [0.0 for _ in cells]
        p_q = [0.0 for _ in cells]
        for _, idxs in by_level.items():
            h_vals = [h_raw[i] for i in idxs]
            p_vals = [p_raw[i] for i in idxs]
            h_rank = quantile_rank(h_vals)
            p_rank = quantile_rank(p_vals)
            for j, gi in enumerate(idxs):
                h_q[gi] = h_rank[j]
                p_q[gi] = p_rank[j]

        for i, c in enumerate(cells):
            c.h_i_q = h_q[i]
            c.p_i_q = p_q[i]
            c.g_i = soft_gate(c.a_i, self.cfg)
            c.score = importance_score(c.g_i, c.h_i_q, c.p_i_q, self.cfg)
            c.feature = _pooled_feature(encoded, c.bounds, volume.shape)

    def build_tokens(
        self,
        volume: object,
        encoded: object,
        artifact_state: object,
        token_budget_b: int,
    ) -> List[EvidenceToken]:
        _ = artifact_state  # Stage 0 precompute can be threaded in here if needed.
        vol = np.asarray(volume, dtype=np.float32)
        if vol.ndim != 3:
            raise ValueError(f"Expected volume shape [D,H,W], got {vol.shape}")
        feat = np.asarray(encoded, dtype=np.float32)
        if feat.ndim == 5:
            feat = feat[0]
        if feat.ndim == 3:
            feat = feat[np.newaxis, ...]

        init_bounds = _build_initial_grid(vol.shape, self.cfg.init_depth)
        leaves: List[OctreeCell] = [
            OctreeCell(bounds=b, level=self.cfg.init_depth, feature=np.zeros((feat.shape[0],), dtype=np.float32))
            for b in init_bounds
        ]
        self._recompute_scores(vol, feat, leaves)

        # Adaptive split loop
        while len(leaves) < token_budget_b:
            leaves_sorted = sorted(leaves, key=lambda c: (-c.score, c.level, c.bounds.z0, c.bounds.y0, c.bounds.x0))
            best = None
            for c in leaves_sorted:
                if c.level >= self.cfg.max_depth:
                    continue
                if _cell_voxels(c.bounds) < self.cfg.min_voxels_to_split:
                    continue
                if c.score <= self.cfg.split_score_threshold:
                    continue
                best = c
                break
            if best is None:
                break
            leaves.remove(best)
            for child_b in _split_bounds_8(best.bounds):
                leaves.append(
                    OctreeCell(
                        bounds=child_b,
                        level=best.level + 1,
                        feature=np.zeros((feat.shape[0],), dtype=np.float32),
                    )
                )
            self._recompute_scores(vol, feat, leaves)

        leaves_sorted = sorted(leaves, key=lambda c: (-c.score, c.bounds.z0, c.bounds.y0, c.bounds.x0))
        selected = _nms_cells(leaves_sorted, self.cfg.nms_iou_threshold, top_b=token_budget_b)

        # Boundary-aware export
        raw_feats = [c.feature for c in selected]
        blended_feats: List[np.ndarray] = []
        for i, c in enumerate(selected):
            neigh = [
                raw_feats[j]
                for j, other in enumerate(selected)
                if i != j and _are_face_neighbors(c.bounds, other.bounds)
            ]
            blend = boundary_context_blend(c.feature.tolist(), [n.tolist() for n in neigh], self.cfg.beta)
            blended_feats.append(np.asarray(blend, dtype=np.float32))

        # Stable token ids by geometry order.
        ordered = sorted(
            list(zip(selected, blended_feats)),
            key=lambda x: (x[0].level, x[0].bounds.z0, x[0].bounds.y0, x[0].bounds.x0),
        )
        out: List[EvidenceToken] = []
        for tid, (c, fexp) in enumerate(ordered):
            b = c.bounds
            out.append(
                EvidenceToken(
                    token_id=tid,
                    bbox=BBox3D(
                        x_min=float(b.x0),
                        x_max=float(b.x1),
                        y_min=float(b.y0),
                        y_max=float(b.y1),
                        z_min=float(b.z0),
                        z_max=float(b.z1),
                    ),
                    level=c.level,
                    feature=fexp.astype(np.float32).tolist(),
                    split_score=float(c.score),
                    metadata={
                        "a_i": float(c.a_i),
                        "h_i_raw": float(c.h_i_raw),
                        "p_i_raw": float(c.p_i_raw),
                        "h_i_q": float(c.h_i_q),
                        "p_i_q": float(c.p_i_q),
                        "g_i": float(c.g_i),
                        "split_source": "adaptive_octree",
                    },
                )
            )
        return out
