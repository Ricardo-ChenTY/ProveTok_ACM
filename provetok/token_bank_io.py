from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from .config import ProveTokConfig
from .types import BBox3D, EvidenceToken


def _bbox_voxel_to_mm(b: BBox3D, spacing_xyz_mm: Tuple[float, float, float]) -> BBox3D:
    sx, sy, sz = spacing_xyz_mm
    return BBox3D(
        x_min=b.x_min * sx,
        x_max=b.x_max * sx,
        y_min=b.y_min * sy,
        y_max=b.y_max * sy,
        z_min=b.z_min * sz,
        z_max=b.z_max * sz,
    )


def _token_json_obj(token: EvidenceToken, spacing_xyz_mm: Tuple[float, float, float], beta: float) -> Dict[str, object]:
    b_mm = _bbox_voxel_to_mm(token.bbox, spacing_xyz_mm)
    return {
        "token_id": int(token.token_id),
        "level": int(token.level),
        "bbox_3d_voxel": {
            "x_min": float(token.bbox.x_min),
            "x_max": float(token.bbox.x_max),
            "y_min": float(token.bbox.y_min),
            "y_max": float(token.bbox.y_max),
            "z_min": float(token.bbox.z_min),
            "z_max": float(token.bbox.z_max),
        },
        "bbox_3d_mm": {
            "x_min": float(b_mm.x_min),
            "x_max": float(b_mm.x_max),
            "y_min": float(b_mm.y_min),
            "y_max": float(b_mm.y_max),
            "z_min": float(b_mm.z_min),
            "z_max": float(b_mm.z_max),
        },
        "cached_boundary_flag": bool(beta > 0.0),
        "cached_boundary_params": {
            "beta": float(beta),
            "neighbor_mode": "6-connected",
        },
    }


def save_token_bank_case(
    out_case_dir: str,
    tokens: Sequence[EvidenceToken],
    cfg: ProveTokConfig,
    spacing_xyz_mm: Tuple[float, float, float],
    encoder_name: str,
    global_bbox_voxel: BBox3D,
) -> None:
    out_dir = Path(out_case_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    feats = np.asarray([t.feature for t in tokens], dtype=np.float32)
    np.save(out_dir / "tokens.npy", feats)
    torch.save(torch.from_numpy(feats), out_dir / "tokens.pt")

    tokens_json = [_token_json_obj(t, spacing_xyz_mm=spacing_xyz_mm, beta=cfg.split.beta) for t in tokens]
    with (out_dir / "tokens.json").open("w", encoding="utf-8") as f:
        json.dump(tokens_json, f, ensure_ascii=False, indent=2)

    global_bbox_mm = _bbox_voxel_to_mm(global_bbox_voxel, spacing_xyz_mm)
    bank_meta = {
        "B": int(len(tokens)),
        "depth_max": int(cfg.split.max_depth),
        "beta": float(cfg.split.beta),
        "encoder_name": encoder_name,
        "voxel_spacing_mm_xyz": [float(x) for x in spacing_xyz_mm],
        "global_bbox_voxel": {
            "x_min": float(global_bbox_voxel.x_min),
            "x_max": float(global_bbox_voxel.x_max),
            "y_min": float(global_bbox_voxel.y_min),
            "y_max": float(global_bbox_voxel.y_max),
            "z_min": float(global_bbox_voxel.z_min),
            "z_max": float(global_bbox_voxel.z_max),
        },
        "global_bbox_mm": {
            "x_min": float(global_bbox_mm.x_min),
            "x_max": float(global_bbox_mm.x_max),
            "y_min": float(global_bbox_mm.y_min),
            "y_max": float(global_bbox_mm.y_max),
            "z_min": float(global_bbox_mm.z_min),
            "z_max": float(global_bbox_mm.z_max),
        },
    }
    with (out_dir / "bank_meta.json").open("w", encoding="utf-8") as f:
        json.dump(bank_meta, f, ensure_ascii=False, indent=2)
