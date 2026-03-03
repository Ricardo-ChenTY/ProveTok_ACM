from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import scipy.ndimage as ndi
import SimpleITK as sitk


def load_volume_with_meta(path: str) -> Tuple[np.ndarray, Dict[str, object]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    if p.suffix.lower() == ".npy":
        arr = np.load(p)
        vol = np.asarray(arr, dtype=np.float32)
        return vol, {
            "spacing_xyz_mm": [1.0, 1.0, 1.0],
            "origin_xyz_mm": [0.0, 0.0, 0.0],
            "direction": None,
            "source": str(p),
        }
    img = sitk.ReadImage(str(p))
    arr = sitk.GetArrayFromImage(img).astype(np.float32)  # [D,H,W]
    spacing_xyz = img.GetSpacing()  # x,y,z
    origin_xyz = img.GetOrigin()
    direction = list(img.GetDirection())
    return arr, {
        "spacing_xyz_mm": [float(spacing_xyz[0]), float(spacing_xyz[1]), float(spacing_xyz[2])],
        "origin_xyz_mm": [float(origin_xyz[0]), float(origin_xyz[1]), float(origin_xyz[2])],
        "direction": direction,
        "source": str(p),
    }


def load_volume(path: str) -> np.ndarray:
    vol, _ = load_volume_with_meta(path)
    return vol


def ct_intensity_normalize(volume: np.ndarray, clip_hu: Tuple[float, float] = (-1000.0, 400.0)) -> np.ndarray:
    v = np.asarray(volume, dtype=np.float32)
    v = np.clip(v, clip_hu[0], clip_hu[1])
    mean = float(np.mean(v))
    std = float(np.std(v)) + 1e-6
    return (v - mean) / std


def resize_volume(volume: np.ndarray, out_shape: Tuple[int, int, int]) -> np.ndarray:
    in_shape = volume.shape
    zoom = (
        out_shape[0] / max(in_shape[0], 1),
        out_shape[1] / max(in_shape[1], 1),
        out_shape[2] / max(in_shape[2], 1),
    )
    return ndi.zoom(volume, zoom=zoom, order=1).astype(np.float32)


def resampled_spacing_xyz_mm(
    original_shape_dhw: Tuple[int, int, int],
    output_shape_dhw: Tuple[int, int, int],
    original_spacing_xyz_mm: Tuple[float, float, float],
) -> Tuple[float, float, float]:
    d0, h0, w0 = original_shape_dhw
    d1, h1, w1 = output_shape_dhw
    sx, sy, sz = original_spacing_xyz_mm
    # x maps to W, y maps to H, z maps to D
    sx_new = sx * (w0 / max(w1, 1))
    sy_new = sy * (h0 / max(h1, 1))
    sz_new = sz * (d0 / max(d1, 1))
    return (float(sx_new), float(sy_new), float(sz_new))
