from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
from monai.networks.nets import SwinUNETR


ArrayLike = Union[np.ndarray, torch.Tensor]


@dataclass
class FrozenSwinUNETREncoder:
    """
    Stage 1 (CP/tex): one-shot frozen 3D encoding.
    """

    img_size: Tuple[int, int, int] = (128, 128, 128)
    in_channels: int = 1
    out_channels: int = 2
    feature_size: int = 48
    use_checkpoint: bool = False
    checkpoint_path: Optional[str] = None
    device: str = "cuda"
    cache_dir: Optional[str] = None

    def __post_init__(self) -> None:
        self._device = torch.device(self.device if torch.cuda.is_available() else "cpu")
        self.model = SwinUNETR(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            feature_size=self.feature_size,
            use_checkpoint=self.use_checkpoint,
            spatial_dims=3,
        )
        if self.checkpoint_path:
            ckpt = torch.load(self.checkpoint_path, map_location="cpu")
            state = ckpt.get("state_dict", ckpt)
            self.model.load_state_dict(state, strict=False)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.to(self._device)
        self._cache_dir = Path(self.cache_dir) if self.cache_dir else None
        if self._cache_dir is not None:
            self._cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _as_tensor(volume: ArrayLike) -> torch.Tensor:
        if isinstance(volume, torch.Tensor):
            x = volume.detach().float()
        else:
            x = torch.from_numpy(np.asarray(volume)).float()
        if x.ndim == 3:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.ndim == 4:
            x = x.unsqueeze(0)
        if x.ndim != 5:
            raise ValueError(f"Expected 3D volume, got shape={tuple(x.shape)}")
        return x

    @staticmethod
    def _extract_encoder_feature(model: SwinUNETR, x: torch.Tensor) -> torch.Tensor:
        # Preferred path in MONAI SwinUNETR: use backbone hidden states directly.
        if hasattr(model, "swinViT"):
            normalize = bool(getattr(model, "normalize", True))
            hs = model.swinViT(x, normalize)
            if isinstance(hs, (list, tuple)) and len(hs) > 0:
                feat = hs[-1]
                if isinstance(feat, (list, tuple)):
                    feat = feat[0]
                return feat
        # Fallback path: use full forward output if backbone hooks are unavailable.
        y = model(x)
        if isinstance(y, (list, tuple)):
            y = y[0]
        return y

    def encode(
        self,
        volume: ArrayLike,
        case_id: Optional[str] = None,
        use_cache: bool = True,
    ) -> np.ndarray:
        cache_path = None
        if self._cache_dir is not None and case_id:
            cache_path = self._cache_dir / f"{case_id}.npy"
            if use_cache and cache_path.exists():
                return np.load(cache_path)

        x = self._as_tensor(volume).to(self._device)
        with torch.no_grad():
            feat = self._extract_encoder_feature(self.model, x)
        out = feat.squeeze(0).detach().cpu().numpy()

        if cache_path is not None:
            np.save(cache_path, out)
        return out
