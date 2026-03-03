from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class DeterministicTextEncoder:
    dim: int = 256

    def __call__(self, text: str) -> List[float]:
        key = hashlib.sha256(text.encode("utf-8")).digest()
        seed = int.from_bytes(key[:8], byteorder="little", signed=False) % (2 ** 32 - 1)
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(self.dim).astype(np.float32)
        n = np.linalg.norm(v) + 1e-8
        return (v / n).tolist()
