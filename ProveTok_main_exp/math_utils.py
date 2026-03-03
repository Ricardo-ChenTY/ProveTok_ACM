import math
from typing import Dict, Iterable, List, Sequence, Tuple


def clamp(x: float, low: float, high: float) -> float:
    return max(low, min(high, x))


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def l2_norm(v: Sequence[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def normalize_l2(v: Sequence[float]) -> List[float]:
    n = l2_norm(v)
    if n == 0.0:
        return [0.0 for _ in v]
    return [x / n for x in v]


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def matvec(m: Sequence[Sequence[float]], v: Sequence[float]) -> List[float]:
    return [dot(row, v) for row in m]


def quantile_rank(values: Sequence[float]) -> List[float]:
    n = len(values)
    if n == 0:
        return []
    pairs = sorted(enumerate(values), key=lambda x: (x[1], x[0]))
    out = [0.0] * n
    for rank, (idx, _) in enumerate(pairs, start=1):
        out[idx] = rank / n
    return out


def top_k_from_dict(scores: Dict[int, float], k: int) -> List[int]:
    return [tid for tid, _ in sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:k]]
