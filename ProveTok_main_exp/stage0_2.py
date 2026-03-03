from typing import Dict, Iterable, List, Optional, Sequence

from .config import SplitConfig
from .math_utils import clamp, quantile_rank, sigmoid
from .types import EvidenceToken


def artifact_risk_score(
    mu: float,
    sigma: float,
    streak: float,
    outlier: float,
    cfg: SplitConfig,
    norm_snr: float,
    norm_streak: float,
    norm_outlier: float,
) -> float:
    """
    CP Eq(1): A_i = clip(w_snr*norm(sigma/(|mu|+eps)) + w_st*norm(streak) + w_out*norm(outlier), 0, 1)
    """
    _ = sigma / (abs(mu) + cfg.epsilon)  # kept for API parity with Eq(1)
    score = (
        cfg.w_snr * norm_snr
        + cfg.w_streak * norm_streak
        + cfg.w_outlier * norm_outlier
    )
    return clamp(score, 0.0, 1.0)


def soft_gate(a_i: float, cfg: SplitConfig) -> float:
    """
    CP Eq(2): g_i = sigmoid(-k_gate * (A_i - tau_A))
    """
    return sigmoid(-cfg.k_gate * (a_i - cfg.tau_a))


def importance_score(g_i: float, h_q: float, p_q: float, cfg: SplitConfig) -> float:
    """
    CP Eq(5): S_i = g_i * (lambda_h*q(H_i) + lambda_p*q(P_i)), lambda_p = 1-lambda_h
    """
    lambda_p = 1.0 - cfg.lambda_h
    return g_i * (cfg.lambda_h * h_q + lambda_p * p_q)


def compute_importance_scores(
    a_values: Sequence[float],
    h_values: Sequence[float],
    p_values: Sequence[float],
    cfg: SplitConfig,
) -> List[float]:
    h_q = quantile_rank(h_values)
    p_q = quantile_rank(p_values)
    out: List[float] = []
    for a_i, h_i, p_i in zip(a_values, h_q, p_q):
        g_i = soft_gate(a_i, cfg)
        out.append(importance_score(g_i, h_i, p_i, cfg))
    return out


def boundary_context_blend(
    token_feature: Sequence[float],
    neighbor_features: Sequence[Sequence[float]],
    beta: float,
) -> List[float]:
    """
    CP Eq(6)(7): b_i = mean_{j in N(i)} f_j ; f_export = (1-beta)f_i + beta*b_i
    """
    if not neighbor_features:
        return list(token_feature)
    dim = len(token_feature)
    mean_neighbor = [0.0] * dim
    for nf in neighbor_features:
        for d, v in enumerate(nf):
            mean_neighbor[d] += v
    n = float(len(neighbor_features))
    mean_neighbor = [x / n for x in mean_neighbor]
    return [(1.0 - beta) * f + beta * b for f, b in zip(token_feature, mean_neighbor)]


def select_top_b(tokens: Sequence[EvidenceToken], b: int) -> List[EvidenceToken]:
    return sorted(tokens, key=lambda t: (-t.split_score, t.token_id))[:b]
