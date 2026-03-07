import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Set

from .config import RouterConfig
from .math_utils import dot, matvec, normalize_l2, top_k_from_dict
from .types import BBox3D, EvidenceToken, RouteResult


@dataclass
class Router:
    """
    CP M2 + M3:
    Eq(9) projected token, Eq(10) cosine routing, Eq(11) spatial prior, Eq(19) top-k.
    """

    cfg: RouterConfig
    text_encoder: Callable[[str], List[float]]
    w_proj: Optional[Sequence[Sequence[float]]] = None

    def _ensure_w_proj(self, d_v: int) -> None:
        if self.w_proj is not None:
            return
        q_dim = len(self.text_encoder("init"))
        mat = [[0.0 for _ in range(d_v)] for _ in range(q_dim)]
        for i in range(min(q_dim, d_v)):
            mat[i][i] = 1.0
        self.w_proj = mat

    def _projected_token(self, feature: Sequence[float]) -> List[float]:
        self._ensure_w_proj(len(feature))
        assert self.w_proj is not None
        v = matvec(self.w_proj, feature)
        return normalize_l2(v)

    def _routing_score(
        self,
        query: Sequence[float],
        token: EvidenceToken,
        anatomy_bbox: Optional[BBox3D],
    ) -> float:
        q = normalize_l2(query)
        v = self._projected_token(token.feature)
        base = dot(q, v)  # Eq(10), in [-1, 1]
        if anatomy_bbox is None:
            return base
        iou = token.bbox.iou(anatomy_bbox)
        if self.cfg.anatomy_spatial_routing:
            # Anatomy-primary mode: IoU dominates, semantic dot is a small tiebreaker.
            # Addresses cross-modal alignment gap: w_proj is identity (untrained),
            # so dot(text_query, image_feature) is unreliable. Spatial IoU is the
            # only grounded signal when anatomy_bbox is available.
            return iou + self.cfg.anatomy_tiebreak_eps * base
        return base + self.cfg.lambda_spatial * iou  # Eq(11)

    def score_tokens(
        self,
        topic: str,
        tokens: Sequence[EvidenceToken],
        anatomy_bbox: Optional[BBox3D],
    ) -> Dict[int, float]:
        q = self.text_encoder(topic)
        return {t.token_id: self._routing_score(q, t, anatomy_bbox) for t in tokens}

    def route(
        self,
        topic: str,
        tokens: Sequence[EvidenceToken],
        anatomy_bbox: Optional[BBox3D] = None,
        score_override: Optional[Dict[int, float]] = None,
    ) -> RouteResult:
        scores = score_override if score_override is not None else self.score_tokens(topic, tokens, anatomy_bbox)
        token_ids = top_k_from_dict(scores, self.cfg.k_per_sentence)
        return RouteResult(token_ids=token_ids, scores=scores)


def infonce_loss(
    scores_by_token: Dict[int, float],
    positive_token_ids: Set[int],
    tau: float = 0.07,
) -> float:
    """
    CP Eq(12): multi-positive InfoNCE on routing scores.
    """
    if not positive_token_ids:
        raise ValueError("InfoNCE undefined when |P_s|=0.")
    logits = {tid: s / tau for tid, s in scores_by_token.items()}
    m = max(logits.values()) if logits else 0.0
    exps = {tid: math.exp(v - m) for tid, v in logits.items()}
    denom = sum(exps.values())
    loss_terms: List[float] = []
    for tid in positive_token_ids:
        if tid not in exps:
            continue
        p = exps[tid] / denom
        loss_terms.append(-math.log(max(p, 1e-12)))
    if not loss_terms:
        raise ValueError("Positive token ids are missing from candidate set.")
    return sum(loss_terms) / len(loss_terms)
