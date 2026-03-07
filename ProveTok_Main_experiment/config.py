from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass
class SplitConfig:
    token_budget_b: int = 64
    init_depth: int = 2
    max_depth: int = 5
    min_voxels_to_split: int = 64
    split_score_threshold: float = 0.0
    nms_iou_threshold: float = 1.0
    tau_a: float = 0.85
    k_gate: float = 15.0
    lambda_h: float = 0.6
    beta: float = 0.1
    w_snr: float = 0.5
    w_streak: float = 0.3
    w_outlier: float = 0.2
    w_entropy: float = 0.5
    w_variance: float = 0.3
    w_boundary: float = 0.2
    epsilon: float = 1e-6


@dataclass
class RouterConfig:
    k_per_sentence: int = 8
    lambda_spatial: float = 0.3
    infonce_tau: float = 0.07
    planning_level_cutoff: int = 2
    planning_budget_cap: int = 32
    anatomy_spatial_routing: bool = False
    anatomy_tiebreak_eps: float = 0.05

    def planning_budget(self, token_budget_b: int) -> int:
        return min(self.planning_budget_cap, token_budget_b // 4)


@dataclass
class VerifierConfig:
    tau_anatomy_iou: float = 0.1
    r2_min_support_ratio: float = 1.0
    use_max_iou_for_r2: bool = False
    r4_disabled: bool = False
    r2_skip_keywords: set = field(default_factory=set)
    severity_by_rule: Dict[str, float] = field(
        default_factory=lambda: {
            "R1_LATERALITY": 1.0,
            "R2_ANATOMY": 0.8,
            "R3_DEPTH": 0.7,
            "R4_SIZE": 0.6,
            "R5_NEGATION": 1.0,
        }
    )
    lateral_tolerance: float = 0.0
    r5_fallback_lexicon: bool = True
    r5_fallback_severity: float = 0.5


@dataclass
class RerouteConfig:
    gamma_penalty: float = 2.0
    max_retry: int = 1


@dataclass
class ProveTokConfig:
    split: SplitConfig = field(default_factory=SplitConfig)
    router: RouterConfig = field(default_factory=RouterConfig)
    verifier: VerifierConfig = field(default_factory=VerifierConfig)
    reroute: RerouteConfig = field(default_factory=RerouteConfig)
