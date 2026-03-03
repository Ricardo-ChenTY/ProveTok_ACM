from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class BBox3D:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float

    def center(self) -> Tuple[float, float, float]:
        return (
            (self.x_min + self.x_max) / 2.0,
            (self.y_min + self.y_max) / 2.0,
            (self.z_min + self.z_max) / 2.0,
        )

    def volume(self) -> float:
        dx = max(0.0, self.x_max - self.x_min)
        dy = max(0.0, self.y_max - self.y_min)
        dz = max(0.0, self.z_max - self.z_min)
        return dx * dy * dz

    def iou(self, other: "BBox3D") -> float:
        ix_min = max(self.x_min, other.x_min)
        iy_min = max(self.y_min, other.y_min)
        iz_min = max(self.z_min, other.z_min)
        ix_max = min(self.x_max, other.x_max)
        iy_max = min(self.y_max, other.y_max)
        iz_max = min(self.z_max, other.z_max)
        inter = BBox3D(ix_min, ix_max, iy_min, iy_max, iz_min, iz_max).volume()
        if inter <= 0.0:
            return 0.0
        union = self.volume() + other.volume() - inter
        return 0.0 if union <= 0 else inter / union

    @staticmethod
    def union_all(boxes: Sequence["BBox3D"]) -> Optional["BBox3D"]:
        if not boxes:
            return None
        return BBox3D(
            x_min=min(b.x_min for b in boxes),
            x_max=max(b.x_max for b in boxes),
            y_min=min(b.y_min for b in boxes),
            y_max=max(b.y_max for b in boxes),
            z_min=min(b.z_min for b in boxes),
            z_max=max(b.z_max for b in boxes),
        )


@dataclass
class EvidenceToken:
    token_id: int
    bbox: BBox3D
    level: int
    feature: List[float]
    split_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SentencePlan:
    sentence_index: int
    topic: str
    anatomy_keyword: Optional[str] = None
    expected_level_range: Optional[Tuple[int, int]] = None
    expected_volume_range: Optional[Tuple[float, float]] = None
    is_negated: bool = False


@dataclass
class RouteResult:
    token_ids: List[int]
    scores: Dict[int, float]


@dataclass
class SentenceOutput:
    sentence_index: int
    text: str
    citations: List[int]
    route_scores: Dict[int, float]
    rerouted: bool = False
    stop_reason: str = "no_violation"


@dataclass
class RuleViolation:
    sentence_index: int
    rule_id: str
    severity: float
    message: str
    token_ids: List[int] = field(default_factory=list)


@dataclass
class SentenceAudit:
    sentence_index: int
    passed: bool
    violations: List[RuleViolation] = field(default_factory=list)
