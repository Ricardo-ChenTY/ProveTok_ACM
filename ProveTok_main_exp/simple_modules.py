from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .types import BBox3D, EvidenceToken, SentencePlan


DEFAULT_ANATOMY_BOXES: Dict[str, BBox3D] = {
    "right upper lobe": BBox3D(0.00, 0.45, 0.50, 1.00, 0.00, 1.00),
    "right lower lobe": BBox3D(0.00, 0.45, 0.00, 0.50, 0.00, 1.00),
    "left upper lobe": BBox3D(0.55, 1.00, 0.50, 1.00, 0.00, 1.00),
    "left lower lobe": BBox3D(0.55, 1.00, 0.00, 0.50, 0.00, 1.00),
    "mediastinum": BBox3D(0.40, 0.60, 0.20, 0.80, 0.00, 1.00),
    "bilateral": BBox3D(0.00, 1.00, 0.00, 1.00, 0.00, 1.00),
}


def normalize_box_to_volume(box: BBox3D, vol_shape: Sequence[int]) -> BBox3D:
    d, h, w = vol_shape
    return BBox3D(
        x_min=box.x_min * w,
        x_max=box.x_max * w,
        y_min=box.y_min * h,
        y_max=box.y_max * h,
        z_min=box.z_min * d,
        z_max=box.z_max * d,
    )


@dataclass
class RuleBasedAnatomyResolver:
    anatomy_boxes: Dict[str, BBox3D] = field(default_factory=lambda: dict(DEFAULT_ANATOMY_BOXES))
    volume_shape: Optional[Sequence[int]] = None

    def __call__(self, keyword: Optional[str]) -> Optional[BBox3D]:
        if not keyword:
            return None
        key = keyword.lower().strip()
        box = self.anatomy_boxes.get(key)
        if box is None:
            return None
        if self.volume_shape is None:
            return box
        return normalize_box_to_volume(box, self.volume_shape)


@dataclass
class ReportSentencePlanner:
    max_sentences: int = 8
    anatomy_keywords: Sequence[str] = tuple(DEFAULT_ANATOMY_BOXES.keys())
    _current_report: str = ""

    def set_report(self, text: str) -> None:
        self._current_report = text or ""

    def _extract_keyword(self, sentence: str) -> Optional[str]:
        s = sentence.lower()
        for k in self.anatomy_keywords:
            if k in s:
                return k
        if ("left" in s) and ("lower" in s):
            return "left lower lobe"
        if ("left" in s) and ("upper" in s):
            return "left upper lobe"
        if ("right" in s) and ("lower" in s):
            return "right lower lobe"
        if ("right" in s) and ("upper" in s):
            return "right upper lobe"
        if ("bilateral" in s) or ("both" in s):
            return "bilateral"
        return None

    @staticmethod
    def _expected_level_range(sentence: str) -> Optional[Tuple[int, int]]:
        s = sentence.lower()
        if any(k in s for k in ("diffuse", "bilateral", "both lungs", "全肺", "双侧")):
            return (0, 2)
        if any(k in s for k in ("nodule", "mass", "结节", "肿块", "focal")):
            return (2, 5)
        return None

    @staticmethod
    def _expected_volume_range(sentence: str) -> Optional[Tuple[float, float]]:
        s = sentence.lower()
        if any(k in s for k in ("tiny", "small", "<", "毫米", "mm")):
            return (0.0, 2.0e4)
        if any(k in s for k in ("large", "extensive", "大量")):
            return (5.0e4, 1.0e9)
        return None

    def plan(self, coarse_tokens: Sequence[EvidenceToken]) -> List[SentencePlan]:
        _ = coarse_tokens
        text = self._current_report.strip()
        if not text:
            return [SentencePlan(sentence_index=0, topic="no acute cardiopulmonary abnormality", anatomy_keyword=None)]

        raw = [x.strip() for x in re.split(r"[.\n]+", text) if x.strip()]
        plans: List[SentencePlan] = []
        for i, s in enumerate(raw[: self.max_sentences]):
            kw = self._extract_keyword(s)
            plans.append(
                SentencePlan(
                    sentence_index=i,
                    topic=s,
                    anatomy_keyword=kw,
                    expected_level_range=self._expected_level_range(s),
                    expected_volume_range=self._expected_volume_range(s),
                    is_negated=("no " in s.lower()) or ("without" in s.lower()) or ("未见" in s),
                )
            )
        return plans
