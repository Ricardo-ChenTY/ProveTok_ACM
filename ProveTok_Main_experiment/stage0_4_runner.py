from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .config import ProveTokConfig
from .simple_modules import ReportSentencePlanner, RuleBasedAnatomyResolver
from .stage0_scorer import DeterministicArtifactScorer
from .stage1_swinunetr_encoder import FrozenSwinUNETREncoder
from .stage2_octree_splitter import AdaptiveOctreeSplitter
from .stage3_router import Router
from .stage4_verifier import Verifier
from .stage5_llm_judge import LLMJudge
from .token_bank_io import save_token_bank_case
from .types import BBox3D, SentenceOutput


@dataclass
class Stage04Components:
    artifact_scorer: DeterministicArtifactScorer
    encoder: FrozenSwinUNETREncoder
    splitter: AdaptiveOctreeSplitter
    planner: ReportSentencePlanner
    anatomy_resolver: RuleBasedAnatomyResolver
    router: Router
    verifier: Verifier
    llm_judge: Optional[LLMJudge] = None


def run_case_stage0_4(
    case_id: str,
    report_text: str,
    volume,
    spacing_xyz_mm: Tuple[float, float, float],
    out_case_dir: str,
    cfg: ProveTokConfig,
    comp: Stage04Components,
) -> Dict[str, object]:
    out_dir = Path(out_case_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    artifact_state = comp.artifact_scorer.score(volume, case_id=case_id)
    encoded = comp.encoder.encode(volume, case_id=case_id)
    tokens = comp.splitter.build_tokens(
        volume=volume,
        encoded=encoded,
        artifact_state=artifact_state,
        token_budget_b=cfg.split.token_budget_b,
    )

    d, h, w = volume.shape
    global_bbox_voxel = BBox3D(x_min=0.0, x_max=float(w), y_min=0.0, y_max=float(h), z_min=0.0, z_max=float(d))
    save_token_bank_case(
        out_case_dir=str(out_dir),
        tokens=tokens,
        cfg=cfg,
        spacing_xyz_mm=spacing_xyz_mm,
        encoder_name=comp.encoder.model.__class__.__name__,
        global_bbox_voxel=global_bbox_voxel,
    )

    comp.planner.set_report(report_text)
    comp.anatomy_resolver.volume_shape = volume.shape
    comp.verifier.volume_shape = tuple(int(x) for x in volume.shape)
    plans = comp.planner.plan(tokens)

    sentence_outputs: List[SentenceOutput] = []
    sentence_logs: List[Dict[str, object]] = []
    for plan in plans:
        q_s = comp.router.text_encoder(plan.topic)
        anatomy_bbox = comp.anatomy_resolver(plan.anatomy_keyword)
        scores = comp.router.score_tokens(plan.topic, tokens, anatomy_bbox)
        topk_ids = sorted(scores.keys(), key=lambda tid: (-scores[tid], tid))[: cfg.router.k_per_sentence]
        topk_scores = [float(scores[tid]) for tid in topk_ids]
        sentence_outputs.append(
            SentenceOutput(
                sentence_index=plan.sentence_index,
                text=plan.topic,
                citations=topk_ids,
                route_scores=scores,
            )
        )
        sentence_logs.append(
            {
                "sentence_index": int(plan.sentence_index),
                "sentence_text": plan.topic,
                "anatomy_keyword": plan.anatomy_keyword,
                "q_s": [float(x) for x in q_s],
                "topk_token_ids": [int(x) for x in topk_ids],
                "topk_scores": topk_scores,
            }
        )

    audits = comp.verifier.audit_all(sentence_outputs, plans, tokens)

    # Stage 5: LLM judge — confirm/dismiss violations and apply score penalty
    stage5_judgements: Dict[int, object] = {}
    if comp.llm_judge is not None:
        judgements = comp.llm_judge.judge_all(sentence_outputs, audits)
        for s_out in sentence_outputs:
            j = judgements.get(s_out.sentence_index)
            if j is None or not j.any_confirmed():
                continue
            # Apply CP .tex penalty: S'_i = S_i * (1 - alpha * sev_i)
            penalized = comp.llm_judge.reroute_scores(s_out.route_scores, j.verdicts)
            s_out.route_scores = penalized
            s_out.rerouted = True
            s_out.stop_reason = "llm_judge_penalty"
            stage5_judgements[s_out.sentence_index] = [
                {
                    "rule_id": v.rule_id,
                    "confirmed": v.confirmed,
                    "adjusted_severity": v.adjusted_severity,
                    "reasoning": v.reasoning,
                }
                for v in j.verdicts
            ]

    violations_by_sentence = {a.sentence_index: [asdict(v) for v in a.violations] for a in audits}
    for row in sentence_logs:
        row["violations"] = violations_by_sentence.get(row["sentence_index"], [])
        row["stage5_judgements"] = stage5_judgements.get(row["sentence_index"], [])

    b_plan = cfg.router.planning_budget(cfg.split.token_budget_b)
    trace_jsonl = out_dir / "trace.jsonl"
    with trace_jsonl.open("w", encoding="utf-8") as f:
        case_meta = {
            "type": "case_meta",
            "case_id": case_id,
            "B": int(cfg.split.token_budget_b),
            "k": int(cfg.router.k_per_sentence),
            "B_plan": int(b_plan),
            "lambda_spatial": float(cfg.router.lambda_spatial),
            "tau_IoU": float(cfg.verifier.tau_anatomy_iou),
            "ell_coarse": int(cfg.router.planning_level_cutoff),
            "beta": float(cfg.split.beta),
            "n_sentences": len(sentence_logs),
        }
        f.write(json.dumps(case_meta, ensure_ascii=False) + "\n")
        for s in sentence_logs:
            payload = {"type": "sentence", **s}
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    n_violate = sum(len(a.violations) for a in audits)
    n_judge_confirmed = sum(
        1
        for v_list in stage5_judgements.values()
        if isinstance(v_list, list) and any(v.get("confirmed") for v in v_list)  # type: ignore[union-attr]
    )
    return {
        "case_id": case_id,
        "n_tokens": len(tokens),
        "n_sentences": len(sentence_logs),
        "n_violations": int(n_violate),
        "n_judge_confirmed": int(n_judge_confirmed),
        "trace_jsonl": str(trace_jsonl),
    }
