import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import torch


REQUIRED_CASE_FILES = [
    "tokens.npy",
    "tokens.pt",
    "tokens.json",
    "bank_meta.json",
    "trace.jsonl",
]

REQUIRED_TOKEN_KEYS = {
    "token_id",
    "level",
    "bbox_3d_voxel",
    "bbox_3d_mm",
    "cached_boundary_flag",
    "cached_boundary_params",
}

REQUIRED_BANK_META_KEYS = {
    "B",
    "depth_max",
    "beta",
    "encoder_name",
    "voxel_spacing_mm_xyz",
    "global_bbox_voxel",
    "global_bbox_mm",
}

REQUIRED_CASE_META_KEYS = {
    "type",
    "case_id",
    "B",
    "k",
    "B_plan",
    "lambda_spatial",
    "tau_IoU",
    "ell_coarse",
    "beta",
    "n_sentences",
}

REQUIRED_SENTENCE_KEYS = {
    "type",
    "sentence_index",
    "sentence_text",
    "q_s",
    "topk_token_ids",
    "topk_scores",
    "violations",
}


@dataclass
class CaseValidation:
    dataset: str
    case_id: str
    passed: bool
    errors: List[str]
    warnings: List[str]


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _is_non_increasing(values: List[float]) -> bool:
    return all(values[i] >= values[i + 1] for i in range(len(values) - 1))


def validate_case(case_dir: Path, dataset: str) -> CaseValidation:
    errors: List[str] = []
    warnings: List[str] = []
    case_id = case_dir.name

    for fname in REQUIRED_CASE_FILES:
        if not (case_dir / fname).exists():
            errors.append(f"missing file: {fname}")

    if errors:
        return CaseValidation(dataset=dataset, case_id=case_id, passed=False, errors=errors, warnings=warnings)

    tokens_npy = np.load(case_dir / "tokens.npy")
    tokens_pt = torch.load(case_dir / "tokens.pt", map_location="cpu")
    if isinstance(tokens_pt, torch.Tensor):
        tokens_pt_arr = tokens_pt.numpy()
    else:
        errors.append("tokens.pt is not a tensor")
        tokens_pt_arr = None

    if tokens_npy.ndim != 2:
        errors.append(f"tokens.npy must be 2D [B, d], got shape={tokens_npy.shape}")
    if tokens_pt_arr is not None and tokens_pt_arr.ndim != 2:
        errors.append(f"tokens.pt must be 2D [B, d], got shape={tokens_pt_arr.shape}")
    if tokens_pt_arr is not None and tokens_npy.shape != tokens_pt_arr.shape:
        errors.append(f"tokens.npy/tokens.pt shape mismatch: {tokens_npy.shape} vs {tokens_pt_arr.shape}")

    tokens_json = _load_json(case_dir / "tokens.json")
    if not isinstance(tokens_json, list):
        errors.append("tokens.json must be a list")
        tokens_json = []
    if len(tokens_json) != tokens_npy.shape[0]:
        errors.append(f"tokens.json length {len(tokens_json)} != tokens.npy B {tokens_npy.shape[0]}")

    token_ids: Set[int] = set()
    for i, t in enumerate(tokens_json):
        if not isinstance(t, dict):
            errors.append(f"tokens.json[{i}] is not an object")
            continue
        miss = REQUIRED_TOKEN_KEYS - set(t.keys())
        if miss:
            errors.append(f"tokens.json[{i}] missing keys: {sorted(miss)}")
        if "token_id" in t:
            try:
                tid = int(t["token_id"])
                token_ids.add(tid)
            except Exception:
                errors.append(f"tokens.json[{i}].token_id is not int-like")

    if len(token_ids) != len(tokens_json):
        errors.append("token_id is not unique or invalid in tokens.json")

    bank_meta = _load_json(case_dir / "bank_meta.json")
    if not isinstance(bank_meta, dict):
        errors.append("bank_meta.json must be an object")
        bank_meta = {}
    miss_meta = REQUIRED_BANK_META_KEYS - set(bank_meta.keys())
    if miss_meta:
        errors.append(f"bank_meta.json missing keys: {sorted(miss_meta)}")
    if "B" in bank_meta:
        try:
            b = int(bank_meta["B"])
            if b != tokens_npy.shape[0]:
                errors.append(f"bank_meta.B {b} != tokens.npy B {tokens_npy.shape[0]}")
        except Exception:
            errors.append("bank_meta.B is not int-like")

    trace_rows = _load_jsonl(case_dir / "trace.jsonl")
    if len(trace_rows) == 0:
        errors.append("trace.jsonl is empty")
        return CaseValidation(dataset=dataset, case_id=case_id, passed=False, errors=errors, warnings=warnings)

    case_meta = trace_rows[0]
    if not isinstance(case_meta, dict):
        errors.append("trace.jsonl first line must be case_meta object")
        return CaseValidation(dataset=dataset, case_id=case_id, passed=False, errors=errors, warnings=warnings)
    miss_case_meta = REQUIRED_CASE_META_KEYS - set(case_meta.keys())
    if miss_case_meta:
        errors.append(f"trace case_meta missing keys: {sorted(miss_case_meta)}")
    if case_meta.get("type") != "case_meta":
        errors.append("trace first line type must be 'case_meta'")

    try:
        k = int(case_meta.get("k", 0))
    except Exception:
        k = 0
        errors.append("case_meta.k is invalid")

    sentence_rows = [r for r in trace_rows[1:] if isinstance(r, dict)]
    if int(case_meta.get("n_sentences", -1)) != len(sentence_rows):
        errors.append(
            f"case_meta.n_sentences {case_meta.get('n_sentences')} != actual sentence lines {len(sentence_rows)}"
        )

    for i, s in enumerate(sentence_rows):
        miss_sent = REQUIRED_SENTENCE_KEYS - set(s.keys())
        if miss_sent:
            errors.append(f"sentence[{i}] missing keys: {sorted(miss_sent)}")
            continue
        if s.get("type") != "sentence":
            errors.append(f"sentence[{i}] type must be 'sentence'")

        q_s = s.get("q_s")
        if not isinstance(q_s, list) or len(q_s) == 0:
            errors.append(f"sentence[{i}] q_s must be non-empty list")

        topk_ids = s.get("topk_token_ids")
        topk_scores = s.get("topk_scores")
        if not isinstance(topk_ids, list) or not isinstance(topk_scores, list):
            errors.append(f"sentence[{i}] topk fields must be list")
            continue
        if len(topk_ids) != len(topk_scores):
            errors.append(f"sentence[{i}] topk ids/scores length mismatch")
        if k > 0 and len(topk_ids) > k:
            errors.append(f"sentence[{i}] topk length {len(topk_ids)} > k={k}")
        if len(set(topk_ids)) != len(topk_ids):
            warnings.append(f"sentence[{i}] duplicated token ids in topk")
        missing_ids = [tid for tid in topk_ids if int(tid) not in token_ids]
        if missing_ids:
            errors.append(f"sentence[{i}] topk has unknown token ids: {missing_ids[:5]}")
        try:
            score_vals = [float(x) for x in topk_scores]
            if not _is_non_increasing(score_vals):
                warnings.append(f"sentence[{i}] topk_scores not strictly sorted non-increasing")
        except Exception:
            errors.append(f"sentence[{i}] topk_scores contain non-numeric value")

        if not isinstance(s.get("violations"), list):
            errors.append(f"sentence[{i}] violations must be a list")

    return CaseValidation(
        dataset=dataset,
        case_id=case_id,
        passed=(len(errors) == 0),
        errors=errors,
        warnings=warnings,
    )


def validate_outputs(out_dir: Path, datasets: List[str]) -> Tuple[List[CaseValidation], int]:
    results: List[CaseValidation] = []
    for ds in datasets:
        ds_dir = out_dir / "cases" / ds
        if not ds_dir.exists():
            results.append(
                CaseValidation(
                    dataset=ds,
                    case_id="__dataset__",
                    passed=False,
                    errors=[f"missing dataset dir: {ds_dir}"],
                    warnings=[],
                )
            )
            continue
        case_dirs = [p for p in ds_dir.iterdir() if p.is_dir()]
        if len(case_dirs) == 0:
            results.append(
                CaseValidation(
                    dataset=ds,
                    case_id="__dataset__",
                    passed=False,
                    errors=[f"no case dirs under: {ds_dir}"],
                    warnings=[],
                )
            )
            continue
        for cdir in sorted(case_dirs):
            results.append(validate_case(cdir, ds))

    n_failed = sum(1 for r in results if not r.passed)
    return results, n_failed


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Stage0-4 output folder against A/B/C/D expectations.")
    parser.add_argument("--out_dir", type=str, default="outputs_stage0_4")
    parser.add_argument("--datasets", type=str, default="ctrate,radgenome")
    parser.add_argument("--save_report", type=str, default="")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]

    results, n_failed = validate_outputs(out_dir, datasets)
    n_total = len(results)
    n_pass = n_total - n_failed

    print(f"Validated cases: {n_total}")
    print(f"Passed: {n_pass}")
    print(f"Failed: {n_failed}")

    for r in results:
        if r.passed:
            continue
        print(f"\n[FAIL] {r.dataset}/{r.case_id}")
        for e in r.errors:
            print(f"  - ERROR: {e}")
        for w in r.warnings:
            print(f"  - WARN : {w}")

    if args.save_report:
        report_path = Path(args.save_report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in results], f, ensure_ascii=False, indent=2)
        print(f"\nSaved report: {report_path}")

    sys.exit(1 if n_failed > 0 else 0)


if __name__ == "__main__":
    main()
