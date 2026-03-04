#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Dict, Tuple

import torch
from monai.networks.nets import SwinUNETR


def _download_if_needed(ckpt_path: str | None, ckpt_url: str | None) -> Tuple[Path, bool]:
    if ckpt_path:
        p = Path(ckpt_path).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {p}")
        return p, False
    if not ckpt_url:
        raise ValueError("Either --ckpt_path or --ckpt_url must be provided.")

    suffix = Path(ckpt_url).suffix or ".pt"
    tmp_dir = Path(tempfile.mkdtemp(prefix="ckpt_probe_"))
    dst = tmp_dir / f"downloaded_ckpt{suffix}"
    urllib.request.urlretrieve(ckpt_url, dst)
    return dst, True


def _extract_state_dict(obj: object) -> Dict[str, torch.Tensor]:
    if isinstance(obj, dict):
        for key in ("state_dict", "model_state_dict", "model", "net"):
            candidate = obj.get(key)
            if isinstance(candidate, dict) and candidate:
                if all(isinstance(v, torch.Tensor) for v in candidate.values()):
                    return candidate
        if obj and all(isinstance(v, torch.Tensor) for v in obj.values()):
            return obj  # already a state_dict
    raise ValueError("Cannot parse checkpoint into a tensor state_dict.")


def _build_model(in_channels: int, out_channels: int, feature_size: int) -> SwinUNETR:
    model = SwinUNETR(
        in_channels=in_channels,
        out_channels=out_channels,
        feature_size=feature_size,
        use_checkpoint=False,
        spatial_dims=3,
    )
    model.eval()
    return model


def _shape_tuple(x: torch.Tensor) -> Tuple[int, ...]:
    return tuple(int(v) for v in x.shape)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Quick probe for SwinUNETR checkpoint compatibility with current ProveTok config. "
            "It checks whether run_mini_experiment's current Stage-1 loader can actually load weights."
        )
    )
    parser.add_argument("--ckpt_path", type=str, default=None, help="Local checkpoint path.")
    parser.add_argument("--ckpt_url", type=str, default=None, help="Checkpoint URL to download and probe.")
    parser.add_argument("--in_channels", type=int, default=1)
    parser.add_argument("--out_channels", type=int, default=2)
    parser.add_argument("--feature_size", type=int, default=48)
    parser.add_argument(
        "--min_match_ratio",
        type=float,
        default=0.6,
        help="Minimum matched-key ratio (matched keys / model keys) to classify as compatible.",
    )
    parser.add_argument(
        "--min_matched_keys",
        type=int,
        default=100,
        help="Minimum number of exact-shape matched keys to classify as compatible.",
    )
    parser.add_argument("--save_report", type=str, default=None, help="Optional JSON report path.")
    args = parser.parse_args()

    ckpt_file, downloaded = _download_if_needed(args.ckpt_path, args.ckpt_url)
    model = _build_model(args.in_channels, args.out_channels, args.feature_size)
    model_state = model.state_dict()

    raw = torch.load(str(ckpt_file), map_location="cpu")
    ckpt_state = _extract_state_dict(raw)

    model_keys = set(model_state.keys())
    ckpt_keys = set(ckpt_state.keys())
    shared = sorted(model_keys.intersection(ckpt_keys))
    only_model = sorted(model_keys - ckpt_keys)
    only_ckpt = sorted(ckpt_keys - model_keys)

    matched = []
    mismatched = []
    for k in shared:
        ms = _shape_tuple(model_state[k])
        cs = _shape_tuple(ckpt_state[k])
        if ms == cs:
            matched.append(k)
        else:
            mismatched.append({"key": k, "model_shape": ms, "ckpt_shape": cs})

    load_error = None
    load_success = False
    try:
        # Mimic current Stage-1 logic exactly (strict=False, no key remap).
        model.load_state_dict(ckpt_state, strict=False)
        load_success = True
    except Exception as exc:  # pragma: no cover
        load_error = str(exc)

    model_key_count = len(model_keys)
    matched_count = len(matched)
    mismatched_count = len(mismatched)
    match_ratio = float(matched_count / max(1, model_key_count))

    compatible = bool(
        load_success
        and mismatched_count == 0
        and matched_count >= int(args.min_matched_keys)
        and match_ratio >= float(args.min_match_ratio)
    )

    reason = "compatible"
    if not compatible:
        if not load_success:
            reason = "load_state_dict_runtime_error"
        elif mismatched_count > 0:
            reason = "shape_mismatch"
        elif matched_count == 0:
            reason = "no_effective_key_match"
        elif matched_count < int(args.min_matched_keys) or match_ratio < float(args.min_match_ratio):
            reason = "insufficient_weight_coverage"

    report = {
        "compatible": compatible,
        "reason": reason,
        "model_signature": {
            "name": "SwinUNETR",
            "in_channels": int(args.in_channels),
            "out_channels": int(args.out_channels),
            "feature_size": int(args.feature_size),
        },
        "checkpoint": {
            "path": str(ckpt_file),
            "downloaded_from_url": bool(downloaded),
            "url": str(args.ckpt_url) if args.ckpt_url else None,
        },
        "stats": {
            "model_key_count": model_key_count,
            "ckpt_key_count": len(ckpt_keys),
            "shared_key_count": len(shared),
            "matched_key_count": matched_count,
            "mismatched_key_count": mismatched_count,
            "match_ratio": match_ratio,
            "only_model_key_count": len(only_model),
            "only_ckpt_key_count": len(only_ckpt),
            "load_success": load_success,
        },
        "load_error": load_error,
        "examples": {
            "mismatched_keys_top20": mismatched[:20],
            "only_model_keys_top20": only_model[:20],
            "only_ckpt_keys_top20": only_ckpt[:20],
        },
    }

    if args.save_report:
        out = Path(args.save_report).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if compatible:
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
