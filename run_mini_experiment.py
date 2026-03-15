import argparse
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd

from ProveTok_Main_experiment.config import ProveTokConfig
from ProveTok_Main_experiment.dataset_tools import build_ctrate_radgenome_minis
from ProveTok_Main_experiment.preprocess import (
    ct_intensity_normalize,
    load_volume_with_meta,
    resize_volume,
    resampled_spacing_xyz_mm,
)
from ProveTok_Main_experiment.simple_modules import ReportSentencePlanner, RuleBasedAnatomyResolver
from ProveTok_Main_experiment.stage0_scorer import DeterministicArtifactScorer
from ProveTok_Main_experiment.stage1_swinunetr_encoder import FrozenSwinUNETREncoder
from ProveTok_Main_experiment.stage2_octree_splitter import AdaptiveOctreeSplitter
from ProveTok_Main_experiment.stage3_router import Router
from ProveTok_Main_experiment.stage4_verifier import Verifier
from ProveTok_Main_experiment.stage0_4_runner import run_case_stage0_4, Stage04Components
from ProveTok_Main_experiment.stage3c_generator import Stage3cGenerator, GeneratorConfig
from ProveTok_Main_experiment.stage5_llm_judge import LLMJudge
from ProveTok_Main_experiment.text_encoder import make_text_encoder


def _run_manifest(
    dataset_name: str,
    manifest_csv: str,
    out_dir: Path,
    cfg: ProveTokConfig,
    volume_col: str,
    report_col: str,
    case_id_col: str,
    max_cases: int,
    encoder_ckpt: Optional[str],
    device: str,
    resize_dhw: Tuple[int, int, int],
    text_encoder: Callable[[str], List[float]],
    expected_cases: int = 0,
    llm_judge: Optional[LLMJudge] = None,
    generator: Optional[Stage3cGenerator] = None,
    shuffle_seed: Optional[int] = None,
    w_proj: Optional[List[List[float]]] = None,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    df_all = pd.read_csv(manifest_csv)
    n_available = int(len(df_all))
    if shuffle_seed is not None:
        df_all = df_all.sample(frac=1, random_state=shuffle_seed).reset_index(drop=True)
    df = df_all.head(max_cases).reset_index(drop=True)
    n_selected = int(len(df))
    if expected_cases > 0 and n_selected != expected_cases:
        raise ValueError(
            f"{dataset_name}: expected {expected_cases} cases, but selected {n_selected} "
            f"(manifest rows={n_available}, max_cases={max_cases})."
        )
    if volume_col not in df.columns:
        raise ValueError(f"Missing column '{volume_col}' in {manifest_csv}")
    if report_col not in df.columns:
        raise ValueError(f"Missing column '{report_col}' in {manifest_csv}")
    if case_id_col not in df.columns:
        df[case_id_col] = [f"{dataset_name}_{i:05d}" for i in range(len(df))]
    df[case_id_col] = df[case_id_col].astype(str).str.strip()
    dup_mask = df[case_id_col].duplicated(keep=False)
    if bool(dup_mask.any()):
        dup_vals = sorted(df.loc[dup_mask, case_id_col].unique().tolist())
        preview = dup_vals[:10]
        suffix = "" if len(dup_vals) <= 10 else f" ... (+{len(dup_vals) - 10} more)"
        raise ValueError(
            f"{dataset_name}: duplicated case_id detected in manifest: {preview}{suffix}. "
            "Please ensure case_id is unique."
        )

    cache_root = out_dir / "cache" / dataset_name
    comp = Stage04Components(
        artifact_scorer=DeterministicArtifactScorer(cache_dir=str(cache_root / "stage0")),
        encoder=FrozenSwinUNETREncoder(
            img_size=resize_dhw,
            checkpoint_path=encoder_ckpt,
            device=device,
            cache_dir=str(cache_root / "stage1"),
        ),
        splitter=AdaptiveOctreeSplitter(cfg.split),
        planner=ReportSentencePlanner(max_sentences=8),
        anatomy_resolver=RuleBasedAnatomyResolver(),
        router=Router(cfg=cfg.router, text_encoder=text_encoder, w_proj=w_proj),
        verifier=Verifier(cfg.verifier, RuleBasedAnatomyResolver()),
        llm_judge=llm_judge,
        generator=generator,
    )
    comp.verifier = Verifier(cfg.verifier, comp.anatomy_resolver)

    case_root = out_dir / "cases" / dataset_name
    case_root.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    for _, row in df.iterrows():
        case_id = str(row[case_id_col])
        report_text = str(row[report_col])
        vol, meta = load_volume_with_meta(str(row[volume_col]))
        orig_shape = tuple(int(x) for x in vol.shape)
        vol = ct_intensity_normalize(vol)
        vol = resize_volume(vol, resize_dhw)
        spacing0 = tuple(float(x) for x in meta.get("spacing_xyz_mm", [1.0, 1.0, 1.0]))
        spacing_resampled = resampled_spacing_xyz_mm(orig_shape, resize_dhw, spacing0)

        result = run_case_stage0_4(
            case_id=case_id,
            report_text=report_text,
            volume=vol,
            spacing_xyz_mm=spacing_resampled,
            out_case_dir=str(case_root / case_id),
            cfg=cfg,
            comp=comp,
        )
        rows.append(
            {
                "dataset": dataset_name,
                "case_id": case_id,
                "n_tokens": result["n_tokens"],
                "n_sentences": result["n_sentences"],
                "n_violations": result["n_violations"],
                "n_judge_confirmed": result.get("n_judge_confirmed", 0),
                "n_generated": result.get("n_generated", 0),
                "n_rerouted": result.get("n_rerouted", 0),
                "n_despecified": result.get("n_despecified", 0),
                "trace_jsonl": result["trace_jsonl"],
            }
        )
    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_dir / f"{dataset_name}_case_summary.csv", index=False)
    return out_df, {
        "dataset": dataset_name,
        "manifest_csv": str(manifest_csv),
        "manifest_rows": n_available,
        "selected_rows": n_selected,
        "processed_rows": int(len(out_df)),
        "max_cases": int(max_cases),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage0-4 only: deterministic token bank + router + verifier.")
    parser.add_argument("--ctrate_csv", type=str, required=True)
    parser.add_argument("--radgenome_csv", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="outputs_stage0_4")
    parser.add_argument("--build_mini", action="store_true", help="Build 450/450 mini subsets first.")
    parser.add_argument("--volume_col", type=str, default="volume_path")
    parser.add_argument("--report_col", type=str, default="report_text")
    parser.add_argument("--case_id_col", type=str, default="case_id")
    parser.add_argument("--max_cases", type=int, default=450)
    parser.add_argument("--shuffle_seed", type=int, default=None,
                        help="If set, shuffle manifest with this seed before taking head(max_cases)")
    parser.add_argument("--encoder_ckpt", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--resize_d", type=int, default=128)
    parser.add_argument("--resize_h", type=int, default=128)
    parser.add_argument("--resize_w", type=int, default=128)
    parser.add_argument("--token_budget_b", type=int, default=64)
    parser.add_argument("--k_per_sentence", type=int, default=8)
    parser.add_argument("--lambda_spatial", type=float, default=0.3)
    parser.add_argument("--tau_iou", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument(
        "--r2_mode",
        type=str,
        default="auto",
        choices=("auto", "ratio", "max_iou"),
        help="Verifier R2 mode. auto follows current defaults/cp_strict behavior.",
    )
    parser.add_argument(
        "--r2_min_support_ratio",
        type=float,
        default=None,
        help="R2 support ratio threshold in ratio mode (recommended scan: 1.0/0.8/0.6).",
    )
    parser.add_argument("--text_encoder", type=str, default="hash", help="hash | semantic")
    parser.add_argument(
        "--text_encoder_model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence-transformers model name when text_encoder=semantic.",
    )
    parser.add_argument(
        "--text_encoder_hash_dim",
        type=int,
        default=256,
        help="Embedding dim when text_encoder=hash.",
    )
    parser.add_argument(
        "--text_encoder_device",
        type=str,
        default="cpu",
        help="Device for semantic text encoder (cpu/cuda).",
    )
    parser.add_argument(
        "--cp_strict",
        action="store_true",
        help="Enable CP strict checks (requires semantic text encoder + SwinUNETR checkpoint).",
    )
    parser.add_argument(
        "--expected_cases_per_dataset",
        type=int,
        default=0,
        help="If >0, require selected case count per dataset to equal this value (e.g. 450).",
    )
    parser.add_argument("--r4_disabled", action="store_true", help="Disable R4_SIZE check (union bbox threshold uncalibrated).")
    parser.add_argument("--r5_fallback_disabled", action="store_true", help="Disable R5 negation fallback lexicon.")
    parser.add_argument("--r2_skip_bilateral", action="store_true", help="Skip R2_ANATOMY for 'bilateral' sentences (bbox=entire volume, structurally cannot pass IoU check).")
    parser.add_argument("--r1_negation_exempt", action="store_true", help="Skip R1_LATERALITY for negated sentences (e.g. 'no left pleural effusion'). Negated cites are not expected to be laterally aligned.")
    parser.add_argument("--r1_skip_midline", action="store_true", help="Skip R1_LATERALITY for midline anatomy keywords (mediastinum, trachea, aorta, esophagus, spine, etc.) that span the midline by definition.")
    parser.add_argument("--r1_min_same_side_ratio", type=float, default=None, help="R1 fires only when fraction of non-cross tokens on the claimed side < this threshold. Default 1.0 (strict all-or-nothing). 0.6 recommended for ratio mode.")
    parser.add_argument("--lateral_tolerance", type=float, default=None, help="Half-width of the midline dead zone (normalized x). Tokens within x_mid ± tol are classified as 'cross' and excluded from R1 laterality check. 0.0 = strict midline (default). Try 0.05 to absorb near-midline tokens.")
    parser.add_argument(
        "--w_proj_path",
        type=str,
        default=None,
        help="Path to trained W_proj .pt file. If set, uses learned projection instead of identity.",
    )
    parser.add_argument(
        "--anatomy_spatial_routing",
        action="store_true",
        help=(
            "Route primarily by anatomy bbox IoU instead of cross-modal dot product. "
            "Recommended when w_proj is untrained (identity). Semantic score used as tiebreaker only."
        ),
    )
    # Stage 5: LLM judge
    parser.add_argument(
        "--llm_judge",
        type=str,
        default=None,
        choices=("ollama", "openai", "anthropic", "huggingface"),
        help="Enable Stage-5 LLM judge. Confirms/dismisses Stage-4 violations and applies score penalty.",
    )
    parser.add_argument(
        "--llm_judge_model",
        type=str,
        default=None,
        help=(
            "LLM model name for Stage-5 judge. "
            "Defaults: ollama='qwen2.5:7b', openai='gpt-4o-mini', "
            "anthropic='claude-haiku-4-5-20251001', huggingface='meta-llama/Llama-3.1-8B-Instruct'."
        ),
    )
    parser.add_argument(
        "--llm_judge_alpha",
        type=float,
        default=0.5,
        help="CP .tex penalty scale for Stage-5: S'=S*(1-alpha*sev). Default 0.5.",
    )
    parser.add_argument(
        "--llm_judge_ollama_host",
        type=str,
        default="http://localhost:11434",
        help="Ollama API host for Stage-5 judge.",
    )
    parser.add_argument(
        "--llm_judge_fail_open",
        action="store_true",
        default=True,
        help="On LLM call failure, keep original violation (conservative). Default True.",
    )
    parser.add_argument(
        "--llm_judge_hf_device_map",
        type=str,
        default="auto",
        help="HuggingFace device_map for Stage-5 judge (auto/cpu/cuda). Default: auto.",
    )
    parser.add_argument(
        "--llm_judge_hf_torch_dtype",
        type=str,
        default="bfloat16",
        choices=("bfloat16", "float16", "float32"),
        help="HuggingFace torch dtype for Stage-5 judge. Default: bfloat16.",
    )
    parser.add_argument(
        "--llm_judge_hf_token",
        type=str,
        default=None,
        help="HuggingFace access token for gated models (e.g. Llama-3). Falls back to HF_TOKEN env var.",
    )
    # Stage 3c: Token-gated LLM generation
    parser.add_argument(
        "--stage3c_backend",
        type=str,
        default=None,
        choices=("ollama", "openai", "anthropic", "huggingface"),
        help="Enable Stage 3c LLM generation. If not set, pipeline uses original topic as text.",
    )
    parser.add_argument(
        "--stage3c_model",
        type=str,
        default=None,
        help="LLM model for Stage 3c generation. Defaults to llm_judge_model if same backend.",
    )
    parser.add_argument(
        "--stage3c_temperature",
        type=float,
        default=0.3,
        help="Temperature for Stage 3c generation. Default 0.3.",
    )
    parser.add_argument(
        "--stage3c_max_tokens",
        type=int,
        default=256,
        help="Max tokens for Stage 3c generation. Default 256.",
    )
    # Reroute config
    parser.add_argument(
        "--reroute_gamma",
        type=float,
        default=2.0,
        help="Log-smooth penalty gamma: r' = r - gamma * ln(1 + sev). Default 2.0.",
    )
    parser.add_argument(
        "--reroute_max_retry",
        type=int,
        default=1,
        help="Max regeneration retries after rerouting. Default 1.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    resize_dhw = (args.resize_d, args.resize_h, args.resize_w)

    ctrate_csv = args.ctrate_csv
    radgenome_csv = args.radgenome_csv
    if args.build_mini:
        mini_paths = build_ctrate_radgenome_minis(
            ctrate_csv=ctrate_csv,
            radgenome_csv=radgenome_csv,
            out_dir=str(out_dir / "mini_manifests"),
            seed=42,
            ctrate_strata=("split",) if "split" in pd.read_csv(ctrate_csv, nrows=1).columns else (),
            radgenome_strata=("split",) if "split" in pd.read_csv(radgenome_csv, nrows=1).columns else (),
        )
        ctrate_csv, radgenome_csv = mini_paths

    cfg = ProveTokConfig()
    cfg.split.token_budget_b = int(args.token_budget_b)
    cfg.split.beta = float(args.beta)
    cfg.router.k_per_sentence = int(args.k_per_sentence)
    cfg.router.lambda_spatial = float(args.lambda_spatial)
    cfg.verifier.tau_anatomy_iou = float(args.tau_iou)
    if args.r2_mode == "ratio":
        cfg.verifier.use_max_iou_for_r2 = False
    elif args.r2_mode == "max_iou":
        cfg.verifier.use_max_iou_for_r2 = True
    if args.r2_min_support_ratio is not None:
        r2_ratio = float(args.r2_min_support_ratio)
        if not (0.0 < r2_ratio <= 1.0):
            raise ValueError(f"--r2_min_support_ratio must be in (0, 1], got {r2_ratio}")
        cfg.verifier.r2_min_support_ratio = r2_ratio
    if args.r4_disabled:
        cfg.verifier.r4_disabled = True
    if args.r5_fallback_disabled:
        cfg.verifier.r5_fallback_lexicon = False
    if args.r2_skip_bilateral:
        cfg.verifier.r2_skip_keywords = {"bilateral"}
    # "left lung" / "right lung" are fallback routing keywords (entire half-volume bbox).
    # Their IoU with small token bboxes is structurally too low for R2 to be meaningful.
    cfg.verifier.r2_skip_keywords.update({"left lung", "right lung"})
    if args.r1_negation_exempt:
        cfg.verifier.r1_negation_exempt = True
    if args.r1_skip_midline:
        cfg.verifier.r1_skip_midline_keywords = {
            "mediastinum", "trachea", "carina", "esophagus",
            "aorta", "spine", "vertebra", "sternum",
        }
    if args.r1_min_same_side_ratio is not None:
        ratio = float(args.r1_min_same_side_ratio)
        if not (0.0 <= ratio <= 1.0):
            raise ValueError(f"--r1_min_same_side_ratio must be in [0, 1], got {ratio}")
        cfg.verifier.r1_min_same_side_ratio = ratio
    if args.anatomy_spatial_routing:
        cfg.router.anatomy_spatial_routing = True
    if args.lateral_tolerance is not None:
        tol = float(args.lateral_tolerance)
        if not (0.0 <= tol <= 0.5):
            raise ValueError(f"--lateral_tolerance must be in [0, 0.5], got {tol}")
        cfg.verifier.lateral_tolerance = tol

    text_encoder_mode = str(args.text_encoder).strip().lower()
    if args.cp_strict:
        if not args.encoder_ckpt:
            raise ValueError("CP strict mode requires --encoder_ckpt for SwinUNETR.")
        if text_encoder_mode in ("hash", "deterministic"):
            text_encoder_mode = "semantic"
            print("[CP strict] text_encoder=hash is overridden to semantic.")
        if args.r2_mode == "auto":
            cfg.verifier.use_max_iou_for_r2 = False
        if args.r2_mode != "max_iou" and args.r2_min_support_ratio is None:
            cfg.verifier.r2_min_support_ratio = 1.0

    text_encoder = make_text_encoder(
        encoder_type=text_encoder_mode,
        hash_dim=int(args.text_encoder_hash_dim),
        st_model_name=args.text_encoder_model,
        st_device=args.text_encoder_device,
    )

    # Load trained W_proj if provided
    _w_proj_matrix = None
    if args.w_proj_path:
        import torch
        _w = torch.load(args.w_proj_path, weights_only=True)
        _w_proj_matrix = _w.tolist()  # Convert to List[List[float]] for Router
        print(f"[W_proj] Loaded trained projection from {args.w_proj_path} (shape {list(_w.shape)})")

    # Common paths
    _PROJECT_ROOT = Path(__file__).parent
    _LOCAL_LLAMA = _PROJECT_ROOT / "models" / "Llama-3.1-8B-Instruct"

    # Stage 5 LLM judge setup
    llm_judge: Optional[LLMJudge] = None
    if args.llm_judge:
        import os
        from ProveTok_Main_experiment.stage5_llm_judge import LLMJudgeConfig

        _default_models = {
            "ollama": "qwen2.5:7b",
            "openai": "gpt-4o-mini",
            "anthropic": "claude-haiku-4-5-20251001",
            "huggingface": str(_LOCAL_LLAMA),  # local path, no HF download needed
        }
        _model = args.llm_judge_model or _default_models[args.llm_judge]

        # For huggingface backend: if a relative path is given, resolve from project root
        if args.llm_judge == "huggingface" and not os.path.isabs(_model) and not _model.startswith("meta-llama/"):
            _model = str(_PROJECT_ROOT / _model)

        _hf_token = args.llm_judge_hf_token or os.environ.get("HF_TOKEN")
        _judge_cfg = LLMJudgeConfig(
            backend=args.llm_judge,
            model=_model,
            alpha=float(args.llm_judge_alpha),
            ollama_host=args.llm_judge_ollama_host,
            fail_open=bool(args.llm_judge_fail_open),
            hf_device_map=args.llm_judge_hf_device_map,
            hf_torch_dtype=args.llm_judge_hf_torch_dtype,
            hf_token=_hf_token,
        )
        llm_judge = LLMJudge(_judge_cfg)
        print(f"[Stage 5] LLM judge enabled: backend={args.llm_judge}, model={_model}, alpha={args.llm_judge_alpha}")

    # Reroute config
    cfg.reroute.gamma_penalty = float(args.reroute_gamma)
    cfg.reroute.max_retry = int(args.reroute_max_retry)

    # Stage 3c generator setup
    generator: Optional[Stage3cGenerator] = None
    if args.stage3c_backend:
        import os as _os

        _s3c_model = args.stage3c_model or (args.llm_judge_model if args.llm_judge else None)
        if _s3c_model is None:
            raise ValueError("--stage3c_model is required when --stage3c_backend is set (or set --llm_judge_model).")

        # Resolve relative paths for huggingface backend
        if args.stage3c_backend == "huggingface" and not _os.path.isabs(_s3c_model) and not _s3c_model.startswith("meta-llama/"):
            _s3c_model = str(_PROJECT_ROOT / _s3c_model)

        _gen_cfg = GeneratorConfig(
            backend=args.stage3c_backend,
            model=_s3c_model,
            temperature=float(args.stage3c_temperature),
            max_tokens=int(args.stage3c_max_tokens),
            hf_torch_dtype=args.llm_judge_hf_torch_dtype,
        )

        # LLM sharing: if same model + huggingface backend, reuse the HF pipeline
        _share_pipe = (
            args.stage3c_backend == "huggingface"
            and llm_judge is not None
            and args.llm_judge == "huggingface"
            and llm_judge._hf_pipe is not None
            and _s3c_model == (args.llm_judge_model or str(_LOCAL_LLAMA))
        )
        if _share_pipe:
            # Skip HF init by temporarily setting backend, then restore + inject pipe
            _gen_cfg_tmp = GeneratorConfig(
                backend="ollama",  # no-op init
                model=_s3c_model,
                temperature=_gen_cfg.temperature,
                max_tokens=_gen_cfg.max_tokens,
            )
            generator = Stage3cGenerator(_gen_cfg_tmp)
            generator.cfg = _gen_cfg  # restore real config
            generator._hf_pipe = llm_judge._hf_pipe
            print(f"[Stage 3c] Sharing HF pipeline with Stage 5 judge: {_s3c_model}")
        else:
            generator = Stage3cGenerator(_gen_cfg)
            print(f"[Stage 3c] Generator enabled: backend={args.stage3c_backend}, model={_s3c_model}")

    ct_df, ct_meta = _run_manifest(
        dataset_name="ctrate",
        manifest_csv=ctrate_csv,
        out_dir=out_dir,
        cfg=cfg,
        volume_col=args.volume_col,
        report_col=args.report_col,
        case_id_col=args.case_id_col,
        max_cases=args.max_cases,
        encoder_ckpt=args.encoder_ckpt,
        device=args.device,
        resize_dhw=resize_dhw,
        text_encoder=text_encoder,
        expected_cases=int(args.expected_cases_per_dataset),
        llm_judge=llm_judge,
        generator=generator,
        shuffle_seed=args.shuffle_seed,
        w_proj=_w_proj_matrix,
    )
    rg_df, rg_meta = _run_manifest(
        dataset_name="radgenome",
        manifest_csv=radgenome_csv,
        out_dir=out_dir,
        cfg=cfg,
        volume_col=args.volume_col,
        report_col=args.report_col,
        case_id_col=args.case_id_col,
        max_cases=args.max_cases,
        encoder_ckpt=args.encoder_ckpt,
        device=args.device,
        resize_dhw=resize_dhw,
        text_encoder=text_encoder,
        expected_cases=int(args.expected_cases_per_dataset),
        llm_judge=llm_judge,
        generator=generator,
        shuffle_seed=args.shuffle_seed,
        w_proj=_w_proj_matrix,
    )
    summary = pd.concat([ct_df, rg_df], axis=0, ignore_index=True)
    summary.to_csv(out_dir / "summary.csv", index=False)
    run_meta = {
        "build_mini": bool(args.build_mini),
        "ctrate": ct_meta,
        "radgenome": rg_meta,
        "token_budget_b": int(cfg.split.token_budget_b),
        "k_per_sentence": int(cfg.router.k_per_sentence),
        "lambda_spatial": float(cfg.router.lambda_spatial),
        "tau_iou": float(cfg.verifier.tau_anatomy_iou),
        "beta": float(cfg.split.beta),
        "device": str(args.device),
        "resize_dhw": [int(x) for x in resize_dhw],
        "cp_strict": bool(args.cp_strict),
        "text_encoder": str(text_encoder_mode),
        "text_encoder_model": str(args.text_encoder_model),
        "text_encoder_device": str(args.text_encoder_device),
        "encoder_ckpt": str(args.encoder_ckpt) if args.encoder_ckpt else None,
        "r2_mode": "max_iou" if cfg.verifier.use_max_iou_for_r2 else "ratio",
        "r2_min_support_ratio": float(cfg.verifier.r2_min_support_ratio),
        "r5_fallback_lexicon": bool(cfg.verifier.r5_fallback_lexicon),
        "r4_disabled": bool(args.r4_disabled),
        "r5_fallback_disabled": bool(args.r5_fallback_disabled),
        "r2_skip_bilateral": bool(args.r2_skip_bilateral),
        "r1_negation_exempt": bool(args.r1_negation_exempt),
        "r1_skip_midline": bool(args.r1_skip_midline),
        "r1_min_same_side_ratio": float(cfg.verifier.r1_min_same_side_ratio),
        "lateral_tolerance": float(cfg.verifier.lateral_tolerance),
        "w_proj_path": str(args.w_proj_path) if args.w_proj_path else None,
        "anatomy_spatial_routing": bool(cfg.router.anatomy_spatial_routing),
        "llm_judge_backend": str(args.llm_judge) if args.llm_judge else None,
        "llm_judge_model": str(args.llm_judge_model) if args.llm_judge else None,
        "llm_judge_alpha": float(args.llm_judge_alpha) if args.llm_judge else None,
        "stage3c_backend": str(args.stage3c_backend) if args.stage3c_backend else None,
        "stage3c_model": str(args.stage3c_model) if args.stage3c_backend else None,
        "stage3c_temperature": float(args.stage3c_temperature) if args.stage3c_backend else None,
        "reroute_gamma": float(cfg.reroute.gamma_penalty),
        "reroute_max_retry": int(cfg.reroute.max_retry),
    }
    with (out_dir / "run_meta.json").open("w", encoding="utf-8") as f:
        json.dump(run_meta, f, ensure_ascii=False, indent=2)
    print(summary.groupby("dataset", as_index=False).agg(cases=("case_id", "count"), violations=("n_violations", "sum")))
    print(f"Saved summary: {out_dir / 'summary.csv'}")
    print(f"Saved run meta: {out_dir / 'run_meta.json'}")


if __name__ == "__main__":
    main()
