import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from ProveTok_main_exp.config import ProveTokConfig
from ProveTok_main_exp.dataset_tools import build_ctrate_radgenome_minis
from ProveTok_main_exp.preprocess import (
    ct_intensity_normalize,
    load_volume_with_meta,
    resize_volume,
    resampled_spacing_xyz_mm,
)
from ProveTok_main_exp.simple_modules import ReportSentencePlanner, RuleBasedAnatomyResolver
from ProveTok_main_exp.stage0_scorer import DeterministicArtifactScorer
from ProveTok_main_exp.stage1_swinunetr_encoder import FrozenSwinUNETREncoder
from ProveTok_main_exp.stage2_octree_splitter import AdaptiveOctreeSplitter
from ProveTok_main_exp.stage3_router import Router
from ProveTok_main_exp.stage4_verifier import Verifier
from ProveTok_main_exp.stage0_4_runner import run_case_stage0_4, Stage04Components
from ProveTok_main_exp.text_encoder import DeterministicTextEncoder


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
) -> pd.DataFrame:
    df = pd.read_csv(manifest_csv).head(max_cases).reset_index(drop=True)
    if volume_col not in df.columns:
        raise ValueError(f"Missing column '{volume_col}' in {manifest_csv}")
    if report_col not in df.columns:
        raise ValueError(f"Missing column '{report_col}' in {manifest_csv}")
    if case_id_col not in df.columns:
        df[case_id_col] = [f"{dataset_name}_{i:05d}" for i in range(len(df))]

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
        router=Router(cfg=cfg.router, text_encoder=DeterministicTextEncoder(dim=256), w_proj=None),
        verifier=Verifier(cfg.verifier, RuleBasedAnatomyResolver()),
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
                "trace_jsonl": result["trace_jsonl"],
            }
        )
    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_dir / f"{dataset_name}_case_summary.csv", index=False)
    return out_df


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

    ct_df = _run_manifest(
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
    )
    rg_df = _run_manifest(
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
    )
    summary = pd.concat([ct_df, rg_df], axis=0, ignore_index=True)
    summary.to_csv(out_dir / "summary.csv", index=False)
    print(summary.groupby("dataset", as_index=False).agg(cases=("case_id", "count"), violations=("n_violations", "sum")))
    print(f"Saved summary: {out_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
