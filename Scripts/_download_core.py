#!/usr/bin/env python3
"""
_download_core.py — Shared logic for downloading NIfTI (.nii.gz) volumes from HuggingFace.
_download_core.py — 从 HuggingFace 下载 NIfTI (.nii.gz) 文件的共享核心逻辑。

This module is NOT meant to be run directly. Use:
本模块不应直接运行，请使用：
  - download_smoke_450.py   (smoke test, ~450 samples / 小样本)
  - download_full_3000.py   (full download, ~3000 samples / 全量)
"""
import argparse
import csv
import os
import re
import sys
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from huggingface_hub import hf_hub_download, HfApi

# ==========================================
# 1. Logging / 日志系统
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ==========================================
# 2. Data Model / 数据模型
# ==========================================
@dataclass
class Job:
    name: str
    repo_id: str
    csv_in: Path
    csv_clean: Path
    manifest_out: Path
    image_out_dir: Path
    volume_col: str
    preferred_prefixes: List[str]


# ==========================================
# 3. Core Logic / 核心处理逻辑
# ==========================================
def normalize_case_id(volume_name: str) -> str:
    return re.sub(r"_[a-z]_\d+\.nii\.gz$", "", volume_name)


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def clean_rows(
    rows: List[Dict[str, str]], volume_col: str
) -> Tuple[List[Dict[str, str]], List[str]]:
    seen = set()
    cleaned: List[Dict[str, str]] = []
    ordered_vols: List[str] = []

    for row in rows:
        if volume_col not in row:
            continue
        vol = (row.get(volume_col) or "").strip()
        if not vol:
            continue
        if vol in seen:
            continue
        seen.add(vol)

        out = {
            k.strip(): (v.strip() if isinstance(v, str) else v)
            for k, v in row.items()
        }
        out[volume_col] = vol
        out["CaseID"] = normalize_case_id(vol)
        out["Split"] = (
            "train"
            if vol.startswith("train_")
            else ("valid" if vol.startswith("valid_") else "unknown")
        )
        cleaned.append(out)
        ordered_vols.append(vol)

    return cleaned, ordered_vols


def write_csv(
    path: Path, rows: List[Dict[str, str]], front_fields: List[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"No rows to write / 无可写入的行: {path}")

    existing_fields = list(rows[0].keys())
    fields = []
    for f in front_fields:
        if f in existing_fields and f not in fields:
            fields.append(f)
    for f in existing_fields:
        if f not in fields:
            fields.append(f)

    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def pick_best_remote_path(
    paths: List[str], preferred_prefixes: List[str]
) -> str:
    for prefix in preferred_prefixes:
        for p in paths:
            if p.startswith(prefix):
                return p
    return paths[0]


def build_remote_map(
    repo_id: str, needed_names: List[str], preferred_prefixes: List[str]
) -> Dict[str, str]:
    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    logger.info(
        f"Listing remote repo files / 正在列出远程仓库文件: {repo_id} ..."
    )
    files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")

    needed = set(needed_names)
    by_name: Dict[str, List[str]] = {}
    for p in files:
        if not p.endswith(".nii.gz"):
            continue
        name = Path(p).name
        if name in needed:
            by_name.setdefault(name, []).append(p)

    matched = len(by_name)
    total = len(needed)
    logger.info(
        f"Remote match / 远程匹配: {matched}/{total} .nii.gz volumes found / 找到"
    )

    out: Dict[str, str] = {}
    for name, paths in by_name.items():
        out[name] = pick_best_remote_path(paths, preferred_prefixes)
    return out


def download_selected(
    job: Job,
    cleaned_rows: List[Dict[str, str]],
    remote_map: Dict[str, str],
    log_interval: int = 50,
) -> List[Dict[str, str]]:
    """Download .nii.gz files directly to target dir (no extra cache copy).
    直接下载 .nii.gz 到目标目录（无额外缓存副本，避免双倍磁盘占用）。
    """
    manifest: List[Dict[str, str]] = []
    job.image_out_dir.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN")

    total = len(cleaned_rows)
    ok_count = 0
    skip_count = 0
    miss_count = 0
    err_count = 0

    for idx, row in enumerate(cleaned_rows, 1):
        vol = row[job.volume_col]
        split = row.get("Split", "unknown")
        remote = remote_map.get(vol, "")
        local_dir = job.image_out_dir / split
        local_dir.mkdir(parents=True, exist_ok=True)
        local_file = local_dir / vol

        status = "ok"
        err = ""
        if not remote:
            status = "missing_remote"
            miss_count += 1
        else:
            try:
                if local_file.exists():
                    status = "ok_existing"
                    skip_count += 1
                else:
                    # Download directly into target dir — no cache copy needed
                    # 直接下载到目标目录 — 无需缓存复制，避免双倍磁盘占用
                    hf_hub_download(
                        repo_id=job.repo_id,
                        repo_type="dataset",
                        filename=remote,
                        token=token,
                        local_dir=str(local_dir),
                        local_dir_use_symlinks=False,
                    )
                    # hf_hub_download with local_dir preserves the remote subpath,
                    # so the file lands at local_dir/<remote_path>.
                    # We need to move it to local_dir/<vol>.
                    downloaded_path = local_dir / remote
                    if downloaded_path.exists() and downloaded_path != local_file:
                        downloaded_path.rename(local_file)
                        # Clean up empty parent dirs left behind
                        # 清理残留的空目录
                        try:
                            for parent in downloaded_path.parents:
                                if parent == local_dir:
                                    break
                                parent.rmdir()  # only removes if empty
                        except OSError:
                            pass
                    ok_count += 1
            except Exception as e:  # noqa: BLE001
                status = "download_error"
                err = str(e)
                err_count += 1
                logger.error(
                    f"  [{idx}/{total}] Download failed / 下载失败: {vol} — {e}"
                )

        manifest.append(
            {
                "dataset": job.name,
                "volume_name": vol,
                "split": split,
                "remote_path": remote,
                "local_path": str(local_file) if status in {"ok", "ok_existing"} else "",
                "status": status,
                "error": err,
            }
        )

        # Progress log at interval / 按间隔打印进度
        if idx % log_interval == 0 or idx == total:
            logger.info(
                f"  [{idx}/{total}] Progress / 进度: "
                f"new={ok_count} skipped={skip_count} missing={miss_count} errors={err_count}"
            )

    return manifest


# ==========================================
# 4. Job Runner / 任务执行器
# ==========================================
def run_job(job: Job, log_interval: int = 50) -> None:
    logger.info(f"========== [{job.name}] Starting job / 开始处理任务 ==========")

    rows = read_csv_rows(job.csv_in)
    logger.info(
        f"[{job.name}] Read {len(rows)} rows from input CSV / "
        f"从输入 CSV 读取了 {len(rows)} 行"
    )

    cleaned_rows, needed = clean_rows(rows, job.volume_col)
    logger.info(
        f"[{job.name}] After dedup & clean: {len(cleaned_rows)} unique volumes / "
        f"去重清洗后: {len(cleaned_rows)} 个唯一 volume"
    )
    write_csv(job.csv_clean, cleaned_rows, [job.volume_col, "CaseID", "Split"])
    logger.info(
        f"[{job.name}] Cleaned CSV saved / 清洗后 CSV 已保存: {job.csv_clean}"
    )

    remote_map = build_remote_map(job.repo_id, needed, job.preferred_prefixes)

    logger.info(
        f"[{job.name}] Begin downloading .nii.gz volumes / 开始下载 .nii.gz 文件 ..."
    )
    manifest = download_selected(job, cleaned_rows, remote_map, log_interval=log_interval)
    write_csv(
        job.manifest_out, manifest, ["dataset", "volume_name", "split", "status"]
    )

    ok = sum(1 for x in manifest if x["status"] in {"ok", "ok_existing"})
    ok_existing = sum(1 for x in manifest if x["status"] == "ok_existing")
    miss = sum(1 for x in manifest if x["status"] == "missing_remote")
    errs = sum(1 for x in manifest if x["status"] == "download_error")

    logger.info(
        f"[{job.name}] Summary / 汇总: "
        f"cleaned={len(cleaned_rows)}  ok={ok} (existing={ok_existing})  "
        f"missing={miss}  errors={errs}"
    )
    if miss > 0:
        logger.warning(
            f"[{job.name}] {miss} volumes missing on remote! "
            f"Check manifest for details / {miss} 个文件在远程仓库中未找到！"
            f"请查看清单定位具体缺失项: {job.manifest_out}"
        )
    if errs > 0:
        logger.warning(
            f"[{job.name}] {errs} download errors! "
            f"Re-run to retry / {errs} 个下载失败！"
            f"重新运行脚本即可自动重试: {job.manifest_out}"
        )
    logger.info(f"[{job.name}] Manifest saved / 下载清单已保存: {job.manifest_out}")
    logger.info(
        f"[{job.name}] NIfTI volumes saved to / .nii.gz 文件保存至: {job.image_out_dir}"
    )
    logger.info(f"========== [{job.name}] Job finished / 任务完成 ==========")


# ==========================================
# 5. Entrypoint builder / 入口构建器
# ==========================================
def build_parser(script_label: str, default_log_interval: int) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            f"[{script_label}] Download NIfTI (.nii.gz) volumes from HuggingFace. "
            f"[{script_label}] 从 HuggingFace 下载 NIfTI (.nii.gz) 文件。"
        )
    )
    parser.add_argument(
        "--jobs",
        default="ct,rad",
        help="Comma-separated jobs to run / 要运行的任务: ct, rad (default: ct,rad)",
    )
    parser.add_argument(
        "--ct_csv",
        type=str,
        help="Path to CT-RATE input CSV / CT-RATE 输入 CSV 路径",
    )
    parser.add_argument(
        "--rad_csv",
        type=str,
        help="Path to RadGenome input CSV / RadGenome 输入 CSV 路径",
    )
    parser.add_argument(
        "--out_dir",
        default=None,
        type=str,
        help="Override output directory / 覆盖输出目录 (default: ../ProveTok_data_download)",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=default_log_interval,
        help=f"Print progress every N files / 每 N 个文件打印进度 (default: {default_log_interval})",
    )
    return parser


def run_main(script_label: str, default_log_interval: int) -> None:
    """Shared main() logic called by both entry scripts.
    两个入口脚本共用的 main() 逻辑。
    """
    parser = build_parser(script_label, default_log_interval)
    args = parser.parse_args()

    # --- HF_TOKEN check / 令牌检查 ---
    if not os.environ.get("HF_TOKEN"):
        logger.error(
            "HF_TOKEN not set! Please run: export HF_TOKEN='your_token' / "
            "未检测到 HF_TOKEN 环境变量！请先执行: export HF_TOKEN='你的token'"
        )
        sys.exit(1)

    # --- Resolve output dir / 解析输出目录 ---
    # Default: ProveTok_ACM/ProveTok_data_download (one level up from Scripts/)
    script_dir = Path(__file__).resolve().parent
    if args.out_dir:
        out_base = Path(args.out_dir).resolve()
    else:
        out_base = (script_dir.parent / "ProveTok_data_download").resolve()

    selected = {x.strip().lower() for x in args.jobs.split(",") if x.strip()}
    jobs = []

    if "ct" in selected:
        if not args.ct_csv:
            logger.error(
                "Job 'ct' enabled but --ct_csv not provided! / "
                "启用了 'ct' 任务，但未提供 --ct_csv 参数！"
            )
            sys.exit(1)

        ct_in = Path(args.ct_csv).resolve()
        if not ct_in.exists():
            logger.error(
                f"CT-RATE input CSV not found / 找不到 CT-RATE 输入文件: {ct_in}"
            )
            sys.exit(1)

        jobs.append(
            Job(
                name="CT-RATE",
                repo_id="ibrahimhamamci/CT-RATE",
                csv_in=ct_in,
                csv_clean=ct_in.with_name(f"{ct_in.stem}_clean.csv"),
                manifest_out=ct_in.with_name(f"{ct_in.stem}_manifest.csv"),
                image_out_dir=out_base / "CT-RATE",
                volume_col="VolumeName",
                preferred_prefixes=[
                    "dataset/train_fixed/",
                    "dataset/valid_fixed/",
                    "dataset/train/",
                    "dataset/valid/",
                ],
            )
        )

    if "rad" in selected:
        if not args.rad_csv:
            logger.error(
                "Job 'rad' enabled but --rad_csv not provided! / "
                "启用了 'rad' 任务，但未提供 --rad_csv 参数！"
            )
            sys.exit(1)

        rad_in = Path(args.rad_csv).resolve()
        if not rad_in.exists():
            logger.error(
                f"RadGenome input CSV not found / 找不到 RadGenome 输入文件: {rad_in}"
            )
            sys.exit(1)

        jobs.append(
            Job(
                name="RadGenome-ChestCT",
                repo_id="RadGenome/RadGenome-ChestCT",
                csv_in=rad_in,
                csv_clean=rad_in.with_name(f"{rad_in.stem}_clean.csv"),
                manifest_out=rad_in.with_name(f"{rad_in.stem}_manifest.csv"),
                image_out_dir=out_base / "RadGenome-ChestCT",
                volume_col="Volumename",
                preferred_prefixes=[
                    "dataset/train_preprocessed/",
                    "dataset/valid_preprocessed/",
                ],
            )
        )

    if not jobs:
        logger.warning("No jobs to run / 没有需要执行的任务。")
        sys.exit(0)

    logger.info(f"[{script_label}] Jobs / 任务: {[j.name for j in jobs]}")
    logger.info(f"[{script_label}] Output dir / 输出目录: {out_base}")
    logger.info(
        f"[{script_label}] Log interval / 日志间隔: every {args.log_interval} files / "
        f"每 {args.log_interval} 个文件"
    )

    for job in jobs:
        run_job(job, log_interval=args.log_interval)

    logger.info(f"[{script_label}] All jobs completed / 所有任务执行完毕。")
