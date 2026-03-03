#!/usr/bin/env python3
import argparse
import csv
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from huggingface_hub import HfApi, hf_hub_download


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


def normalize_case_id(volume_name: str) -> str:
    return re.sub(r"_[a-z]_\d+\.nii\.gz$", "", volume_name)


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def clean_rows(rows: List[Dict[str, str]], volume_col: str) -> Tuple[List[Dict[str, str]], List[str]]:
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

        out = {k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in row.items()}
        out[volume_col] = vol
        out["CaseID"] = normalize_case_id(vol)
        out["Split"] = "train" if vol.startswith("train_") else ("valid" if vol.startswith("valid_") else "unknown")
        cleaned.append(out)
        ordered_vols.append(vol)

    return cleaned, ordered_vols


def write_csv(path: Path, rows: List[Dict[str, str]], front_fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"No rows to write: {path}")

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


def pick_best_remote_path(paths: List[str], preferred_prefixes: List[str]) -> str:
    for prefix in preferred_prefixes:
        for p in paths:
            if p.startswith(prefix):
                return p
    return paths[0]


def build_remote_map(repo_id: str, needed_names: List[str], preferred_prefixes: List[str]) -> Dict[str, str]:
    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")

    needed = set(needed_names)
    by_name: Dict[str, List[str]] = {}
    for p in files:
        if not p.endswith(".nii.gz"):
            continue
        name = Path(p).name
        if name in needed:
            by_name.setdefault(name, []).append(p)

    out: Dict[str, str] = {}
    for name, paths in by_name.items():
        out[name] = pick_best_remote_path(paths, preferred_prefixes)
    return out


def download_selected(job: Job, cleaned_rows: List[Dict[str, str]], remote_map: Dict[str, str]) -> List[Dict[str, str]]:
    manifest: List[Dict[str, str]] = []
    job.image_out_dir.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN")

    for row in cleaned_rows:
        vol = row[job.volume_col]
        split = row.get("Split", "unknown")
        remote = remote_map.get(vol, "")
        local_file = job.image_out_dir / split / vol
        local_file.parent.mkdir(parents=True, exist_ok=True)

        status = "ok"
        err = ""
        if not remote:
            status = "missing_remote"
        else:
            try:
                if local_file.exists():
                    status = "ok_existing"
                else:
                    cached = hf_hub_download(
                        repo_id=job.repo_id,
                        repo_type="dataset",
                        filename=remote,
                        token=token,
                    )
                    shutil.copy2(cached, local_file)
            except Exception as e:  # noqa: BLE001
                status = "download_error"
                err = str(e)

        manifest.append(
            {
                "dataset": job.name,
                "volume_name": vol,
                "split": split,
                "remote_path": remote,
                "local_path": str(local_file) if status == "ok" else "",
                "status": status,
                "error": err,
            }
        )
    return manifest


def run_job(job: Job) -> None:
    rows = read_csv_rows(job.csv_in)
    cleaned_rows, needed = clean_rows(rows, job.volume_col)
    write_csv(job.csv_clean, cleaned_rows, [job.volume_col, "CaseID", "Split"])

    remote_map = build_remote_map(job.repo_id, needed, job.preferred_prefixes)
    manifest = download_selected(job, cleaned_rows, remote_map)
    write_csv(job.manifest_out, manifest, ["dataset", "volume_name", "split", "status"])

    ok = sum(1 for x in manifest if x["status"] in {"ok", "ok_existing"})
    ok_existing = sum(1 for x in manifest if x["status"] == "ok_existing")
    miss = sum(1 for x in manifest if x["status"] == "missing_remote")
    err = sum(1 for x in manifest if x["status"] == "download_error")
    print(
        f"[{job.name}] cleaned={len(cleaned_rows)} downloaded_ok_total={ok} "
        f"(existing={ok_existing}) missing_remote={miss} download_error={err}"
    )
    print(f"[{job.name}] cleaned_csv={job.csv_clean}")
    print(f"[{job.name}] manifest={job.manifest_out}")
    print(f"[{job.name}] image_dir={job.image_out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean smoke CSVs and download selected NIfTI files.")
    parser.add_argument(
        "--jobs",
        default="ct,rad",
        help="Comma-separated jobs to run: ct, rad",
    )
    args = parser.parse_args()

    selected = {x.strip().lower() for x in args.jobs.split(",") if x.strip()}

    jobs = [
        Job(
            name="CT-RATE",
            repo_id="ibrahimhamamci/CT-RATE",
            csv_in=Path(r"C:\Users\DeCapo\Desktop\povetok\CT-RATE\outputs\smoke_dataset_450_broad.csv"),
            csv_clean=Path(r"C:\Users\DeCapo\Desktop\povetok\CT-RATE\outputs\smoke_dataset_450_broad_clean.csv"),
            manifest_out=Path(r"C:\Users\DeCapo\Desktop\povetok\CT-RATE\outputs\smoke_dataset_450_broad_download_manifest.csv"),
            image_out_dir=Path(r"C:\Users\DeCapo\Desktop\povetok\CT-RATE\selected_nii"),
            volume_col="VolumeName",
            preferred_prefixes=[
                "dataset/train_fixed/",
                "dataset/valid_fixed/",
                "dataset/train/",
                "dataset/valid/",
            ],
        ),
        Job(
            name="RadGenome-ChestCT",
            repo_id="RadGenome/RadGenome-ChestCT",
            csv_in=Path(r"C:\Users\DeCapo\Desktop\povetok\RadGenome-ChestCT\outputs\smoke_450_broad.csv"),
            csv_clean=Path(r"C:\Users\DeCapo\Desktop\povetok\RadGenome-ChestCT\outputs\smoke_450_broad_clean.csv"),
            manifest_out=Path(r"C:\Users\DeCapo\Desktop\povetok\RadGenome-ChestCT\outputs\smoke_450_broad_download_manifest.csv"),
            image_out_dir=Path(r"C:\Users\DeCapo\Desktop\povetok\RadGenome-ChestCT\selected_nii"),
            volume_col="Volumename",
            preferred_prefixes=[
                "dataset/train_preprocessed/",
                "dataset/valid_preprocessed/",
            ],
        ),
    ]

    for job in jobs:
        if job.name == "CT-RATE" and "ct" not in selected:
            continue
        if job.name == "RadGenome-ChestCT" and "rad" not in selected:
            continue
        run_job(job)


if __name__ == "__main__":
    main()
