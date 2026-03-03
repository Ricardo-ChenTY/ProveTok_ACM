#!/usr/bin/env python3
"""
Smoke test: download ~450 NIfTI (.nii.gz) samples from HuggingFace.

Usage:
    python download_smoke_450.py --jobs ct,rad \
        --ct_csv  ../CT-RATE/outputs/smoke_dataset_450_broad.csv \
        --rad_csv ../RadGenome-ChestCT/outputs/smoke_450_broad.csv
"""
from _download_core import run_main

if __name__ == "__main__":
    run_main(script_label="smoke_450", default_log_interval=10)
