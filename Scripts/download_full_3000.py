#!/usr/bin/env python3
"""
Full download: ~3000 NIfTI (.nii.gz) samples from HuggingFace.

Usage:
    python download_full_3000.py --jobs ct,rad \
        --ct_csv  ../CT-RATE/outputs/full_dataset_3000.csv \
        --rad_csv ../RadGenome-ChestCT/outputs/full_3000.csv
"""
from _download_core import run_main

if __name__ == "__main__":
    run_main(script_label="full_3000", default_log_interval=100)
