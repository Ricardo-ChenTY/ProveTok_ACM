#!/usr/bin/env bash
set -euo pipefail

# Example usage in Colab terminal:
# bash Scripts/run_450_cp_strict_colab.sh

CTRATE_CSV="/content/drive/MyDrive/Data/manifests/ctrate_manifest.csv"
RADGENOME_CSV="/content/drive/MyDrive/Data/manifests/radgenome_manifest.csv"
ENCODER_CKPT="/content/drive/MyDrive/Data/checkpoints/swinunetr.ckpt"
OUT_DIR="/content/drive/MyDrive/Data/outputs_stage0_4_full450_cp_strict"

mkdir -p "${OUT_DIR}"

# Step 0: checkpoint compatibility gate (must pass before full run)
python Scripts/ckpt_probe.py \
  --ckpt_path "${ENCODER_CKPT}" \
  --in_channels 1 \
  --out_channels 2 \
  --feature_size 48 \
  --save_report "${OUT_DIR}/ckpt_probe_report.json"

python run_mini_experiment.py \
  --ctrate_csv "${CTRATE_CSV}" \
  --radgenome_csv "${RADGENOME_CSV}" \
  --out_dir "${OUT_DIR}" \
  --max_cases 450 \
  --expected_cases_per_dataset 450 \
  --cp_strict \
  --encoder_ckpt "${ENCODER_CKPT}" \
  --text_encoder semantic \
  --text_encoder_model sentence-transformers/all-MiniLM-L6-v2 \
  --text_encoder_device cuda \
  --device cuda \
  --token_budget_b 64 \
  --k_per_sentence 8 \
  --lambda_spatial 0.3 \
  --tau_iou 0.1 \
  --beta 0.1

python validate_stage0_4_outputs.py \
  --out_dir "${OUT_DIR}" \
  --datasets ctrate,radgenome \
  --expected_cases_map ctrate=450,radgenome=450 \
  --save_report "${OUT_DIR}/validation_report.json"

echo "Done: ${OUT_DIR}"
