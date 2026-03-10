#!/usr/bin/env bash
# ============================================================
# Step 1: Stage 0-4 Baseline (服务器版，450/450，无 LLM)
#
# 用法:
#   bash Scripts/run_stage0_4_server.sh
#
# 改下面的路径后直接跑，结果存到 OUT_DIR
# ============================================================
set -euo pipefail

# ─── 改这里 ──────────────────────────────────────────────
CTRATE_CSV="/path/to/ctrate_manifest.csv"
RADGENOME_CSV="/path/to/radgenome_manifest.csv"
ENCODER_CKPT="/path/to/swinunetr.ckpt"
OUT_DIR="$(dirname "$0")/../outputs_stage0_4_450"
# ─────────────────────────────────────────────────────────

PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "${OUT_DIR}"

echo "=========================================="
echo "Stage 0-4 Baseline  |  450/450"
echo "OUT: ${OUT_DIR}"
echo "=========================================="

# 1. checkpoint 兼容性检测（快速，跑失败直接报错）
python "${PROJ_ROOT}/Scripts/ckpt_probe.py" \
  --ckpt_path "${ENCODER_CKPT}" \
  --in_channels 1 \
  --out_channels 2 \
  --feature_size 48 \
  --save_report "${OUT_DIR}/ckpt_probe_report.json"

# 2. 主实验
python "${PROJ_ROOT}/run_mini_experiment.py" \
  --ctrate_csv    "${CTRATE_CSV}" \
  --radgenome_csv "${RADGENOME_CSV}" \
  --out_dir       "${OUT_DIR}" \
  --max_cases 450 \
  --expected_cases_per_dataset 450 \
  --cp_strict \
  --encoder_ckpt  "${ENCODER_CKPT}" \
  --text_encoder  semantic \
  --text_encoder_model sentence-transformers/all-MiniLM-L6-v2 \
  --text_encoder_device cuda \
  --device cuda \
  --token_budget_b 64 \
  --k_per_sentence 8 \
  --lambda_spatial 0.3 \
  --tau_iou 0.05 \
  --beta 0.1 \
  --r2_mode ratio \
  --r2_min_support_ratio 0.8 \
  --r4_disabled \
  --r5_fallback_disabled \
  --anatomy_spatial_routing \
  --r2_skip_bilateral \
  --r1_negation_exempt \
  --r1_skip_midline \
  --r1_min_same_side_ratio 0.6

# 3. 结构验收
python "${PROJ_ROOT}/validate_stage0_4_outputs.py" \
  --out_dir "${OUT_DIR}" \
  --datasets ctrate,radgenome \
  --expected_cases_map ctrate=450,radgenome=450 \
  --save_report "${OUT_DIR}/validation_report.json"

echo ""
echo "✅ Stage 0-4 完成: ${OUT_DIR}"
echo "   summary.csv 和 validation_report.json 已生成"
echo ""
echo "下一步:"
echo "  Stage 0-5 (加 LLM 裁判): bash Scripts/run_stage0_5_llama_server.sh"
echo "  训练 W_proj:              bash Scripts/run_wprojection_train.sh --cases_dir ${OUT_DIR}/cases"
