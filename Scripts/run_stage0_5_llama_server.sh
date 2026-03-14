#!/usr/bin/env bash
# ============================================================
# Step 2: Stage 0-5  (Stage 0-4 + Llama-3.1-8B LLM 裁判)
#
# 前置条件:
#   models/Llama-3.1-8B-Instruct/ 目录存在（已下载模型）
#   下载方法: huggingface-cli download meta-llama/Llama-3.1-8B-Instruct \
#               --local-dir models/Llama-3.1-8B-Instruct
#
# 用法:
#   bash Scripts/run_stage0_5_llama_server.sh
# ============================================================
set -euo pipefail

# ─── 固定输入路径 ────────────────────────────────────────
CTRATE_CSV="/data/ProveTok_ACM/manifests/ctrate_900_manifest.csv"
RADGENOME_CSV="/data/ProveTok_ACM/manifests/radgenome_900_manifest.csv"
ENCODER_CKPT="/data/ProveTok_ACM/checkpoints/swinunetr.ckpt"
# ─────────────────────────────────────────────────────────

PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="${PROJ_ROOT}/outputs/stage0_5_llama_450"
MODEL_DIR="${PROJ_ROOT}/models/Llama-3.1-8B-Instruct"
CACHE_ROOT="${PROJ_ROOT}/.cache"
HF_HOME_DIR="${PROJ_ROOT}/.hf"

mkdir -p "${OUT_DIR}"
mkdir -p \
  "${CACHE_ROOT}/huggingface/hub" \
  "${CACHE_ROOT}/huggingface/transformers" \
  "${CACHE_ROOT}/sentence_transformers" \
  "${HF_HOME_DIR}"

export XDG_CACHE_HOME="${CACHE_ROOT}"
export HF_HOME="${HF_HOME_DIR}"
export HUGGINGFACE_HUB_CACHE="${CACHE_ROOT}/huggingface/hub"
export TRANSFORMERS_CACHE="${CACHE_ROOT}/huggingface/transformers"
export SENTENCE_TRANSFORMERS_HOME="${CACHE_ROOT}/sentence_transformers"

# 检查模型目录是否存在
if [ ! -d "${MODEL_DIR}" ]; then
  echo "❌ 模型目录不存在: ${MODEL_DIR}"
  echo ""
  echo "请先下载模型:"
  echo "  cd ${PROJ_ROOT}"
  echo "  huggingface-cli download meta-llama/Llama-3.1-8B-Instruct \\"
  echo "    --local-dir models/Llama-3.1-8B-Instruct"
  exit 1
fi

echo "=========================================="
echo "Stage 0-5  |  450/450  |  Llama-3.1-8B 裁判"
echo "tau_iou = 0.04  |  token_budget_b = 128  |  shuffle_seed = 42"
echo "模型: ${MODEL_DIR}"
echo "OUT:  ${OUT_DIR}"
echo "=========================================="

# 1. checkpoint 检测
python "${PROJ_ROOT}/Scripts/ckpt_probe.py" \
  --ckpt_path "${ENCODER_CKPT}" \
  --in_channels 1 \
  --out_channels 2 \
  --feature_size 48 \
  --save_report "${OUT_DIR}/ckpt_probe_report.json"

# 2. 主实验 + Stage 5 LLM 裁判
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
  --shuffle_seed 42 \
  --token_budget_b 128 \
  --k_per_sentence 8 \
  --lambda_spatial 0.3 \
  --tau_iou 0.04 \
  --beta 0.1 \
  --r2_mode ratio \
  --r2_min_support_ratio 0.8 \
  --r4_disabled \
  --r5_fallback_disabled \
  --anatomy_spatial_routing \
  --r2_skip_bilateral \
  --r1_negation_exempt \
  --r1_skip_midline \
  --r1_min_same_side_ratio 0.6 \
  --llm_judge huggingface \
  --llm_judge_model "${MODEL_DIR}" \
  --llm_judge_hf_torch_dtype bfloat16 \
  --llm_judge_alpha 0.5 2>&1 | tee "${OUT_DIR}/run.log"

# 3. 结构验收
python "${PROJ_ROOT}/validate_stage0_4_outputs.py" \
  --out_dir "${OUT_DIR}" \
  --datasets ctrate,radgenome \
  --expected_cases_map ctrate=450,radgenome=450 \
  --save_report "${OUT_DIR}/validation_report.json"

echo ""
echo "✅ Stage 0-5 完成: ${OUT_DIR}"
echo ""
echo "查看 LLM 裁判效果:"
echo "  summary.csv 中 n_judge_confirmed 列 = LLM 确认的违规句数"
echo "  n_violations 应低于纯 Stage 0-4 的结果（误报被过滤）"
