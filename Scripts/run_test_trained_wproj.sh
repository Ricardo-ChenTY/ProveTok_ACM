#!/usr/bin/env bash
# ============================================================
# Evaluate trained W_proj on 180 test cases (semantic routing)
#
# Comparison:
#   Baseline (existing):  identity W_proj + anatomy_spatial_routing
#   This run:             trained W_proj + semantic routing (Eq 6-8)
#
# Usage:
#   bash Scripts/run_test_trained_wproj.sh
# ============================================================
set -euo pipefail

# ─── Paths ────────────────────────────────────────
PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CTRATE_CSV="${PROJ_ROOT}/manifests/ctrate_test.csv"
RADGENOME_CSV="${PROJ_ROOT}/manifests/radgenome_test.csv"
ENCODER_CKPT="${PROJ_ROOT}/checkpoints/swinunetr.ckpt"
W_PROJ_PATH="${PROJ_ROOT}/outputs_wprojection/w_proj.pt"
MODEL_DIR="${PROJ_ROOT}/models/Llama-3.1-8B-Instruct"
OUT_DIR="${PROJ_ROOT}/outputs/test_trained_wproj"
CACHE_ROOT="${PROJ_ROOT}/.cache"
HF_HOME_DIR="${PROJ_ROOT}/.hf"
# ──────────────────────────────────────────────────

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

# Activate conda environment
CONDA_BASE="${PROJ_ROOT}/miniconda3"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate provetok

echo "=========================================="
echo "Test Set Evaluation: Trained W_proj"
echo "  90 ctrate + 90 radgenome = 180 cases"
echo "  W_proj: ${W_PROJ_PATH}"
echo "  Routing: semantic (NOT anatomy_spatial)"
echo "  OUT: ${OUT_DIR}"
echo "=========================================="

python "${PROJ_ROOT}/run_mini_experiment.py" \
  --ctrate_csv    "${CTRATE_CSV}" \
  --radgenome_csv "${RADGENOME_CSV}" \
  --out_dir       "${OUT_DIR}" \
  --max_cases 90 \
  --expected_cases_per_dataset 90 \
  --cp_strict \
  --encoder_ckpt  "${ENCODER_CKPT}" \
  --text_encoder  semantic \
  --text_encoder_model sentence-transformers/all-MiniLM-L6-v2 \
  --text_encoder_device cuda \
  --device cuda \
  --token_budget_b 128 \
  --k_per_sentence 8 \
  --lambda_spatial 0.3 \
  --tau_iou 0.04 \
  --beta 0.1 \
  --r2_mode ratio \
  --r2_min_support_ratio 0.8 \
  --r4_disabled \
  --r5_fallback_disabled \
  --r2_skip_bilateral \
  --r1_negation_exempt \
  --r1_skip_midline \
  --r1_min_same_side_ratio 0.6 \
  --w_proj_path "${W_PROJ_PATH}" \
  --llm_judge huggingface \
  --llm_judge_model "${MODEL_DIR}" \
  --llm_judge_hf_torch_dtype bfloat16 \
  --llm_judge_alpha 0.5 \
  --stage3c_backend huggingface \
  --stage3c_model "${MODEL_DIR}" \
  --stage3c_temperature 0.3 \
  --stage3c_max_tokens 256 \
  --reroute_gamma 2.0 \
  --reroute_max_retry 1 2>&1 | tee "${OUT_DIR}/run.log"

echo ""
echo "Done: ${OUT_DIR}"
echo "Compare with baseline: outputs/stage0_5_llama_450/"
