#!/usr/bin/env bash
# ============================================================
# Step 3: 训练 W_proj（InfoNCE，Stage 3c 前置）
#
# 前置条件:
#   先跑过 Stage 0-4，有 outputs_stage0_4_450/cases/ 目录
#
# 用法:
#   bash Scripts/run_wprojection_train.sh
#   bash Scripts/run_wprojection_train.sh --cases_dir /custom/path/cases
# ============================================================
set -euo pipefail

PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# ─── 默认参数（可通过命令行覆盖）────────────────────────
CASES_DIR="${PROJ_ROOT}/outputs_stage0_4_450/cases"
OUT_DIR="${PROJ_ROOT}/outputs_wprojection"
EPOCHS=50
BATCH_SIZE=32
LR=1e-3
TAU=0.07
DEVICE=cuda
# ─────────────────────────────────────────────────────────

# 支持 --cases_dir 参数覆盖默认路径
while [[ $# -gt 0 ]]; do
  case "$1" in
    --cases_dir) CASES_DIR="$2"; shift 2 ;;
    --out_dir)   OUT_DIR="$2";   shift 2 ;;
    --epochs)    EPOCHS="$2";    shift 2 ;;
    --device)    DEVICE="$2";    shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

# 检查 cases 目录
if [ ! -d "${CASES_DIR}" ]; then
  echo "❌ cases 目录不存在: ${CASES_DIR}"
  echo ""
  echo "请先运行 Stage 0-4:"
  echo "  bash Scripts/run_stage0_4_server.sh"
  exit 1
fi

N_TRACES=$(find "${CASES_DIR}" -name "trace.jsonl" | wc -l)
echo "=========================================="
echo "W_proj 训练 (InfoNCE)  |  ${EPOCHS} epochs"
echo "cases: ${CASES_DIR}  (${N_TRACES} trace files)"
echo "out:   ${OUT_DIR}"
echo "=========================================="

mkdir -p "${OUT_DIR}"

python "${PROJ_ROOT}/train_wprojection.py" \
  --cases_dir     "${CASES_DIR}" \
  --out_dir       "${OUT_DIR}" \
  --text_encoder  semantic \
  --text_encoder_model sentence-transformers/all-MiniLM-L6-v2 \
  --text_encoder_device "${DEVICE}" \
  --epochs        "${EPOCHS}" \
  --batch_size    "${BATCH_SIZE}" \
  --lr            "${LR}" \
  --tau           "${TAU}" \
  --device        "${DEVICE}"

echo ""
echo "✅ W_proj 训练完成: ${OUT_DIR}/w_proj.pt"
echo "   训练曲线: ${OUT_DIR}/train_log.json"
echo ""
echo "加载方式 (在代码里):"
echo "  import torch"
echo "  w = torch.load('${OUT_DIR}/w_proj.pt').tolist()"
echo "  router = Router(cfg=cfg.router, text_encoder=..., w_proj=w)"
