# ProveTok_ACM 运行说明

当前主流程：`Stage 0-4`（规则验证）+ 可选 `Stage 5`（LLM 裁判）。

## 1. 当前锁定配置

这轮准备直接跑 `450/450` 主实验，推荐固定以下开关：

- `--cp_strict`
- `--r2_mode ratio`
- `--r2_min_support_ratio 0.8`
- `--tau_iou 0.05`
- `--anatomy_spatial_routing`
- `--r2_skip_bilateral`
- `--r1_negation_exempt`
- `--r1_skip_midline`
- `--r1_min_same_side_ratio 0.6`
- `--r4_disabled`
- `--r5_fallback_disabled`

50-case 回归历程（128^3 基准）：

- 第五轮（negation_exempt + skip_midline）：R1: 100 → 90，违规率 17.375% → 16.50%
- 第六轮（ratio 0.6）：R1: 90，ratio 模式对全错侧 token 无效
- 第七轮（left/right lung bbox fallback）：R1: 90 → 82，R2: 42 → 42（128^3）
- 第八轮（bbox 边界对齐 0.50）：R1: 82，Plan A 无收益（死区不是主因）

配置已锁定，直接上 450/450。

## 2. 必需输入

需要两个 manifest CSV：

- `ctrate_manifest.csv`
- `radgenome_manifest.csv`

每个 CSV 至少包含：

- `case_id`
- `volume_path`
- `report_text`

还需要一个可加载的 SwinUNETR checkpoint：

- `SwinUNETR(in_channels=1, out_channels=2, feature_size=48)`

## 3. 环境准备

```bash
nvidia-smi

conda env create -f environment.yaml
conda activate provetok
pip install sentence-transformers
```

## 4. 450/450 主实验命令

### 4.1 Linux / Colab Shell

```bash
CTRATE_CSV="/path/to/ctrate_manifest.csv"
RADGENOME_CSV="/path/to/radgenome_manifest.csv"
ENCODER_CKPT="/path/to/swinunetr.ckpt"
OUT_ROOT="/path/to/outputs"

python run_mini_experiment.py \
  --ctrate_csv "${CTRATE_CSV}" \
  --radgenome_csv "${RADGENOME_CSV}" \
  --out_dir "${OUT_ROOT}/outputs_stage0_4_450_128" \
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
```

### 4.2 Windows PowerShell

```powershell
$CTRATE_CSV    = "C:\path\to\ctrate_manifest.csv"
$RADGENOME_CSV = "C:\path\to\radgenome_manifest.csv"
$ENCODER_CKPT  = "C:\path\to\swinunetr.ckpt"
$OUT_ROOT      = "C:\path\to\outputs"

python run_mini_experiment.py `
  --ctrate_csv $CTRATE_CSV `
  --radgenome_csv $RADGENOME_CSV `
  --out_dir "$OUT_ROOT\outputs_stage0_4_450_128" `
  --max_cases 450 `
  --expected_cases_per_dataset 450 `
  --cp_strict `
  --encoder_ckpt $ENCODER_CKPT `
  --text_encoder semantic `
  --text_encoder_model sentence-transformers/all-MiniLM-L6-v2 `
  --text_encoder_device cuda `
  --device cuda `
  --token_budget_b 64 `
  --k_per_sentence 8 `
  --lambda_spatial 0.3 `
  --tau_iou 0.05 `
  --beta 0.1 `
  --r2_mode ratio `
  --r2_min_support_ratio 0.8 `
  --r4_disabled `
  --r5_fallback_disabled `
  --anatomy_spatial_routing `
  --r2_skip_bilateral `
  --r1_negation_exempt `
  --r1_skip_midline `
  --r1_min_same_side_ratio 0.6
```

## 5. 结构验收

```bash
python validate_stage0_4_outputs.py \
  --out_dir "${OUT_ROOT}/outputs_stage0_4_450_128" \
  --datasets ctrate,radgenome \
  --expected_cases_map ctrate=450,radgenome=450 \
  --save_report "${OUT_ROOT}/outputs_stage0_4_450_128/validation_report.json"
```

通过标准：

- `Validated cases: 900`
- `Failed: 0`
- `run_meta.json` 中 `cp_strict = true`
- `run_meta.json` 中 `ctrate.selected_rows = 450`
- `run_meta.json` 中 `radgenome.selected_rows = 450`

## 6. Colab 验收 Notebook

使用：

- [OUTPUT_ANALYSIS_COLAB.ipynb](c:\Users\34228\Desktop\ACM\Smoke_analysis\OUTPUT_ANALYSIS_COLAB.ipynb)

你朋友在 Colab 里只需要：

1. 挂载 Google Drive
2. 打开 notebook
3. 把 `OUT_DIR` 改成 `450/450` 输出目录
4. 运行全部 cells

notebook 现在默认按 `450/450` 口径验收，会检查：

- `expected_cases_map = ctrate=450,radgenome=450`
- `validation_report.json` 是否全过
- `run_meta.json` 的关键开关是否符合主实验配置
- `summary.csv`、`cases/*/*/trace.jsonl` 是否能正常汇总

notebook 会导出：

- `analysis_exports/dataset_aggregate.csv`
- `analysis_exports/sentence_violation_rate.csv`
- `analysis_exports/rule_violation_count.csv`
- `analysis_exports/abnormal_cases_ranked.csv`
- `analysis_exports/sentence_detail.csv`

## 7. 结果怎么看

硬性要求：

- 结构验收通过
- 样本数是 `450/450`
- `R1` 不出现明显反弹
- `R2` 相比旧基线 `2651` 明显下降

软性目标：

- 总体 `violation_sentence_rate` 远低于旧基线 `0.721`
- 如果最终落在 `0.15 ~ 0.25`，说明 `50/50` smoke 的收益基本稳定迁移到了全量
- 如果高于 `0.35`，要回头检查数据分布漂移或参数/规则联动问题

## 8. 输出目录最小交付物

重跑结束后，至少保留：

- `summary.csv`
- `ctrate_case_summary.csv`
- `radgenome_case_summary.csv`
- `run_meta.json`
- `validation_report.json`
- `cases/*/*/trace.jsonl`

如果只做分析，`cache/` 不是必需。

## 9. 常见问题

- `expected_cases_map` 不通过：先检查 manifest 是否真的各有 `450` 行。
- `sentence-transformers` 下载慢：可提前缓存模型，或把 `--text_encoder_model` 指向本地目录。
- 显存不足：可临时改成 `64^3` 验证，但主实验仍建议保留 `128^3`。
- 只要做结果分析：直接用 notebook，不需要重跑主实验。

---

## 10. 整体规划与完整执行流程

### 10.1 当前进度

| 组件 | 状态 | 说明 |
|------|------|------|
| Stage 0-4 规则验证 | ✅ 完成 | 可直接跑 450/450 |
| Stage 5 LLM 裁判 | ✅ 代码完成 | 等 Llama 模型下载后即可运行 |
| W_proj InfoNCE 训练 | ✅ 脚本完成 | 需先有 Stage 0-4 的 token bank |
| Stage 3c LLM 生成 | ✅ 框架完成 | 需 W_proj 训练完才能用 |

### 10.2 在服务器上的完整执行顺序

```bash
# === 0. 先改脚本里的路径 ===
vim Scripts/run_stage0_4_server.sh       # 填 CTRATE_CSV / RADGENOME_CSV / ENCODER_CKPT
vim Scripts/run_stage0_5_llama_server.sh # 同上

# === 1. 下载 Llama 模型（一次性，约 16GB）===
cd /path/to/ProveTok_ACM
huggingface-cli login
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct \
  --local-dir models/Llama-3.1-8B-Instruct

# === 2. 跑 Stage 0-4 baseline（纯规则，450/450）===
bash Scripts/run_stage0_4_server.sh
# 输出: outputs_stage0_4_450/summary.csv  validation_report.json

# === 3. 跑 Stage 0-5（带 LLM 裁判，450/450）===
bash Scripts/run_stage0_5_llama_server.sh
# 输出: outputs_stage0_5_llama_450/summary.csv
# 对比 Step 2 的 n_violations，LLM 过滤误报后应明显降低

# === 4. 训练 W_proj（用 Step 2 的 token bank）===
bash Scripts/run_wprojection_train.sh \
  --cases_dir outputs_stage0_4_450/cases
# 输出: outputs_wprojection/w_proj.pt  train_log.json

# === 5. Stage 3c（后续，需 W_proj 完成后再做）===
# 代码在 ProveTok_Main_experiment/stage3c_generator.py，待集成
```

### 10.3 结果对比方法

Step 2 vs Step 3 对比看 `summary.csv`：

| 指标 | Stage 0-4 (Step 2) | Stage 0-5 (Step 3) | 期望 |
|------|--------------------|--------------------|------|
| `n_violations` 总和 | baseline | 应下降 | LLM 过滤误报 |
| `n_judge_confirmed` 总和 | 0（无裁判） | > 0 | LLM 确认的真实违规数 |
| `violation_sentence_rate` | baseline | 应下降 | 误报减少 |

---

## 11. Stage 5 LLM 裁判（详细说明）

Stage 5 在 Stage 4 规则验证后接入 LLM，对每条违规做二次裁决，过滤误报，并对确认违规句应用路由分数惩罚。

### 11.1 支持的后端

| 后端 | 说明 | 默认模型 |
|------|------|----------|
| `huggingface` | 本地模型（推荐，服务器部署） | `models/Llama-3.1-8B-Instruct` |
| `ollama` | 本地 Ollama 服务 | `qwen2.5:7b` |
| `openai` | OpenAI API | `gpt-4o-mini` |
| `anthropic` | Anthropic API | `claude-haiku-4-5-20251001` |

### 11.2 准备本地 Llama-3.1-8B-Instruct 模型

```bash
# 在服务器上下载（需要 HuggingFace 账号并申请 Llama-3 访问权限）
cd /path/to/ProveTok_ACM
huggingface-cli login
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct \
  --local-dir models/Llama-3.1-8B-Instruct

# 确认目录结构
ls models/Llama-3.1-8B-Instruct/
# 应包含: config.json, tokenizer.json, model-*.safetensors, ...
```

### 11.3 运行（带 Stage 5）

```bash
python run_mini_experiment.py \
  --ctrate_csv "${CTRATE_CSV}" \
  --radgenome_csv "${RADGENOME_CSV}" \
  --out_dir "${OUT_ROOT}/outputs_stage0_5_450_128" \
  --max_cases 450 \
  --expected_cases_per_dataset 450 \
  --cp_strict \
  --encoder_ckpt "${ENCODER_CKPT}" \
  --text_encoder semantic \
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
  --r1_min_same_side_ratio 0.6 \
  --llm_judge huggingface \
  --llm_judge_hf_torch_dtype bfloat16
  # 默认自动找 models/Llama-3.1-8B-Instruct，不需要指定 --llm_judge_model
```

### 11.4 主要参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--llm_judge` | 启用 Stage 5，选后端 | 不传则不启用 |
| `--llm_judge_model` | 模型名或路径 | 各后端有默认值 |
| `--llm_judge_alpha` | CP 惩罚系数 α，`S'=S×(1-α×sev)` | `0.5` |
| `--llm_judge_hf_torch_dtype` | bfloat16 / float16 / float32 | `bfloat16` |
| `--llm_judge_hf_device_map` | auto / cuda / cpu | `auto` |

### 11.5 Stage 5 输出

- `trace.jsonl` 每条 sentence 行新增 `stage5_judgements` 字段：
  ```json
  "stage5_judgements": [
    {"rule_id": "R1_LATERALITY", "confirmed": true, "adjusted_severity": 0.8, "reasoning": "..."}
  ]
  ```
- `summary.csv` 新增 `n_judge_confirmed` 列（LLM 确认的违规句数）

### 11.6 W_proj 训练（Stage 3c 前置，暂不强制）

```bash
# 先用 Stage 0-4 跑 50+ cases 生成 token bank
python run_mini_experiment.py --ctrate_csv ... --radgenome_csv ... --max_cases 50

# 训练 W_proj（InfoNCE，约 50 epoch）
python train_wprojection.py \
  --cases_dir outputs_stage0_4/cases \
  --text_encoder semantic \
  --epochs 50 \
  --out_dir outputs_wprojection

# 输出：outputs_wprojection/w_proj.pt，训练 loss 曲线见 train_log.json
```
