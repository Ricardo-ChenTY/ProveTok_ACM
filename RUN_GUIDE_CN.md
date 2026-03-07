# ProveTok_ACM 运行说明（给协作者）

当前主流程：**Stage 0-4**（不调用 LLM 生成）。

## 1. 项目概述

Stage 0-4 做四件事：
1. Stage 0-2：确定性 token bank 构建（SwinUNETR + 八叉树分裂）
2. Stage 3：Router（teacher-forcing，用参考报告句子做查询）
3. Stage 4：Verifier 规则审计（R1-R5）
4. Stage 6：结构化日志落盘（trace.jsonl，每例可追溯）

关键入口：
- `run_mini_experiment.py`：主运行脚本
- `validate_stage0_4_outputs.py`：结果验收脚本

---

## 2. 数据接口

需要两个 manifest CSV（CT-RATE / RadGenome 各一个）。

必需列：
- `case_id`：病例唯一 ID
- `volume_path`：体数据路径（医学影像文件或 `.npy`）
- `report_text`：参考报告文本

---

## 3. 环境准备

```bash
# 确认 GPU
nvidia-smi

conda env create -f environment.yaml
conda activate provetok

# semantic encoder 必需
pip install sentence-transformers
```

首次运行 semantic encoder 会自动下载 `all-MiniLM-L6-v2`（约 90MB），之后缓存。

---

## 4. 当前推荐运行命令

### 变量定义（先设好路径）

```bash
CTRATE_CSV="/path/to/ctrate_manifest.csv"
RADGENOME_CSV="/path/to/radgenome_manifest.csv"
ENCODER_CKPT="/path/to/swinunetr.ckpt"
OUT_ROOT="/path/to/outputs"
```

### 4.1 50-case 验证（快速，推荐先跑）

使用 64³ resize，关闭 R4/R5（阈值未校准 / fallback 假阳性），隔离真实 R1/R2 违规率：

```bash
python run_mini_experiment.py \
  --ctrate_csv "${CTRATE_CSV}" \
  --radgenome_csv "${RADGENOME_CSV}" \
  --out_dir "${OUT_ROOT}/r2_taut005_ratio_0.8_nor4r5" \
  --max_cases 50 \
  --expected_cases_per_dataset 50 \
  --cp_strict \
  --encoder_ckpt "${ENCODER_CKPT}" \
  --text_encoder semantic \
  --text_encoder_model sentence-transformers/all-MiniLM-L6-v2 \
  --text_encoder_device cuda \
  --device cuda \
  --resize_d 64 --resize_h 64 --resize_w 64 \
  --token_budget_b 64 \
  --k_per_sentence 8 \
  --lambda_spatial 0.3 \
  --tau_iou 0.05 \
  --beta 0.1 \
  --r2_mode ratio \
  --r2_min_support_ratio 0.8 \
  --r4_disabled \
  --r5_fallback_disabled \
  --anatomy_spatial_routing
```

### 4.2 450/450 全量跑（128³，主实验）

```bash
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
  --anatomy_spatial_routing
```

> 128³ 不加 `--resize_d/h/w`（默认即为 128）。

### 4.3 验收

```bash
python validate_stage0_4_outputs.py \
  --out_dir "${OUT_ROOT}/outputs_stage0_4_450_128" \
  --datasets ctrate,radgenome \
  --expected_cases_map ctrate=450,radgenome=450 \
  --save_report "${OUT_ROOT}/outputs_stage0_4_450_128/validation_report.json"
```

---

## 5. 输出文件说明

所有输出在 `--out_dir` 下：

```
out_dir/
  summary.csv                    # 全量汇总
  ctrate_case_summary.csv
  radgenome_case_summary.csv
  run_meta.json                  # 本次运行参数记录
  cases/
    ctrate/<case_id>/
      tokens.npy / tokens.pt / tokens.json
      bank_meta.json
      trace.jsonl                # 逐句路由 + verifier 结果
    radgenome/<case_id>/
      ...
```

**trace.jsonl 格式：**
- 第 1 行：`{"type": "case_meta", "B": ..., "k": ..., "tau_IoU": ...}`
- 后续行：`{"type": "sentence", "sentence_text": ..., "anatomy_keyword": ..., "topk_token_ids": [...], "violations": [...]}`

---

## 6. 关键参数说明

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `--tau_iou` | 0.05 | R2 IoU 阈值 |
| `--r2_min_support_ratio` | 0.8 | cited tokens 中命中比例下限 |
| `--r4_disabled` | 建议开启 | R4 union bbox 阈值未校准，开启避免假阳性 |
| `--r5_fallback_disabled` | 建议开启 | R5 negation fallback 产生假阳性，开启隔离 |
| `--r2_mode` | ratio | 可选 `max_iou`（更宽松，见下节） |
| `--anatomy_spatial_routing` | 建议开启 | 有 anatomy keyword 的句子改用 IoU 主导路由，解决 w_proj 未训练导致的跨模态对齐缺失；预期 R2 从 33.6% 降至 15-20% |

---

## 7. 可选：R2 max_iou sweep

如果想对比 R2 max_iou 模式（只要最大 IoU ≥ tau 即通过）：

```bash
bash Scripts/run_r2_maxiou_sweep.sh
```

扫描 `tau_iou ∈ {0.10, 0.05, 0.02}`，输出到 Google Drive。汇总：

```bash
python Scripts/summarize_r2_sweep.py \
  --sweep_root /content/drive/MyDrive/Data/outputs_stage0_4_r2maxiou_sweep_50 \
  --glob "r2_maxiou_tau*" \
  --save_csv .../maxiou_sweep_summary.csv
```

---

## 8. 常见问题

| 问题 | 解决 |
|------|------|
| `sentence-transformers` 下载慢 | 挂 VPN 或提前下载后用 `--text_encoder_model /path/to/model` |
| 显存不足 | `--device cpu` 或减小 `--resize_d/h/w` |
| `expected_cases` 不匹配 | 去掉 `--expected_cases_per_dataset` 或检查 manifest 行数 |
| 结果分析 | 打开 `Smoke_analysis/OUTPUT_ANALYSIS_COLAB.ipynb` 在 Colab 跑 |
