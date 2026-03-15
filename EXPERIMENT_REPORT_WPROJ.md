# ProveTok — W_proj 训练与测试集评估实验报告

> 实验日期：2026-03-14
> 输出目录：`outputs/test_trained_wproj/`（测试集）、`outputs_wprojection/`（训练产出）

---

## 1. 背景与动机

### 1.1 问题

在之前的 450/450 Baseline 实验中，Router（Stage 3b）使用 **identity W_proj**（单位矩阵）做跨模态路由——即 3D token embedding 直接与文本 embedding 做余弦相似度，没有经过 InfoNCE 对齐训练。同时，路由模式使用 `anatomy_spatial_routing`（IoU 主导，语义只作 tiebreaker）。

论文 §5.2 明确要求：
1. **W_proj 应经 InfoNCE 训练**：学习 3D 视觉 → 文本语义空间的线性投影
2. **路由应使用 Eq 6-8 语义路由**：`r_i = cos(q_s, W_proj · f_i) + λ_spatial · IoU`

本实验的目标是：训练 W_proj，然后在 hold-out 测试集上评估 trained W_proj + semantic routing 是否比 identity W_proj + spatial routing 产生更少的 rule violations。

### 1.2 两组实验对比

| | **Baseline** | **Trained W_proj** |
|--|--|--|
| W_proj | Identity（单位矩阵） | InfoNCE 训练后的 [384×768] 矩阵 |
| 路由模式 | `anatomy_spatial_routing`（IoU 主导） | Semantic routing（Eq 6-8） |
| Stage 3c 生成 | 未启用（使用原始 report 文本） | 启用（Llama-3.1-8B token-gated 生成） |
| Stage 5 重路由 | 未启用 | 启用（log-smooth penalty + re-top-k + regeneration） |
| 评估数据 | 900 全集（含 train/val/test） | 180 test cases only |

---

## 2. 数据 Split

从 900 采样池中，使用 `shuffle_seed=42` 划分 train/val/test：

| Split | 数量 | 占比 | 用途 |
|-------|------|------|------|
| train | 540 (270+270) | 60% | W_proj InfoNCE 训练 |
| val | 180 (90+90) | 20% | 训练时 early stopping |
| test | 180 (90+90) | 20% | 本次评估（与 baseline 对比） |

Split 文件：`manifests/split_seed42/{train,val,test}.txt`

测试集 manifest：
- `manifests/ctrate_test.csv`（90 cases）
- `manifests/radgenome_test.csv`（90 cases）

---

## 3. W_proj 训练

### 3.1 训练配置

| 参数 | 值 |
|------|-----|
| 损失函数 | InfoNCE (temperature-scaled softmax CE) |
| 温度 τ | 0.07 |
| 学习率 | 1e-3 |
| Optimizer | Adam |
| Batch size | 64 |
| Epochs | 30 |
| 投影维度 | 768 → 384 (text_dim → token_dim) |
| 文本编码器 | sentence-transformers/all-MiniLM-L6-v2 |
| 3D 编码器 | SwinUNETR (frozen) |

### 3.2 训练结果

| 指标 | Identity W_proj | Trained W_proj | 变化 |
|------|----------------|---------------|------|
| Test InfoNCE Loss | 2.9151 | 1.6926 | **-41.9%** |

训练产出：`outputs_wprojection/w_proj.pt`（shape: [384, 768]）

InfoNCE loss 的大幅下降说明 **trained W_proj 在嵌入空间对齐上显著优于 identity**——text query 和 3D token 之间的跨模态匹配质量提升了。

---

## 4. 测试集评估配置

### 4.1 Trained W_proj 组（本次实验）

```bash
python run_mini_experiment.py \
  --ctrate_csv manifests/ctrate_test.csv \
  --radgenome_csv manifests/radgenome_test.csv \
  --max_cases 90 --expected_cases_per_dataset 90 \
  --text_encoder semantic \
  --w_proj_path outputs_wprojection/w_proj.pt \
  --stage3c_backend huggingface --stage3c_model models/Llama-3.1-8B-Instruct \
  --reroute_gamma 2.0 --reroute_max_retry 1 \
  --tau_iou 0.04 --token_budget_b 128 --k_per_sentence 8 \
  --lambda_spatial 0.3 --beta 0.1 \
  --llm_judge huggingface --llm_judge_model models/Llama-3.1-8B-Instruct
```

关键差异：
- `--w_proj_path`：加载训练后的 W_proj
- **无** `--anatomy_spatial_routing`：使用 semantic routing（Eq 6-8）
- `--stage3c_backend`：启用 LLM 生成
- `--reroute_gamma 2.0 --reroute_max_retry 1`：启用 Stage 5 log-smooth 重路由

### 4.2 Baseline 组（从 450/450 实验中提取同一 180 cases）

- Identity W_proj + `anatomy_spatial_routing`
- 无 Stage 3c 生成（使用原始 report 文本）
- 无 Stage 5 重路由
- 数据来源：`outputs/stage0_5_llama_450/summary.csv` 中筛选 180 个 test case_id

---

## 5. 实验结果

### 5.1 总览

| 指标 | Baseline (Identity + Spatial) | Trained W_proj + Semantic | 变化 |
|------|------|------|------|
| **Total violations** | 191 | 285 | **+49.2%** |
| **Judge confirmed** | 90 | 172 | **+91.1%** |
| Avg violations/case | 1.06 | 1.58 | +0.52 |
| Avg confirmed/case | 0.50 | 0.96 | +0.46 |
| Zero-violation cases | 63 (35.0%) | 32 (17.8%) | -17.2pp |

### 5.2 按数据集

**CT-RATE (90 test cases, 717 sentences):**

| 指标 | Baseline | Trained | 变化 |
|------|----------|---------|------|
| Total violations | 108 | 139 | +28.7% |
| Judge confirmed | 46 | 72 | +56.5% |
| Violation rate (per sentence) | 15.1% | 19.4% | +4.3pp |
| Confirmed rate (per sentence) | 6.4% | 10.0% | +3.6pp |
| Zero-violation cases | 26 (28.9%) | 19 (21.1%) | -7.8pp |

**RadGenome (90 test cases, 720 sentences):**

| 指标 | Baseline | Trained | 变化 |
|------|----------|---------|------|
| Total violations | 83 | 146 | +75.9% |
| Judge confirmed | 44 | 100 | +127.3% |
| Violation rate (per sentence) | 11.5% | 20.3% | +8.8pp |
| Confirmed rate (per sentence) | 6.1% | 13.9% | +7.8pp |
| Zero-violation cases | 37 (41.1%) | 13 (14.4%) | -26.7pp |

### 5.3 Case 级别配对对比 (n=180)

| 指标 | 变好 | 持平 | 变差 |
|------|------|------|------|
| n_violations | 29 (16.1%) | 68 (37.8%) | **83 (46.1%)** |
| n_judge_confirmed | 19 (10.6%) | 79 (43.9%) | **82 (45.6%)** |

### 5.4 Violation 分布

**Trained W_proj — CT-RATE:**

| Violations/case | 0 | 1 | 2 | 3 | 4 |
|-----------------|---|---|---|---|---|
| Cases | 19 | 25 | 27 | 16 | 3 |
| 占比 | 21.1% | 27.8% | 30.0% | 17.8% | 3.3% |

**Trained W_proj — RadGenome:**

| Violations/case | 0 | 1 | 2 | 3 | 4 | 5 |
|-----------------|---|---|---|---|---|---|
| Cases | 13 | 35 | 23 | 13 | 4 | 2 |
| 占比 | 14.4% | 38.9% | 25.6% | 14.4% | 4.4% | 2.2% |

### 5.5 Stage 3c 生成 + Stage 5 重路由统计

| 指标 | 值 |
|------|-----|
| Total sentences | 1,437 |
| LLM generated (Stage 3c) | 1,437 (100%) |
| Rerouted (Stage 5) | 172 (12.0%) |
| De-specified fallback | 21 (1.5%) |

---

## 6. 分析与讨论

### 6.1 结论：Trained W_proj + Semantic Routing 的 violation 率高于 Baseline

这一结果**与预期相反**。虽然 W_proj 的 InfoNCE loss 下降了 41.9%（嵌入对齐显著改善），但下游 pipeline 的 violation 率反而上升了约 50%。

### 6.2 可能的原因

#### (a) 变量混淆：同时改变了两个因素

本实验同时修改了**两个变量**：
1. W_proj：identity → trained
2. 路由模式：`anatomy_spatial_routing` → semantic routing

无法区分 violation 增加是因为 W_proj 变差还是去掉 spatial routing 导致的。Baseline 的 `anatomy_spatial_routing` 模式中 IoU 是主导分量（语义只作 tiebreaker），这对 R1（侧别）和 R2（解剖区域）验证天然友好。切换到纯语义路由后，路由决策更依赖 embedding 相似度，但 verifier 的空间规则（IoU-based）不一定被满足。

#### (b) InfoNCE 优化目标 vs Verifier 规则的 gap

InfoNCE loss 优化的是 **text-token embedding 的余弦对齐**，而 Stage 4 verifier 检查的是**空间一致性**（bbox IoU、侧别、深度层级）。一个 token 在嵌入空间里与句子很接近，但它的 bbox 可能不与目标解剖区域重叠。

训练目标和评估指标之间存在 gap：
- InfoNCE ↓ ≠ Violations ↓
- 语义匹配 ≠ 空间合规

#### (c) Stage 3c LLM 生成引入额外 violation 来源

Baseline 使用原始 report 文本（直接来自数据集），而 trained W_proj 组使用 LLM 生成的文本。LLM 生成可能引入幻觉或不精确的解剖描述，导致 verifier 检测到更多 violation。

### 6.3 建议的后续实验

为了隔离各因素的贡献，建议做以下消融实验：

| 实验 | W_proj | 路由模式 | Stage 3c | 目的 |
|------|--------|---------|---------|------|
| A（已完成：Baseline） | Identity | Spatial | OFF | 对照组 |
| B（已完成：本次） | Trained | Semantic | ON | 全量对比 |
| **C（建议）** | Trained | **Spatial** | ON | 隔离 W_proj 效果 |
| **D（建议）** | Identity | Semantic | ON | 隔离路由模式效果 |
| **E（建议）** | Trained | Semantic | **OFF** | 隔离 Stage 3c 影响 |

优先建议实验 **C**（trained W_proj + spatial routing），因为这只改变了投影质量，保留了空间路由的约束，可以验证 W_proj 训练本身是否有益。

---

## 7. 文件清单

### 7.1 W_proj 训练相关

```
outputs_wprojection/
├── w_proj.pt                    # 训练后的 W_proj 矩阵 [384×768]
├── training_log.json            # 训练日志（每 epoch loss）
└── ...

train_wprojection.py             # W_proj 训练脚本
Scripts/eval_wprojection_test.py # W_proj InfoNCE loss 评估脚本
```

### 7.2 测试集评估相关

```
manifests/
├── split_seed42/
│   ├── train.txt   (540 cases)
│   ├── val.txt     (180 cases)
│   └── test.txt    (180 cases)
├── ctrate_test.csv              # 90 ctrate test cases
└── radgenome_test.csv           # 90 radgenome test cases

outputs/test_trained_wproj/
├── run.log                      # 完整运行日志
├── run_meta.json                # 实验超参数
├── summary.csv                  # 逐 case 结果（180 行）
├── ctrate_case_summary.csv      # ctrate 中间汇总
└── cases/
    ├── ctrate/    (90 cases)
    └── radgenome/ (90 cases)

Scripts/
├── run_test_trained_wproj.sh    # 测试集评估运行脚本
└── filter_manifest_by_split.py  # Manifest 按 split 过滤工具
```

---

## 8. 可复现性

```bash
# 1. 训练 W_proj
python train_wprojection.py

# 2. 评估 W_proj InfoNCE loss
python Scripts/eval_wprojection_test.py

# 3. 生成测试集 manifest
python Scripts/filter_manifest_by_split.py \
  --manifest manifests/ctrate_900_manifest.csv \
  --split_file manifests/split_seed42/test.txt \
  --dataset ctrate --out manifests/ctrate_test.csv

python Scripts/filter_manifest_by_split.py \
  --manifest manifests/radgenome_900_manifest.csv \
  --split_file manifests/split_seed42/test.txt \
  --dataset radgenome --out manifests/radgenome_test.csv

# 4. 运行测试集评估
bash Scripts/run_test_trained_wproj.sh
```
