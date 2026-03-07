# Stage0-4 阶段性分析（含本次 450/450 结果）

更新日期：2026-03-07  
作者说明：本文件覆盖此前同名报告，加入本次 450/450 全量运行与 notebook 导出结果。

## 1. 本次新增结果（450/450）

运行目录：`outputs_stage0_4_follow_request_20260307_1705/outputs_stage0_4_450_128`

关键配置（与 guide 一致）：

- `cp_strict = true`
- `r2_mode = ratio`
- `r2_min_support_ratio = 0.8`
- `tau_iou = 0.05`
- `anatomy_spatial_routing = true`
- `r2_skip_bilateral = true`
- `r4_disabled = true`
- `r5_fallback_disabled = true`

结构验收：

- `validation_report.json`: `900` 条，`failed = 0`

## 2. 450/450 核心数据

### 2.1 数据集聚合（来自 `dataset_aggregate.csv`）

- `ctrate`: `450` case, `avg_violations=2.658`, `total_violations=1196`, `avg_vio_per_sent=0.333`
- `radgenome`: `450` case, `avg_violations=2.209`, `total_violations=994`, `avg_vio_per_sent=0.277`

### 2.2 句级违规率（来自 `sentence_violation_rate.csv`）

- `ctrate`: `1166 / 3596 = 32.42%`
- `radgenome`: `974 / 3591 = 27.12%`
- 总体：`2140 / 7187 = 29.78%`

### 2.3 规则贡献（来自 `rule_violation_count.csv`）

- `R1_LATERALITY = 1770`
- `R2_ANATOMY = 420`

占比（按规则计数）：

- `R1`: 约 `80.8%`
- `R2`: 约 `19.2%`

### 2.4 病例分布（来自 `case_violation_ranked.csv`）

- `violation_ratio == 1.0`: `0`
- `violation_ratio >= 0.75`: `12`
- 中位数：`0.25`
- Q3：`0.375`

## 3. 与前几次 50-case 的对比

| Run | Cases | Sentences | Violation Sentence Rate | Total Violations | R1 | R2 |
|---|---:|---:|---:|---:|---:|---:|
| 50 baseline（3/6） | 100 | 800 | 45.75% | 388 | 119 | 269 |
| 50 + anatomy（3/7 13:44） | 100 | 800 | 42.00% | 339 | 100 | 239 |
| 50 + anatomy + skip（3/7 15:00） | 100 | 800 | 17.38% | 142 | 100 | 42 |
| **450 + anatomy + skip（本次）** | **900** | **7187** | **29.78%** | **2190** | **1770** | **420** |

## 4. 结果解读

1. `r2_skip_bilateral` 在大规模上仍然有效：  
`R2` 维持在较低水平（`420 / 7187 = 5.84%`），说明 bilateral 结构性误判被明显抑制。

2. 当前主瓶颈已转移到 `R1_LATERALITY`：  
`R1` 占到总规则违规约 80%，是后续优化的第一优先级。

3. 50-case 到 450/450 有明显分布差异：  
50-case 的 `17.38%` 在全量上升到 `29.78%`，说明小样本结果偏乐观，存在泛化差距。

4. R2 仍有残留集中点：  
`anatomy_r2_breakdown.csv` 显示本次 R2 主要来自 `mediastinum`（`420`）。

## 5. 下一步建议（按优先级）

### P0：先攻 R1（最高收益）

1. laterality 触发前增加“显式侧别词”门槛（无 left/right 时不强触发）。  
2. 对中线/非侧别解剖词（如 mediastinum 等）加 R1 豁免或降权。  
3. 对否定句中的侧别匹配单独处理，减少误触发。

### P1：收尾 R2（集中在 mediastinum）

1. 扩充 mediastinum 词表映射与同义词。  
2. 仅对 mediastinum 句子做定向阈值校准（避免全局放松）。

### P2：验证策略

1. 继续保留两条实验线：  
`strict`（不 skip）用于能力评估；`relaxed`（skip bilateral）用于当前工程可交付。  
2. 每次只改一个规则点，先做 50-case 回归，再上 450/450 复核。

## 6. 本次可交付文件

- `summary.csv`  
`outputs_stage0_4_follow_request_20260307_1705/outputs_stage0_4_450_128/summary.csv`
- `run_meta.json`  
`outputs_stage0_4_follow_request_20260307_1705/outputs_stage0_4_450_128/run_meta.json`
- `validation_report.json`  
`outputs_stage0_4_follow_request_20260307_1705/outputs_stage0_4_450_128/validation_report.json`
- `analysis_exports/`  
`outputs_stage0_4_follow_request_20260307_1705/outputs_stage0_4_450_128/analysis_exports/`
- `OUTPUT_ANALYSIS_COLAB.executed.ipynb`  
`outputs_stage0_4_follow_request_20260307_1705/outputs_stage0_4_450_128/OUTPUT_ANALYSIS_COLAB.executed.ipynb`
