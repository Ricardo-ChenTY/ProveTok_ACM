# 50-case 三次对比分析与降违规方案（更新版）

更新时间：2026-03-07  
范围：CT-RATE 50 + RadGenome 50（共 100 case）

## 1. 对比对象

1. 上上次（baseline）：`2026-03-06 22:50:10`  
路径：`outputs_stage0_4_follow_request_20260306_225010/r2_taut005_ratio_0.8_nor4r5`
2. 上次：`2026-03-07 13:44`  
路径：`outputs_stage0_4_follow_request_20260307_1344/r2_taut005_ratio_0.8_nor4r5`
3. 这次：`2026-03-07 14:10`  
路径：`outputs_stage0_4_follow_request_20260307_1410/r2_taut005_ratio_0.8_nor4r5`

说明：三次均为 50/50，核心参数一致；上次和这次均开启 `--anatomy_spatial_routing`。

## 2. 关键结论

1. 相比上上次（baseline），当前方案已稳定下降：总体句级违规率 `45.75% -> 42.00%`（`-3.75` pct）。
2. 上次与这次结果完全一致（复现通过）：`336/800` 违规句，`R2=239`，`R1=100`。
3. 当前主要瓶颈依然集中在 `R2_ANATOMY`（尤其 `bilateral`、`mediastinum`），其次是 `R1_LATERALITY`。

## 3. 三次核心指标对比

| Run | Violation Sentences | Violation Sentence Rate | Total Violations | R2_ANATOMY | R1_LATERALITY |
|---|---:|---:|---:|---:|---:|
| 上上次 | 366 / 800 | 45.75% | 388 | 269 | 119 |
| 上次 | 336 / 800 | 42.00% | 339 | 239 | 100 |
| 这次 | 336 / 800 | 42.00% | 339 | 239 | 100 |

这次 vs 上次：无差异。  
这次 vs 上上次：违规句 `-30`，总违规 `-49`，R2 `-30`，R1 `-19`。

## 4. 分数据集结果（这次）

- `ctrate`：`186/400 = 46.5%`
- `radgenome`：`150/400 = 37.5%`

## 5. case 分布变化（上上次 -> 这次）

- `violation_ratio == 1.0`：`0 -> 0`
- `violation_ratio >= 0.75`：`9 -> 5`
- 中位数：`0.500 -> 0.375`
- Q3：`0.625 -> 0.500`

说明：高违规 case 占比继续收缩，但仍有一批 `0.625~0.75` 的高风险 case。

## 6. anatomy 热点（这次）

`anatomy_r2_breakdown.csv`：

- `bilateral`: 197
- `mediastinum`: 42

`anatomy_all_violation_rate.csv`：

- `bilateral`: `197/197 = 1.0`
- `mediastinum`: `42/42 = 1.0`
- 其余关键词（left/right upper/lower lobe）本批次违规率接近 `0`

结论：`bilateral` 和 `mediastinum` 是当前 R2 的核心“硬点”。

## 7. 降违规建议（按优先级）

### P0（应先做，直接针对主要误差源）

1. `bilateral` 改为“并侧支持”判定，不再按单侧 anatomy 约束。  
建议：R2 对 `bilateral` 采用 `support(left OR right)` 逻辑，避免 top-k 全部被判 0 支持。
2. `mediastinum` 增加 anatomy 映射覆盖与 bbox 匹配容差。  
建议：扩充 mediastinum 词表同义项，并放宽其 IoU/支持比阈值（仅对该关键词生效）。

### P1（第二步，控制副作用）

1. 对 `R1_LATERALITY` 增加“无显式侧别词”豁免。  
建议：句子未出现 left/right 且 anatomy 为中线/双侧时，不触发强 laterality mismatch。
2. 对 anatomy 句子引入“关键词分组阈值”。  
建议：`r2_min_support_ratio` 采用分组阈值，例如 `bilateral/mediastinum` 用较松阈值，其余保持 0.8。

### P2（验证策略）

1. 先做 50-case 局部回归（固定数据、固定随机种子），观察 R2 和总体是否继续下降。
2. 若句级违规率可降到 `<=40%` 且 R1 不反弹，再进入 450/450 全量复核。

## 8. 本次产物路径（这次 run）

- `summary.csv`  
`outputs_stage0_4_follow_request_20260307_1410/r2_taut005_ratio_0.8_nor4r5/summary.csv`
- `validation_report.json`  
`outputs_stage0_4_follow_request_20260307_1410/r2_taut005_ratio_0.8_nor4r5/validation_report.json`
- `analysis_exports/`  
`outputs_stage0_4_follow_request_20260307_1410/r2_taut005_ratio_0.8_nor4r5/analysis_exports/`
