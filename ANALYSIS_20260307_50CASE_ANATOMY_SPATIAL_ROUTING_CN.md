# 50-case 对比分析与下一步建议（更新版）

更新日期：2026-03-07  
范围：CT-RATE 50 + RadGenome 50（共 100 case）

## 1. 对比对象

1. Baseline（2026-03-06 22:50:10）  
路径：`outputs_stage0_4_follow_request_20260306_225010/r2_taut005_ratio_0.8_nor4r5`
2. Run-A（2026-03-07 13:44）  
路径：`outputs_stage0_4_follow_request_20260307_1344/r2_taut005_ratio_0.8_nor4r5`
3. Run-B（2026-03-07 14:10）  
路径：`outputs_stage0_4_follow_request_20260307_1410/r2_taut005_ratio_0.8_nor4r5`
4. Run-C（2026-03-07 15:00，新增 `--r2_skip_bilateral`）  
路径：`outputs_stage0_4_follow_request_20260307_1500/r2_taut005_ratio_0.8_nor4r5_skip_bilateral`

## 2. 总结结论

1. Run-A/Run-B 完全一致，说明在同参数下复现稳定。  
2. Run-C（加 `--r2_skip_bilateral`）相对 Run-B 明显下降：总体句级违规率从 `42.00%` 降到 `17.375%`。  
3. 但 Run-C 与前面 run 的 `R2` 口径不再完全等价（因为跳过了 bilateral 句子的 R2 判定），解释结果时应单独标注。

## 3. 四次核心指标对比

| Run | Violation Sentences | Violation Sentence Rate | Total Violations | R2_ANATOMY | R1_LATERALITY |
|---|---:|---:|---:|---:|---:|
| Baseline | 366 / 800 | 45.75% | 388 | 269 | 119 |
| Run-A | 336 / 800 | 42.00% | 339 | 239 | 100 |
| Run-B | 336 / 800 | 42.00% | 339 | 239 | 100 |
| Run-C (`r2_skip_bilateral`) | 139 / 800 | 17.375% | 142 | 42 | 100 |

Run-B -> Run-C 变化：
- 违规句率：`42.00% -> 17.375%`（`-24.625` pct）
- 总违规条数：`339 -> 142`（`-197`）
- `R2_ANATOMY`：`239 -> 42`（`-197`）
- `R1_LATERALITY`：`100 -> 100`（不变）

## 4. Run-C 分数据集结果

- `ctrate`: `90 / 400 = 22.50%`
- `radgenome`: `52 / 400 = 13.00%`

## 5. 结果解释（重要）

`--r2_skip_bilateral` 的效果本质是“对 bilateral 句子不计 R2”，因此可大幅降低当前 R2 主导的违规数。  
这对工程稳定性有价值，但和“改进模型理解能力”的意义不同；建议在报告和后续实验中将该配置标记为 `policy-relaxed`。

## 6. 下一步建议

1. 保留两条线并行：  
`strict线`：不加 `r2_skip_bilateral`，用于真实能力评估；  
`relaxed线`：加 `r2_skip_bilateral`，用于当前版本可交付结果。
2. 若目标是继续降 strict 线：优先处理 `mediastinum` 与 laterality 误触发（R1）。
3. 进入 450/450 前，建议先在 50-case 再做一轮最小 sweep（只动 1 个参数），确认 R1 不反弹。

## 7. Run-C 关键产物

- `summary.csv`  
`outputs_stage0_4_follow_request_20260307_1500/r2_taut005_ratio_0.8_nor4r5_skip_bilateral/summary.csv`
- `run_meta.json`  
`outputs_stage0_4_follow_request_20260307_1500/r2_taut005_ratio_0.8_nor4r5_skip_bilateral/run_meta.json`
- `validation_report.json`  
`outputs_stage0_4_follow_request_20260307_1500/r2_taut005_ratio_0.8_nor4r5_skip_bilateral/validation_report.json`
