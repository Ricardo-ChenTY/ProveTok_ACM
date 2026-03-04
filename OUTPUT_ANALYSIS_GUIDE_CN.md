# Stage0-4 实验结果说明（最新覆盖版）

更新日期：2026-03-04  
范围：CT-RATE 450 + RadGenome 450（共 900 例）  
口径：Stage0-4（无 LLM 生成）

---

## 1. 本轮产出是什么

本轮分析基于输出目录中的聚合文件：

- `dataset_aggregate.csv`
- `sentence_violation_rate.csv`
- `rule_violation_count.csv`
- `abnormal_cases_ranked.csv`

这些文件用于做“流程是否成立 + 误差分布诊断”，不是最终论文结果。

---

## 2. 关键结果（450/450）

### 2.1 数据集级统计

- `ctrate`: 450 例，平均句子数 7.99，平均违规数 6.93
- `radgenome`: 450 例，平均句子数 7.98，平均违规数 7.43

### 2.2 句级违规率

- `ctrate`: 3596 句中 2630 句有违规，违规率 `0.731`
- `radgenome`: 3591 句中 2555 句有违规，违规率 `0.712`
- 总体：7187 句中 5185 句有违规，总体约 `0.721`

### 2.3 规则贡献（按违规条数）

- `R2_ANATOMY`: 2651（最高）
- `R1_LATERALITY`: 2009
- `R5_NEGATION`: 925
- `R4_SIZE`: 879
- `R3_DEPTH`: 0（当前基本未触发）

### 2.4 异常 case 分布

- `violation_ratio = 1.0` 的 case：54 / 900
- `n_violation_sentences >= 7` 的 case：283 / 900
- `violation_ratio` 中位数约 0.75，四分位上界约 0.875

---

## 3. 结果解读（当前阶段）

1. 流程已跑通：
- 450/450 规模完成，可做系统性误差定位。

2. 当前质量属于“可诊断、待优化”：
- 句级违规率 > 0.70，偏高。
- 主要矛盾集中在 `R2` 和 `R1`。

3. 优先优化顺序：
- 先 `R2_ANATOMY`（最大头）
- 再 `R1_LATERALITY`
- 然后修 `R5_NEGATION`
- 同时补 `R3_DEPTH` 触发覆盖

---

## 4. 下一步实验方案（已接入脚本）

### 4.1 先做 50/50 的 R2 sweep

目标：只调整 `r2_min_support_ratio`，观察是否能显著降低总体违规率且不引入副作用。

推荐扫描：

- `1.0`（当前严格基线）
- `0.8`
- `0.6`

Colab 一键脚本：

```bash
bash Scripts/run_r2_sweep_50_cp_strict_colab.sh
```

汇总脚本：

```bash
python Scripts/summarize_r2_sweep.py \
  --sweep_root /content/drive/MyDrive/Data/outputs_stage0_4_r2sweep_50_cp_strict \
  --save_csv /content/drive/MyDrive/Data/outputs_stage0_4_r2sweep_50_cp_strict/sweep_summary.csv
```

### 4.2 决策门槛

从 sweep 里选择进入 450/450 的配置时，建议满足：

1. `violation_sentence_rate` 明显下降
2. `R2` 下降明显
3. `R1/R5` 不出现爆发式上升
4. `validation_report` 结构验收通过

---

## 5. 给协作同学的最小交付物

重跑后至少交：

- `summary.csv`
- `ctrate_case_summary.csv`
- `radgenome_case_summary.csv`
- `run_meta.json`
- `validation_report.json`
- `cases/*/*/trace.jsonl`

如果是 strict 重跑，额外保留：

- `ckpt_probe_report.json`

---

## 6. 当前结论（一句话）

这版 450/450 结果证明了 Stage0-4 流程可运行并可诊断，但违规率仍高；下一步应先用 R2 sweep 在小规模（50/50）上选稳态参数，再回到 450/450 复核。
