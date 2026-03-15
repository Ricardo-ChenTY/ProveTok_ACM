# ProveTok v3.0 实现状态 vs Overleaf 数学规范

更新日期：2026-03-14

本文档逐公式对照 `CP_priorjudgement.pdf`（Overleaf 编译版）与代码实现，
标注已完成、可接受差异、以及尚未对齐的部分。

---

## 1. M1 — Stage 0-2：证据 Token 化与预算约束

### 已完成

| Overleaf 公式 | 描述 | 对应代码 | 状态 |
|-------------|------|---------|------|
| Eq(1) | 伪影风险聚合 `A_i = clip(w_snr·norm(σ/(|μ|+ε)) + w_st·norm(streak) + w_out·norm(outlier), 0, 1)` | `stage0_artifacts.py:compute_artifact_components()` | 完全对齐 |
| Eq(2) | 伪影门控 `g_i = σ(-k(A_i - τ_A))` | `stage0_2.py:soft_gate()` | 完全对齐 |
| Eq(3) | 重要性分数 `S_i = g_i·(λ_h·q(H_i) + λ_p·q(P_i))` | `stage0_2.py:importance_score()` | 完全对齐 |
| Eq(4) | 边界缓存 `b_i = mean_{j∈N(i)} f_j` | `stage0_2.py:boundary_context_blend()` | 完全对齐 |
| Eq(5) | 特征混合 `f_export = (1-β)f_i + β·b_i` | 同上 | 完全对齐 |
| — | 分位数归一化 `q(·)` 按病例/深度层 | `math_utils.py:quantile_rank()` + `stage2_octree_splitter.py` 按 level 分组 | 完全对齐 |
| — | 邻域为空时 fallback `b_i = f_i` | `boundary_context_blend()` 检查 `not neighbor_features` → 返回原特征 | 完全对齐 |
| §2.6 | norm(·) 用 min-max 归一化 | `stage0_artifacts.py:_minmax_norm()` | 完全对齐 |
| — | 自适应八叉树分裂 + NMS + 预算 B | `stage2_octree_splitter.py:AdaptiveOctreeSplitter` | 完全对齐 |

### 无缺口

---

## 2. M2 — Stage 3a/3b：粗证据规划 + 轻量路由器

### 已完成

| Overleaf 公式 | 描述 | 对应代码 | 状态 |
|-------------|------|---------|------|
| Eq(6) | 投影归一化 `v_i = W_proj·f_export / ‖W_proj·f_export‖₂` | `stage3_router.py:_projected_token()` — `matvec()` + `normalize_l2()` | 完全对齐 |
| Eq(7) | 路由分数 `r_i = cos(q_s, v_i) = q_s^T v_i` | `_routing_score()` — `dot(q, v)` 两边已 L2 归一化 | 完全对齐 |
| Eq(8) | 空间先验 `r̂_i = r_i + λ_spatial·IoU(bbox_i, bbox_anatomy)` | `_routing_score()` 第 54 行 | 完全对齐 |
| Eq(9) | InfoNCE 多正样本训练损失 | `infonce_loss()` — log-sum-exp 稳定化，|P_s|=0 时 raise | 完全对齐 |
| §3.4 | 规划预算 `B_plan = min(32, B/4)` | `config.py:RouterConfig.planning_budget()` | 完全对齐 |
| §4.7 | 推理时 top-k 选择，k=8 | `RouterConfig.k_per_sentence = 8` | 完全对齐 |

### 已知差异（可接受）

| 项目 | Overleaf | 实现 | 说明 |
|------|---------|------|------|
| W_proj 训练状态 | 需 InfoNCE 训练 | 当前 fallback 到 identity 矩阵 | `_ensure_w_proj()` 自动构建单位矩阵；`anatomy_spatial_routing=True` 模式下用 IoU 主导路由绕过 W_proj 依赖。训练脚本 `train_wprojection.py` 已存在，等数据充足后启用 |

---

## 3. M3 — Stage 3c：逐句 Token-Gated 生成

### 已完成

| Overleaf 规范 | 描述 | 对应代码 | 状态 |
|-------------|------|---------|------|
| Eq(10) | 引用正确性 `Ê_s = E_s`（生成时只看 E_s 中的 token） | `stage3c_generator.py` — citations 恒等于 routed token IDs | 完全对齐 |
| §5.2 | 输入：证据集合 E_s + 前文已生成句子 | `_build_generation_prompt()` 接收 `history` 列表 | 完全对齐 |
| §5.7 | 句间历史上下文避免依赖断裂 | `stage0_4_runner.py` 维护 `generated_history` 累积传入 | 完全对齐 |
| §5.8 | 日志记录每句证据 ID | trace.jsonl 中 `topk_token_ids` 字段 | 完全对齐 |

### C_attn 注意力代理公式 — 已决定：改 Overleaf 对齐代码

代码使用 `C_attn = Σ_s k·B`（k=top-k 数，B=token bank 大小）。
Overleaf 原 Eq(11-12) 基于实际 prompt/gen token 数，依赖具体 LLM tokenizer，换模型数字就变。
`k·B` 只取决于系统架构超参，跨模型可比，更适合当 cost proxy。

**决定**：更新 Overleaf §5.5 将 C_attn 定义改为 `Σ_s k·B`，原 Eq(11-12) 移至附录作理论参考。

---

## 4. M4 — Stage 4-5：规则校验与重路由

### 已完成

| Overleaf 规范 | 描述 | 对应代码 | 状态 |
|-------------|------|---------|------|
| Eq(13) | 规则校验 `R(y_s, anatomy)` | `stage4_verifier.py:Verifier.audit_sentence()` | 完全对齐 |
| R1 | 左右侧一致性（center x vs midline） | R1_LATERALITY — 支持 bilateral、tolerance、midline 关键词跳过 | 完全对齐 |
| R2 | 解剖 IoU 一致性（τ_IoU 阈值） | R2_ANATOMY — 支持 max-IoU 和 support-ratio 两种模式 | 完全对齐 |
| R3 | 深度层一致性 | R3_DEPTH — level range 检查 | 完全对齐 |
| R4 | 尺寸/体积范围 | R4_SIZE — union bbox volume 检查 | 完全对齐 |
| R5 | 否定处理 | R5_NEGATION — metadata conflict + lexicon fallback | 完全对齐 |
| — | 跨句一致性 | R6a (laterality conflict) + R6b (presence/absence conflict) | **超出 Overleaf 范围**（增强） |
| §6.5 | LLM Judge 二次裁决 | `stage5_llm_judge.py:LLMJudge` — 支持 4 种后端 | 完全对齐 |
| — | 重路由后再生成 + 再校验 | `stage0_4_runner.py` 第 166-197 行 | 完全对齐 |
| — | De-specify 降级（去除空间定位词后重生成） | `stage3c_generator.py:despecify_text()` | 完全对齐 |
| §6.7 | 重路由最多一次 | `cfg.reroute.max_retry = 1` | 完全对齐 |

### 已知差异 — 惩罚公式

| Overleaf 公式 | 描述 | 当前实现 | 差距 |
|-------------|------|---------|------|
| Eq(14) | `r̂_new = r̂ - α·sev_i`（加法惩罚） | `reroute_scores()` 用 `S' = S*(1-α*sev)`（乘法惩罚） | 形式不同但效果等价；**主 pipeline 已使用 log-smooth，此函数仅作 legacy fallback** |

**主 pipeline 实际使用的惩罚**：`reroute_scores_log_smooth()` — `r'_i = r_i - γ·ln(1 + sev_i)`

这个 log-smooth 形式在 Overleaf 中没有显式出现（Overleaf 用的是线性 Eq(14)），
但这是导师建议的改进版本，实测效果更平滑。

**建议**：如果需要与 Overleaf 完全对齐，可以将主 pipeline 切回 Eq(14) 的加法形式：
```python
# 加法形式 Eq(14)
result[tid] = s - alpha * sev_i
```
或在 Overleaf 中更新公式为 log-smooth 版本。

### 每-token vs 全句惩罚

| 维度 | Overleaf | 实现 |
|------|---------|------|
| 惩罚粒度 | Eq(14) 按 token 惩罚 `sev_i` | `reroute_scores_log_smooth()` 支持 per-token 模式（通过 `violations` 参数定位 token_ids） |
| 无 violation token_ids 时 | 未说明 | fallback 到全句 uniform penalty |

---

## 5. M5 — Stage 6：输出与统计评估

### 已完成

| Overleaf 公式 | 描述 | 对应代码 | 状态 |
|-------------|------|---------|------|
| Eq(15-16) | Bootstrap CI（病人级，R=5000） | `analyze_outputs.py:_bootstrap_ci()` — `R=5000`, percentile method | 完全对齐 |
| §7.4.2 | Holm step-down 多重检验校正 | `_holm_correction()` — 升序排列，`p_adj = p*(m-k)`，强制单调 | 完全对齐 |
| §7.3 | 各项指标 + 置信区间 + p 值 | `analyze_m5_protocol()` — violation rate CI + C_LLM CI + Holm | 完全对齐 |
| §7.6 | 抽样单位 = 病人 | `_bootstrap_ci()` 按 case 抽样 | 完全对齐 |

### C_attn — 已决定改 Overleaf

| 项目 | 实现 | 状态 |
|------|------|------|
| C_LLM | `C_LLM = n_gen + n_judge`（含 regen/despec 额外调用） | 完全对齐 |
| C_attn | `C_attn = Σ_s k·B`（架构级 proxy，跨模型可比） | 待更新 Overleaf |

---

## 总结：未对齐清单

### 需在 Overleaf 端更新

| # | 项目 | 操作 |
|---|------|------|
| 1 | **C_attn 公式** — Overleaf Eq(11-12) 改为 `C_attn = Σ_s k·B` | 更新 §5.5，原公式移附录 |
| 2 | **惩罚公式** — Overleaf Eq(14) `r̂ - α·sev` vs 实现 `r - γ·ln(1+sev)` | 主 pipeline 用 log-smooth（导师建议改进）；建议更新 Overleaf |

### 代码侧已知差异（不影响）

| # | 差异 | 说明 |
|---|------|------|
| 3 | **`reroute_scores()` legacy** 用乘法 `S*(1-α*sev)` | 不影响 — 主 pipeline 不调用此函数 |
| 4 | **W_proj 未训练** — identity fallback + spatial routing 绕过 | 等实验数据充足后训练；不影响 workflow 完整性 |
| 5 | **R6 跨句规则** — 实现中有但 Overleaf 未写 | 属于增强，可后续加入 Overleaf 或保持 code-only |

### 完全对齐（无需修改）

- M1 全部公式 Eq(1-5) + 归一化 + 邻域 fallback
- M2 全部公式 Eq(6-9) + top-k + B_plan
- M3 引用正确性 Eq(10) + 句间历史
- M4 规则 R1-R5 + LLM Judge + de-specify + max_retry=1
- M5 Bootstrap CI Eq(15-16) + Holm 校正 + C_LLM

---

## 文件清单

| 文件 | 对应 Module | 关键函数/类 |
|------|-----------|-----------|
| `stage0_scorer.py` | M1 Stage 0 | `DeterministicArtifactScorer` |
| `stage0_artifacts.py` | M1 Stage 0 | `compute_artifact_components()`, `_minmax_norm()` |
| `stage0_2.py` | M1 Stage 0-2 | `artifact_risk_score()`, `soft_gate()`, `importance_score()`, `boundary_context_blend()` |
| `math_utils.py` | 通用 | `sigmoid()`, `quantile_rank()`, `normalize_l2()`, `dot()` |
| `stage1_swinunetr_encoder.py` | M1 Stage 1 | `FrozenSwinUNETREncoder` |
| `stage2_octree_splitter.py` | M1 Stage 2 | `AdaptiveOctreeSplitter` |
| `simple_modules.py` | M2 Stage 3a | `ReportSentencePlanner`, `RuleBasedAnatomyResolver` |
| `stage3_router.py` | M2 Stage 3b | `Router`, `infonce_loss()` |
| `stage3c_generator.py` | M3 Stage 3c | `Stage3cGenerator`, `despecify_text()` |
| `stage4_verifier.py` | M4 Stage 4 | `Verifier` (R1-R6) |
| `stage5_llm_judge.py` | M4 Stage 5 | `LLMJudge`, `reroute_scores_log_smooth()` |
| `stage0_4_runner.py` | 全流程 | `run_case_stage0_4()` — 编排 Stage 0→5 |
| `config.py` | 全局配置 | `ProveTokConfig` (split/router/verifier/reroute/llm_judge) |
| `types.py` | 数据类型 | `BBox3D`, `EvidenceToken`, `SentenceOutput`, `RuleViolation` |
| `analyze_outputs.py` | M5 | `analyze_m5_protocol()`, `_bootstrap_ci()`, `_holm_correction()` |
| `run_mini_experiment.py` | CLI 入口 | 命令行参数 + 组件实例化 |
| `train_wprojection.py` | M2 训练 | InfoNCE W_proj 训练脚本 |
