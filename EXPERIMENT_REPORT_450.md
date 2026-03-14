# ProveTok v3.0 — 450/450 实验报告

> 实验日期：2026-03-13
> 运行时长：~4.2 小时（250.8 分钟）
> 输出目录：`outputs/stage0_5_llama_450/`

---

## 1. 数据概况

### 1.1 数据来源

从 HuggingFace 上两个公开数据集中各随机采样 900 个 CT volume（`.nii.gz`），再通过 `shuffle_seed=42` 从 900 中选取 450 个参与实验。

| 数据集 | HuggingFace Repo | 总可用 volumes | 采样池 | 实验选取 |
|--------|-------------------|-------------|--------|---------|
| **CT-RATE** | `ibrahimhamamci/CT-RATE` | 50,188 | 900 | 450 |
| **RadGenome-ChestCT** | `ibrahimhamamci/RadGenome-ChestCT` | 25,692 | 900 | 450 |

### 1.2 数据 Split 分布（900 采样池）

| 数据集 | train | valid |
|--------|-------|-------|
| CT-RATE | 845 | 55 |
| RadGenome | 839 | 61 |

### 1.3 报告文本来源

- **CT-RATE**：`dataset/radiology_text_reports/{train,validation}_reports.csv`，使用 `Findings_EN` 列
- **RadGenome**：`dataset/radgenome_files/{train,validation}_region_report.csv`，逐句格式（`Sentence` 列），按 `Volumename` 聚合为整段报告

### 1.4 Manifest 文件

- 采样源：`manifests/ctrate_900_source.csv`、`manifests/radgenome_900_source.csv`
- 下载 manifest：`manifests/ctrate_900_manifest.csv`、`manifests/radgenome_900_manifest.csv`
- 格式：`case_id, volume_path, report_text, split, volume_name`

---

## 2. 实验配置

### 2.1 核心超参数

| 超参数 | 实验值 | 论文默认值 | 备注 |
|--------|-------|---------|------|
| Token 预算 B | **128** | 64 | 高于论文默认，提供更多证据 |
| 每句 top-k | **8** | 8 | 与论文一致 |
| 空间先验 λ_spatial | **0.3** | 0.3 | 与论文一致 |
| IoU 阈值 τ_IoU | **0.04** | 0.1 | 经 mediastinum sweep 优化后降低 |
| 边界混合 β | **0.1** | 0.1 | 与论文一致 |
| Shuffle seed | **42** | — | 保证可复现 |

### 2.2 规则校验器配置（Stage 4）

| 规则 | 状态 | 配置详情 |
|------|------|---------|
| **R1 — 侧别一致性** | 启用 | `r1_min_same_side_ratio=0.6`, `r1_negation_exempt=true`, `r1_skip_midline=true` |
| **R2 — 解剖区域一致性** | 启用 | `r2_mode=ratio`, `r2_min_support_ratio=0.8`, `r2_skip_bilateral=true`, `anatomy_spatial_routing=true` |
| **R3 — 深度层级一致性** | 启用 | 默认配置 |
| **R4 — 大小/范围合理性** | **禁用** | `r4_disabled=true` |
| **R5 — 否定处理** | 启用（部分） | `r5_fallback_disabled=true`（禁用回退词典） |

### 2.3 LLM Judge 配置（Stage 5）

| 参数 | 值 |
|------|------|
| 后端 | HuggingFace (本地推理) |
| 模型 | `Llama-3.1-8B-Instruct` |
| 精度 | bfloat16 |
| Alpha (混合权重) | 0.5 |

### 2.4 编码器与文本模型

| 组件 | 模型 | 设备 |
|------|------|------|
| 3D 编码器 (Stage 1) | SwinUNETR (`swinunetr.ckpt`) | CUDA |
| 文本编码器 (Router) | `sentence-transformers/all-MiniLM-L6-v2` | CUDA |
| 输入体积尺寸 | 128 × 128 × 128 | — |

---

## 3. 实验结果

### 3.1 总览

| 指标 | CT-RATE (450) | RadGenome (450) | 合计 (900) |
|------|--------------|----------------|-----------|
| Avg tokens/case | 128.0 | 128.0 | 128.0 |
| Avg sentences/case | 7.9 | 8.0 | 8.0 |
| **Stage 0-4 violations** | **529** | **445** | **974** |
| Cases with violations | 332 (73.8%) | 290 (64.4%) | 622 (69.1%) |
| Avg violations/case | 1.18 | 0.99 | 1.08 |
| **Judge confirmed** | **229** | **220** | **449** |
| Cases with judge confirmed | 207 (46.0%) | 167 (37.1%) | 374 (41.6%) |
| **Judge filter rate** | **56.7%** | **50.6%** | **53.9%** |
| Zero-violation cases | 118 (26.2%) | 160 (35.6%) | 278 (30.9%) |
| Zero-judge cases | 243 (54.0%) | 283 (62.9%) | 526 (58.4%) |

### 3.2 Violation 分布

**CT-RATE:**
| Violations/case | 0 | 1 | 2 | 3 | 4 |
|-----------------|---|---|---|---|---|
| Cases | 118 | 183 | 107 | 36 | 6 |
| 占比 | 26.2% | 40.7% | 23.8% | 8.0% | 1.3% |

**RadGenome:**
| Violations/case | 0 | 1 | 2 | 3 | 4 |
|-----------------|---|---|---|---|---|
| Cases | 160 | 166 | 97 | 23 | 4 |
| 占比 | 35.6% | 36.9% | 21.6% | 5.1% | 0.9% |

### 3.3 跨规模对比

| 指标 | 50/50 (B=64) | 200/200 (B=128) | **450/450 (B=128)** |
|------|-------------|----------------|---------------------|
| **CT-RATE** ||||
| Violation rate | 70.0% | 60.0% | **73.8%** |
| Judge confirmed rate | 8.0% | 29.5% | **46.0%** |
| Judge filter rate | 91.5% | 66.1% | **56.7%** |
| **RadGenome** ||||
| Violation rate | 64.0% | 64.5% | **64.4%** |
| Judge confirmed rate | 16.0% | 39.5% | **37.1%** |
| Judge filter rate | 81.4% | 48.1% | **50.6%** |
| **运行时间** | 15.9 min | 110.3 min | **250.8 min** |

### 3.4 关键观察

1. **RadGenome 表现稳定**：violation rate 在 200/200 和 450/450 中几乎一致（64.5% vs 64.4%），Judge confirmed rate 也接近（39.5% vs 37.1%），说明参数配置在不同数据子集上具有良好的泛化性。

2. **CT-RATE violation rate 偏高**（73.8% vs 200/200 的 60.0%）：新 900 采样池引入了更多样的 case，部分 case 可能涉及更复杂的解剖描述，导致更多 rule violation。

3. **Judge filter rate 趋于稳定**：200/200 和 450/450 的 filter rate 接近（CT-RATE: 66.1% → 56.7%, RadGenome: 48.1% → 50.6%），50/50 的 91.5% 是因为使用了 B=64（token 少，violations 简单）。

4. **验证 100% 通过**：所有 900 个 case 的输出结构验证全部通过，无格式/完整性错误。

---

## 4. 流水线实现状态

对照论文方法 M1-M5，标注当前代码实现情况：

### 4.1 M1: 预算约束的 3D 证据 Token Bank (Stage 0-2)

| 组件 | 论文描述 | 实现状态 | 代码文件 |
|------|--------|---------|---------|
| Stage 0: 伪影风险评分 Ai | 逆 SNR + 条纹 + 离群值加权 | ✅ 已实现 | `stage0_scorer.py`, `stage0_artifacts.py` |
| Sigmoid 软门控 gi | σ(-k_gate(Ai - τA)) | ✅ 已实现 | `stage0_scorer.py` |
| Stage 1: 冻结 3D 编码器 | SwinUNETR, 权重冻结 | ✅ 已实现 | `stage1_swinunetr_encoder.py` |
| Stage 2: 分位数归一化 | rank/n 按深度层、按病例 | ✅ 已实现 | `stage2_octree_splitter.py` |
| Stage 2: 综合评分 Si | gi · (λh q(Hi) + λp q(Pi)) | ✅ 已实现 | `stage2_octree_splitter.py` |
| 自适应八叉树分裂 | 迭代分裂 + top-B 裁剪 | ✅ 已实现 | `stage2_octree_splitter.py` |
| 边界感知导出 | (1-β)fi + β·bi 凸组合 | ✅ 已实现 | `stage2_octree_splitter.py` |

### 4.2 M2: 轻量跨模态路由器 (Stage 3b)

| 组件 | 论文描述 | 实现状态 | 代码文件 |
|------|--------|---------|---------|
| 线性投影 + L2 归一化 | Wproj fi / ‖Wproj fi‖ | ✅ 已实现 | `stage3_router.py` |
| 余弦路由分数 | cos(qs, vi) | ✅ 已实现 | `stage3_router.py` |
| 空间先验增强 | r + λ_spatial · IoU | ✅ 已实现 | `stage3_router.py` |
| Top-k 选取 | 取 top-k 构成证据集 Es | ✅ 已实现 | `stage3_router.py` |
| InfoNCE 训练损失 | 温度缩放 softmax CE | ⬜ **未实现（无训练循环）** | — |

> **注**：当前 Router 使用文本编码器 (all-MiniLM-L6-v2) 的冻结嵌入做余弦路由，Wproj 未经 InfoNCE 训练。论文中 InfoNCE 损失的完整推导（§5.2）在数学上已完备，但代码中尚未实现训练循环。

### 4.3 M3: 逐句 Token 门控生成 (Stage 3a/3c)

| 组件 | 论文描述 | 实现状态 | 代码文件 |
|------|--------|---------|---------|
| Stage 3a: 证据规划 | 粗 token → 句子主题列表 | ✅ 已实现 | `simple_modules.py` |
| Stage 3b: 逐句路由 | 每句 top-k 路由 | ✅ 已实现 | `stage3_router.py` |
| Stage 3c: Token 门控 LLM 生成 | 逐句 LLM 调用 | ⬜ **未实现** | `stage3c_generator.py`（存在但未集成） |
| 构造性引用 Ê_t ≡ E_t | 接口正确性保证 | ⬜ **未实现（依赖 Stage 3c）** | — |

> **注**：当前流水线不做实际 LLM 文本生成（Stage 3c）。实验验证的是 Token Bank → Router → Verifier → LLM Judge 路径的规则校验能力，而非端到端报告生成。

### 4.4 M4: 规则校验器与重路由 (Stage 4-5)

| 组件 | 论文描述 | 实现状态 | 代码文件 |
|------|--------|---------|---------|
| R1 — 侧别一致性 | bbox 中心 vs 中线 | ✅ 已实现 | `stage4_verifier.py` |
| R2 — 解剖区域一致性 | IoU > τ_anatomy | ✅ 已实现 | `stage4_verifier.py` |
| R3 — 深度层级一致性 | 八叉树深度范围检查 | ✅ 已实现 | `stage4_verifier.py` |
| R4 — 大小/范围合理性 | 联合 bbox vs 典型大小 | ⚠️ 已实现但**禁用** | `stage4_verifier.py` (`r4_disabled=true`) |
| R5 — 否定处理 | 否定检测 + 逻辑取反 | ⚠️ 部分启用 | `stage4_verifier.py` (`r5_fallback_disabled=true`) |
| Stage 5: 对数平滑惩罚 | r' = r - γ·ln(1 + sev) | ⬜ **未实现** | — |
| Stage 5: 重路由协议 | 更新路由 → 重新 top-k → 重新生成 | ⬜ **未实现** | — |
| Stage 5: LLM Judge | Llama-3.1-8B 二次判定 | ✅ 已实现 | `stage5_llm_judge.py` |

> **注**：论文 Stage 5 原意是「对数平滑惩罚 + 重路由 + 重新生成」，当前实现的 Stage 5 是 LLM Judge 对 violation 做二次确认/过滤，而非原论文的重路由机制。

### 4.5 M5: 匹配计算量评估与统计协议

| 组件 | 论文描述 | 实现状态 |
|------|--------|---------|
| LLM Token 代价 C_LLM | Σ(|p| + |g|) | ⬜ **未实现** |
| 注意力代价代理 (KV-Cache) | p² + gp + g(g-1)/2 | ⬜ **未实现** |
| 病人级配对 Bootstrap CI | 有放回重采样 5000 次 | ⬜ **未实现** |
| Holm 步降多重比较校正 | FWER 控制 | ⬜ **未实现** |

### 4.6 实现总结

| 类别 | 已实现 | 部分/禁用 | 未实现 |
|------|--------|---------|--------|
| M1 (Token Bank) | 7/7 | — | — |
| M2 (Router) | 4/5 | — | InfoNCE 训练 |
| M3 (门控生成) | 2/4 | — | Stage 3c 生成, 构造性引用 |
| M4 (校验+重路由) | 4/7 | R4 禁用, R5 部分 | 对数惩罚, 重路由 |
| M5 (统计协议) | 0/4 | — | 全部 |
| **合计** | **17/27 (63%)** | **2** | **8** |

---

## 5. 已知问题与未决项

对照论文 §10「未决问题清单」：

| 问题 | 状态 |
|------|------|
| norm(·) 与 q(·) 形式化定义 | ✅ 代码中已实现（分位数归一化） |
| 3D NMS 规格 | ✅ 代码中已实现 |
| 侧别坐标系 (LPS/RAS) | ⚠️ 代码使用固定假设，未在报告中记录 |
| \|Ps\| = 0 处理 | ⚠️ 未显式记录策略 |
| λ_spatial × τ 联合敏感性 | ⬜ 未做消融 |
| sev_i 归一化 | ⬜ 未实现（Stage 5 重路由未实现） |
| 跨句一致性指标 | ⬜ 未实现 |
| 注意力代理约定 | ⬜ 未实现 |

---

## 6. 文件清单

```
outputs/stage0_5_llama_450/
├── ckpt_probe_report.json     # SwinUNETR checkpoint 检测报告
├── run.log                     # 完整运行日志
├── run_meta.json               # 实验超参数与数据元信息
├── summary.csv                 # 逐 case 结果汇总（900 行）
├── validation_report.json      # 结构验收报告（900 条，全部 PASS）
├── cache/                      # 编码器缓存
└── cases/
    ├── ctrate/                 # 450 个 case 目录
    │   └── {case_id}/
    │       └── trace.jsonl     # 逐句 trace（含路由分数、violations、judge 判定）
    └── radgenome/              # 450 个 case 目录
        └── {case_id}/
            └── trace.jsonl
```

---

## 7. 可复现性

```bash
# 1. 确保数据就位
ls manifests/ctrate_900_manifest.csv manifests/radgenome_900_manifest.csv

# 2. 运行实验
bash Scripts/run_stage0_5_llama_server.sh

# 3. 关键参数（硬编码在脚本中）
#    --max_cases 450
#    --shuffle_seed 42
#    --token_budget_b 128
#    --tau_iou 0.04
#    --llm_judge huggingface
#    --llm_judge_model models/Llama-3.1-8B-Instruct
```

---

## 8. 下一步计划

1. **扩展到 900/900**：当前数据池已就绪，可直接将 `--max_cases` 改为 900
2. **实现 Stage 3c (Token 门控生成)**：接入 LLM 做实际报告生成
3. **实现 Stage 5 重路由**：对数平滑惩罚 + 重新 top-k + 重新生成
4. **R4 规则启用**：大小/范围合理性检查需要收集更多解剖统计数据
5. **M5 统计协议**：实现 Bootstrap CI 和 Holm 校正，为论文提供统计显著性
6. **InfoNCE 训练循环**：训练 Wproj 优化路由质量
7. **λ_spatial × τ 联合消融**：验证空间先验与温度参数的交互
