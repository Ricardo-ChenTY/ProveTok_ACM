# ProveTok_ACM 项目报告

更新日期：2026-03-08

啊，这是 Mermaid 语法解析的一个常见问题！

错误的原因是节点名称中包含了圆括号 () 和 <br/> 标签，而在 Mermaid 中，[ ] 定义矩形节点时，如果内部包含这些特殊字符而不加双引号，解析器就会报错（它把 ( 误当成了某种特殊形状的语法标志）。

解决方法很简单：把所有带有复杂文本、<br/> 或括号的节点描述，用双引号 " " 括起来。 比如把 S0[Stage 0: 伪影风险评分<br/>(Artifact Risk Ai)] 改成 S0["Stage 0: 伪影风险评分<br/>(Artifact Risk Ai)"]。

我帮你把整个图表的文本都加上了安全的引号，请用下面这段代码替换你原来的：

Markdown
# ProveTok_ACM 项目报告

更新日期：2026-03-08

---

## 0. 最新系统架构概览 (Stage 0-4 锁定版本)，ProveTok_ACM 深度技术架构与演进路线图 (当前违规率: 15.2%)

```mermaid
graph TD
    %% 专为 GitHub 深色模式优化的配色方案
    classDef done fill:#0f3f21,stroke:#2ea043,stroke-width:2px,color:#ffffff;
    classDef todo fill:#592f00,stroke:#d25c00,stroke-width:2px,stroke-dasharray: 5 5,color:#ffffff;
    classDef future fill:#0b2c5c,stroke:#1f6feb,stroke-width:2px,stroke-dasharray: 5 5,color:#ffffff;
    classDef disabled fill:#21262d,stroke:#484f58,stroke-width:1px,stroke-dasharray: 5 5,color:#8b949e;
    classDef highlight fill:#490202,stroke:#da3633,stroke-width:2px,color:#ffffff;

    subgraph Inputs ["1. 原始输入 (Inputs)"]
        V["CT 体数据 (NIfTI)"]:::done
        R["人类专家放射报告"]:::done
    end

    subgraph Stage0_2 ["2. M1: 空间证据 Token Bank (基于确定性规则) [已完成]"]
        S0["Stage 0: 伪影风险评分<br/>(综合 SNR, streak, outlier)"]:::done
        S1["Stage 1: 冻结的 3D 编码器<br/>(Frozen SwinUNETR, 提取特征)"]:::done
        S2["Stage 2: 自适应八叉树分割<br/>(基于重要性分数分裂)"]:::done
        TB[("Evidence Token Bank<br/>最大 64 个, 含 BBox & 特征")]:::done

        V --> S0 & S1
        S0 -.->|"Artifact State"| S2
        S1 -.->|"Feature Map"| S2
        S2 --> TB
    end

    subgraph Stage3 ["3. M2: 跨模态轻量路由层 (Router)"]
        P["Report Planner (当前)<br/>[已完成] 正则提取: 解剖词/侧别/否定"]:::done
        AR["Anatomy Resolver (当前)<br/>[已完成] 词表映射 & Left/Right Lung 兜底"]:::done
        
        LLM_P["Stage 3a: LLM 高级规划器 [还没做]<br/>(解析上下文, 输出 JSON 结构化病灶/解剖/否定)"]:::future
        FixR1["动态解剖词表扩展 [待做 P1]<br/>(如新增 pleural space 等精细 BBox)"]:::todo

        RT["Router: Anatomy-Primary [已完成]<br/>Score = IoU + ε * Semantic (语义仅作 tiebreaker)"]:::done

        R --> P
        P --> AR
        AR -.->|"短期打补丁"| FixR1
        P -.->|"长期彻底替代"| LLM_P

        P -->|"Text Query"| RT
        AR -->|"Anatomy BBox"| RT
        TB -->|"Token BBox & 特征"| RT
        
        RT -->|"Top-k 采样 (默认 k=8)"| TKT["句子对应的 Top-k 证据 Tokens"]:::done
    end

    subgraph Stage4 ["4. M4: 纯空间与几何验证器 (当前总体违规率 15.2%)"]
        VF{"Verifier: 5条规则并行审计"}:::highlight
        
        R1["R1: LATERALITY (侧别)<br/>[已完成] Ratio ≥ 0.6 触发<br/>(已豁免否定句与中线结构)"]:::done
        R2["R2: ANATOMY (解剖)<br/>[已完成] Ratio ≥ 0.8 触发<br/>(已豁免 bilateral/lung 大框)"]:::done
        R3["R3: DEPTH (深度)<br/>[已完成] 粗/细粒度校验"]:::done
        R4["R4: SIZE (大小)<br/>[已禁用] 阈值未校准"]:::disabled
        R5["R5: NEGATION (否定)<br/>[已禁用] Fallback词表关闭"]:::disabled

        FixR2["修复 R2 结构性误报 [待做 P0]<br/>(将 mediastinum 加入双侧豁免<br/>预计将全局违规率压至 ~9%)"]:::todo
        LLM_J["LLM 语义裁判 (Judge) [还没做]<br/>(输入 Top-k 视觉特征, 判定是否真实包含描述病灶)"]:::future

        TKT --> VF
        P -->|"传入 Sentence Plan"| VF
        TB -->|"传入全局 Midline 参考"| VF

        VF --> R1 & R2 & R3 & R4 & R5
        R2 -.->|"1行代码消除误报"| FixR2
        VF -.->|"弥补纯空间无语义的短板"| LLM_J
    end

    subgraph Stage5 ["5. 终极愿景: Token 门控生成与自纠错 [还没做]"]
        TG["Stage 3c: Token-Gated Generation<br/>(多模态 LLM 强制基于物理 Tokens 生成描述)"]:::future
        SR["Stage 5: 严重度驱动重路由 (Reroute)<br/>(r' = r - γ * ln(1 + sev) 更新分数并重新采样)"]:::future
    end

    R1 & R2 & R3 -.->|"若判定违规且严重度 sev > 0"| SR
    SR -.->|"反馈惩罚, 触发一次性 Retry"| RT
    TKT -.->|"作为 Grounded 视觉前缀"| TG
```
    
## 1. 项目背景与目标

### 1.1 任务定义

ProveTok（Provenance Token）是一个**无需 LLM 生成、完全基于规则与空间推理**的 CT 报告溯源验证系统。

给定：
- 一段 CT 放射报告文本（如 "There is a nodule in the left lower lobe."）
- 对应的 3D CT 体积（NIfTI 格式）
- 一个冻结的 SwinUNETR 图像编码器 checkpoint

目标：
- 将报告中每个句子与 CT 体积中最相关的 3D 空间区域（EvidenceToken）关联起来
- 用规则验证这些关联是否**空间上合理**（侧别一致、解剖区域一致等）
- 输出每个句子的违规情况，作为报告可信度的量化指标

### 1.2 技术意义

- 不依赖 LLM 打分，推理成本低，完全可复现
- 可作为医学报告 grounding 质量的自动化评估工具
- 数据集：CT-RATE（英文报告）+ RadGenome（中英文报告），各 450 例，共 900 例

---

## 2. 系统架构（Stage 0-4）

整个流程分五个阶段，串行执行：

```
CT Volume + Report Text
       │
  Stage 0: Artifact Scoring（伪影检测，DeterministicArtifactScorer）
       │
  Stage 1: Image Encoding（FrozenSwinUNETREncoder，SwinUNETR feature_size=48，冻结权重）
       │
  Stage 2: Octree Splitting（AdaptiveOctreeSplitter，自适应八叉树分割，生成 EvidenceToken Bank）
       │
  Stage 3: Report-to-Token Routing（Router，将报告句子路由到最相关 token）
       │
  Stage 4: Rule-based Verification（Verifier，5条规则审计每个句子的引用合理性）
       │
  输出：trace.jsonl / summary.csv / violation report
```

### 2.1 核心数据结构

**EvidenceToken**：CT 体积中的一个 3D 区域
- `bbox: BBox3D`（归一化坐标，x∈[0,1]，x=0→患者右侧，x=1→患者左侧）
- `level: int`（八叉树层级，0=粗粒度，5=细粒度）
- `feature: List[float]`（SwinUNETR 编码特征）
- `split_score: float`（分裂优先级）

**SentencePlan**：报告句子的结构化解析
- `anatomy_keyword`：解剖部位关键词（如 "left lower lobe"）
- `expected_level_range`：预期 token 层级范围
- `expected_volume_range`：预期体积范围
- `is_negated`：是否为否定句

### 2.2 Stage 3：Router 打分公式

标准模式（原始）：
```
score = dot(text_query, token_feature) + λ_spatial × IoU(token_bbox, anatomy_bbox)
```

**Anatomy-primary 模式**（`--anatomy_spatial_routing`，当前使用）：
```
score = IoU(token_bbox, anatomy_bbox) + ε × dot(text_query, token_feature)
```

切换原因：`w_proj`（文本→图像投影矩阵）未经训练，语义点积不可靠；IoU 是唯一有校准意义的信号。

### 2.3 Stage 4：验证规则（5条）

| 规则 | 含义 | 当前状态 |
|---|---|---|
| **R1_LATERALITY** | cited token 与句子侧别声明（左/右）一致性 | 启用，ratio 模式 |
| **R2_ANATOMY** | cited token bbox 与解剖区域 bbox 的 IoU ≥ τ | 启用，ratio 模式 |
| **R3_DEPTH** | cited token 层级与句子语义深度一致 | 启用 |
| **R4_SIZE** | cited token 联合 bbox 体积与大小描述一致 | **已禁用**（`--r4_disabled`） |
| **R5_NEGATION** | 否定句不应引用高 positive 置信度 token | **fallback 已禁用**（`--r5_fallback_disabled`） |

---

## 3. 优化历程

### 3.1 起点：旧基线

450/450 全量（旧基线，约 2026-03 初）：
- 句级违规率：**72.1%**
- R1_LATERALITY：高（具体值未精确记录）
- R2_ANATOMY：2651

---

### 3.2 第一轮：Anatomy Spatial Routing

**改动**：`stage3_router.py`
- 新增 `anatomy_spatial_routing` 模式：`score = IoU + ε × semantic`
- 新增 `anatomy_tiebreak_eps = 0.05`（语义分仅作 tiebreaker）

**50-case 效果**：违规率 45.75% → 42.00%，R1: 119→100，R2: 269→239

---

### 3.3 第二轮：R2 Bilateral Skip

**改动**：`run_mini_experiment.py`
- 新增 `--r2_skip_bilateral` flag，将 `"bilateral"` 加入 `r2_skip_keywords`
- 原因：bilateral 句子天然覆盖双侧，任何 token 与 bilateral bbox 的 IoU 都很低，是结构性误报

**50-case 效果**：违规率 42.00% → **17.38%**，**R2: 239→42（大幅下降）**，R1: 100（未变）

---

### 3.4 第三/四轮：R1 否定句豁免 + 中线解剖词豁免

**根因分析**：
1. **否定句误触发**：`"no left pleural effusion"` → `parse_laterality` 检测到 "left" → R1 触发，但否定句 cite 的是背景 token，不要求在左侧
2. **中线解剖词误触发**：`mediastinum / trachea / aorta` 等结构天生跨中线，cited token 必然覆盖双侧

**改动**：
- `config.py`：新增 `r1_negation_exempt: bool`、`r1_skip_midline_keywords: set`
- `stage4_verifier.py`：R1 块前增加 `r1_skipped` guard
- `run_mini_experiment.py`：新增 `--r1_negation_exempt` 和 `--r1_skip_midline` flag
  - `r1_skip_midline` 默认词表：mediastinum / trachea / carina / esophagus / aorta / spine / vertebra / sternum

**50-case 效果**：R1: 100 → 90（含 ratio=1.0 严格模式）

---

### 3.5 第五轮：R1 Ratio 模式

**根因分析**：原 R1 是 all-or-nothing，只要 1 个 cited token 在错误侧就触发。但有时只有少数 token 跑偏，整体路由方向是对的。

**改动**：
- `config.py`：新增 `r1_min_same_side_ratio: float = 1.0`
- `stage4_verifier.py`：R1 改为 ratio 模式
  - 将 cited tokens 按 `token_side()` 分类为 same / opp / cross
  - cross tokens（中线附近）排除在分母之外
  - `same_ratio = len(same) / (len(same) + len(opp))`
  - 仅当 `same_ratio < r1_min_same_side_ratio` 时触发

**50-case 效果（ratio=0.6）**：R1 仍为 90（ratio 模式对 `same_ratio≈0` 的全错侧 case 无效）

---

### 3.6 第六轮：Left/Right Lung BBox Fallback

**根因分析**：90 个 R1 违规句子中，`same_side_ratio ≈ 0`，即 8 个 cited token 全部在错误侧。根因是 `"left lung opacity"` 等句子在 `_extract_keyword()` 中无法匹配到任何 anatomy keyword → `anatomy_keyword=None` → `anatomy_bbox=None` → router 退回纯语义打分（w_proj 未训练）→ token 随机路由到错误侧。

**改动**：`simple_modules.py`
- `DEFAULT_ANATOMY_BOXES` 新增 `"left lung"` 和 `"right lung"` 两个大 bbox
- `_extract_keyword()` 新增 fallback：有 "left" 但无细化解剖词 → `"left lung"`；有 "right" → `"right lung"`

**副作用修复**（第六轮 v2）：加入 left/right lung bbox 后，这些句子开始被 R2 检查，R2 从 42 暴涨到 133。修复：`"left lung"` 和 `"right lung"` 无条件加入 `r2_skip_keywords`（路由兜底词，不应做 R2 精度校验）。

**50-case 效果**：**R1: 90→82**，R2: 42（稳定）

---

### 3.7 第七轮：BBox 边界对齐（Plan A）

**假设**：x∈[0.48, 0.52] 死区导致 midline 附近 token IoU=0，将 bbox 边界对齐到 x=0.50 可消除死区。

**改动**：`simple_modules.py`
- 所有 bbox x 边界对齐到 0.50（right lung x_max: 0.48→0.50，left lung x_min: 0.52→0.50，lobe 边界 0.45/0.55→0.50）
- `run_mini_experiment.py`：新增 `--lateral_tolerance` flag

**50-case 效果**：R1=82，**无收益**。根因：R1 违规 token 分布在对面整个半肺，不在 midline 死区附近。

---

## 4. 实验结果汇总

### 4.1 50-case 回归历程（128³ 分辨率）

| 轮次 | 配置要点 | R1 | R2 | 违规句率 |
|---|---|---:|---:|---:|
| 基线 | 无优化 | 119 | 269 | 45.75% |
| +anatomy_spatial_routing | IoU 主导打分 | 100 | 239 | 42.00% |
| +r2_skip_bilateral | bilateral 不检查 R2 | 100 | **42** | **17.38%** |
| +negation_exempt +skip_midline | 否定句/中线词豁免 R1 | 90 | 42 | 16.50% |
| +r1_min_same_side_ratio=0.6 | R1 ratio 模式 | 90 | 42 | — |
| +left/right lung bbox fallback | 兜底 bbox | **82** | 42 | — |
| +bbox 边界对齐 0.50 | Plan A（无收益） | 82 | 42 | — |

### 4.2 450/450 全量对比

| 指标 | 旧基线（~3/6） | 中间版本（3/7，含前3轮） | **本次（Round 8，3/8）** |
|---|---:|---:|---:|
| 样本量 | 900 | 900 | 900 |
| 总句数 | — | 7187 | — |
| 总 violations | — | 2190 | **1089** |
| R1_LATERALITY | — | 1770 | **669** |
| R2_ANATOMY | — | 420 | **420** |
| ctrate 句级违规率 | — | 32.42% | **16.74%** |
| radgenome 句级违规率 | — | 27.12% | **13.56%** |
| 结构验收（900 cases） | — | 通过 | **通过** |

**总体违规率：约 15%，相比最早旧基线（72.1%）降幅约 79%。**

---

## 5. 当前锁定配置

```bash
python run_mini_experiment.py \
  --max_cases 450 --expected_cases_per_dataset 450 \
  --cp_strict \
  --text_encoder semantic \
  --text_encoder_model sentence-transformers/all-MiniLM-L6-v2 \
  --device cuda \
  --token_budget_b 64 --k_per_sentence 8 \
  --lambda_spatial 0.3 --tau_iou 0.05 --beta 0.1 \
  --r2_mode ratio --r2_min_support_ratio 0.8 \
  --r4_disabled --r5_fallback_disabled \
  --anatomy_spatial_routing \
  --r2_skip_bilateral \
  --r1_negation_exempt \
  --r1_skip_midline \
  --r1_min_same_side_ratio 0.6
```

---

## 6. 当前问题分析

### 6.1 R2_ANATOMY 残留（420 条，全来自 mediastinum）

本次 450/450 中 R2=420，且 anatomy 细项显示 mediastinum 违规率 **100%（420/420）**。

根因：mediastinum bbox 定义为 `BBox3D(0.40, 0.60, 0.20, 0.80, 0.00, 1.00)`，体积约为整体的 4%，与 SwinUNETR token bbox（通常为小区域）的 IoU 天然极低（<0.05），必然触发 R2。

类比：bilateral 和 left/right lung 同样因为 bbox 尺度不匹配而被加入 `r2_skip_keywords`，mediastinum 面临相同问题。

### 6.2 R1_LATERALITY 剩余（669 条）

当前剩余 R1 的特征：`same_side_ratio ≈ 0`（全部 cited token 在错误侧），说明 router 路由失败，不是 ratio 阈值问题。根因可能是：
- 句子中有侧别词但对应解剖结构不在 `anatomy_boxes` 词表中（如 `"right pleural effusion"` 中 "pleural space" 没有精细 bbox）
- 语义打分（w_proj 未训练）在无 bbox fallback 时完全不可靠

---

## 7. 后续工作计划

### P0：修复 R2 mediastinum（1行代码，预计 R2 归零）

将 `"mediastinum"` 加入 `r2_skip_keywords`：

```python
# run_mini_experiment.py 中
cfg.verifier.r2_skip_keywords.update({"left lung", "right lung", "mediastinum"})
```

或新增 `--r2_skip_mediastinum` flag。

**预期**：R2 从 420 → 0，总体违规率从 ~15% 降到 ~9%。

---

### P1：R1 进一步优化（可选，视截止日期决定）

当前剩余 R1=669，方向：

1. **扩充 anatomy_boxes 词表**：补充 `"pleural space"` / `"pleural effusion"` / `"hilar"` 等常见放射词汇对应的 bbox，减少 router 退回纯语义打分的概率
2. **动态 anatomy keyword 扩展**：在 `_extract_keyword()` 中增加更多 pattern 匹配（如 "pleural" → left/right pleural space bbox）
3. **降低 r1_min_same_side_ratio**：从 0.6 进一步降到 0.4-0.5，容忍更多混合侧向 token（需先用 50-case 验证副作用）

---

### P2：消融实验与论文写作支撑

当前实验结果已经可以支撑以下 ablation table：

| 配置 | R1 | R2 | 违规句率 |
|---|---:|---:|---:|
| Baseline（纯语义打分） | 119 | 269 | 45.75% |
| +anatomy_spatial_routing | 100 | 239 | 42.00% |
| +r2_skip_bilateral | 100 | 42 | 17.38% |
| +negation/midline exempt | 90 | 42 | 16.50% |
| +left/right lung fallback | 82 | 42 | — |
| Full（450/450） | 669 | 420 | **15.2%** |

---

## 8. 代码文件索引

| 文件 | 说明 |
|---|---|
| `ProveTok_Main_experiment/types.py` | 核心数据类型（BBox3D, EvidenceToken, SentenceOutput, RuleViolation 等） |
| `ProveTok_Main_experiment/config.py` | 所有配置 dataclass（SplitConfig, RouterConfig, VerifierConfig 等） |
| `ProveTok_Main_experiment/stage0_scorer.py` | Stage 0：伪影评分 |
| `ProveTok_Main_experiment/stage1_swinunetr_encoder.py` | Stage 1：SwinUNETR 冻结编码器 |
| `ProveTok_Main_experiment/stage2_octree_splitter.py` | Stage 2：自适应八叉树 token 分割 |
| `ProveTok_Main_experiment/stage3_router.py` | Stage 3：Router（anatomy-primary 打分） |
| `ProveTok_Main_experiment/stage4_verifier.py` | Stage 4：5条规则验证 |
| `ProveTok_Main_experiment/simple_modules.py` | 解剖 bbox 字典、ReportSentencePlanner、RuleBasedAnatomyResolver |
| `ProveTok_Main_experiment/stage0_4_runner.py` | 单 case 端到端运行入口 |
| `run_mini_experiment.py` | 主实验 CLI 入口，所有 flag 定义 |
| `validate_stage0_4_outputs.py` | 结构验收脚本 |
| `Smoke_analysis/OUTPUT_ANALYSIS_COLAB.ipynb` | Colab 分析 notebook（违规率统计、可视化、导出） |

---

## 9. 关键结论

1. **anatomy_spatial_routing 是系统核心改进**：在 w_proj 未训练的情况下，用 IoU 主导 router 打分是唯一有效的空间校准手段。

2. **结构性误报需要结构性豁免**：bilateral / left lung / right lung / mediastinum 等"大 bbox 结构"与小 token 的 IoU 天然极低，不应做 R2 精度校验，应加入 skip_keywords。

3. **50-case → 450/450 存在分布漂移**：50-case 违规率 17%，全量 15%，较接近；但历史上曾出现 17% → 30% 的跳跃，说明全量验证不可省略。

4. **当前系统完全不依赖 LLM**：所有决策基于 bbox 几何、规则词表匹配、SwinUNETR 特征，推理速度快，结果完全可复现。
