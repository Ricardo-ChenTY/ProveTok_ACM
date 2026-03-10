# Recent Changes (CN)

更新日期：2026-03-10

## 第九轮：接入 LLM —— Stage 5 裁判 + Stage 3c 生成 + W_proj 训练

### 背景

Stage 0-4 是纯规则 + 空间 IoU 的验证系统，没有语义理解能力。本轮新增两条 LLM 路线，分步推进：

- **P1（Stage 5 LLM Judge）**：用 LLM 对 Stage 4 找到的违规做二次裁决，确认真实违规 → 按 CP .tex 公式惩罚路由分数，过滤误报
- **Stage 3c（Token-Gated Generation）**：需先训练 W_proj（InfoNCE），再用 LLM 以 token 空间坐标为上下文生成报告句子（CP .tex 核心贡献）

### 新增文件

| 文件 | 作用 |
|------|------|
| `ProveTok_Main_experiment/stage5_llm_judge.py` | Stage 5 LLM 裁判，支持 Ollama / OpenAI / Anthropic / HuggingFace 四种后端 |
| `ProveTok_Main_experiment/stage3c_generator.py` | Stage 3c LLM 报告生成，同样支持四种后端 |
| `train_wprojection.py` | InfoNCE 训练 W_proj，读取 Stage 0-4 的 trace.jsonl + tokens.pt 作为训练数据 |
| `models/` | 本地模型目录（git 不追踪），下载模型放到 `models/Llama-3.1-8B-Instruct/` |
| `.gitignore` | 排除 `models/*/` 和大文件 |

### 修改文件

**`ProveTok_Main_experiment/config.py`**

- 新增 `LLMJudgeConfig` dataclass（backend / model / alpha / ollama_host / hf_device_map / hf_torch_dtype / hf_token / fail_open）
- `ProveTokConfig` 新增 `llm_judge: LLMJudgeConfig`

**`ProveTok_Main_experiment/stage0_4_runner.py`**

- `Stage04Components` 新增 `llm_judge: Optional[LLMJudge] = None`
- `run_case_stage0_4()` 在 Stage 4 审计后新增 Stage 5 块：
  - 调用 `llm_judge.judge_all()` 对所有违规句做 LLM 裁决
  - 对确认违规句应用 CP .tex 惩罚：`S'_i = S_i × (1 - α × sev_i)`
  - 标记 `s_out.rerouted = True, stop_reason = "llm_judge_penalty"`
  - `stage5_judgements` 写入 `trace.jsonl` 每个 sentence 行
- 返回值新增 `n_judge_confirmed`

**`run_mini_experiment.py`**

- 新增 `--llm_judge` / `--llm_judge_model` / `--llm_judge_alpha` / `--llm_judge_ollama_host` / `--llm_judge_hf_device_map` / `--llm_judge_hf_torch_dtype` / `--llm_judge_hf_token` 共 7 个参数
- HuggingFace 后端默认模型路径：`models/Llama-3.1-8B-Instruct`（本地目录，自动解析为绝对路径）
- 本地路径不传 token，远程 Hub 才传
- `run_meta.json` 记录 `llm_judge_backend / llm_judge_model / llm_judge_alpha`
- `summary.csv` 新增 `n_judge_confirmed` 列

### Stage 5 核心逻辑

```
Stage 4 Verifier
  → violations (R1/R2/R3/R5)
  → LLM Judge（每条违规独立问 LLM）
      LLM 回复：{"confirmed": true/false, "severity": 0.0-1.0, "reasoning": "..."}
  → 确认的违规 → 惩罚路由分数
      S'_i = S_i × (1 - alpha × max_confirmed_severity)
  → trace.jsonl 记录裁决结果
```

### W_proj 训练（Stage 3c 前置）

```bash
# 先跑 Stage 0-4 生成 token bank
python run_mini_experiment.py --ctrate_csv ... --radgenome_csv ... --max_cases 50

# 再训练 W_proj
python train_wprojection.py \
  --cases_dir outputs_stage0_4/cases \
  --text_encoder semantic \
  --epochs 50 \
  --out_dir outputs_wprojection
```

### 决策

两条路线都要做，顺序：P1 先上（快，LLM 直接接 Stage 4 后），Stage 3c 后做（需要训练 W_proj）。

---

更新日期：2026-03-08

## 第八轮：bbox 边界对齐 + 直接上 450/450

### 改动

**`ProveTok_Main_experiment/simple_modules.py`**

- `DEFAULT_ANATOMY_BOXES` 边界对齐到 x=0.50（Plan A）：
  - `right lung` x_max: `0.48 → 0.50`
  - `left lung` x_min: `0.52 → 0.50`
  - 所有 lobe x 边界从 `0.45/0.55` → `0.50`
- 消除 x∈[0.48,0.52] 死区，midline 附近 token 不再 IoU=0

**`run_mini_experiment.py`**

- 新增 `--lateral_tolerance` flag（暴露已有 `cfg.verifier.lateral_tolerance`，默认 0.0）
- `run_meta.json` 记录 `lateral_tolerance`

### 结果

50-case 128^3 回归（Plan A）：R1=82，R2=42（与 r7v2 128 基线持平，Plan A 无收益）。

根因：R1 违规 token 分布在对面整个半肺，不在 midline 附近。死区修复未触及主要失败模式。

### 决策

不再继续压 R1，直接上 450/450 主实验。当前 50-case 128^3 改善幅度（R2: 269→42，R1: 119→82）相比上次 450/450 基线（R2=420，R1=1770）预计有明显收益。

---

## 第七轮改进：anatomy resolver 补 left lung / right lung bbox

### 背景

50-case 回归（含 ratio 0.6）显示 R1 仍为 90，ratio 模式完全无效。根因：90 个违规句子的 `same_side_ratio ≈ 0`，即 8 个 cited token 全部在错误侧。

原因是 `"left lung opacity"` / `"right pleural effusion"` 等句子在 `_extract_keyword()` 中无法匹配到任何 anatomy keyword（只有 lobe 级别的词），导致 `anatomy_keyword=None` → `anatomy_bbox=None` → router 退回纯语义打分（w_proj 未训练，不可靠）→ 全部 token 路由到错误侧。

### 改动

**`ProveTok_Main_experiment/simple_modules.py`**

- `DEFAULT_ANATOMY_BOXES` 新增：
  - `"left lung": BBox3D(0.52, 1.00, 0.00, 1.00, 0.00, 1.00)`
  - `"right lung": BBox3D(0.00, 0.48, 0.00, 1.00, 0.00, 1.00)`
- `_extract_keyword()` 新增 fallback：有 `"left"` 但无细化解剖词 → `"left lung"`；有 `"right"` → `"right lung"`

### 预期效果

router 拿到 left/right lung bbox 后，`anatomy_spatial_routing` 模式下 IoU 主导打分，token 会优先从正确半肺里选取 → R1 same_side_ratio 大幅提升 → R1 明显减少触发。

### 副作用修复（第七轮 v2）

加入 left/right lung bbox 后，之前 `anatomy_keyword=None`（→R2 不检查）的句子现在有了 bbox → R2 开始检查 → R2 从 42 暴涨到 133。

原因：half-volume 大 bbox（约 0.48 体积）与小 token bbox 的 IoU 天然很低（~0.02），低于 `tau_iou=0.05`。

修复：`run_mini_experiment.py` 中无条件把 `"left lung"` 和 `"right lung"` 加入 `r2_skip_keywords`，这两个是路由兜底词，不应做 R2 精度校验。

### 验证策略

50-case 回归，预期 R1 从 90 降到 < 30，R2 维持 ~42（不上升）。

---

## 第六轮改进：R1 ratio 模式（same_side_ratio 阈值）

### 背景

50-case 回归（全部 4 flag）显示：R1: 100 → 90（仅降 10），negation_exempt + skip_midline 的收益有限。剩余 90 个 R1 是 router 路由到错误侧的真实 case，根因是原始 R1 逻辑**全匹配（all-or-nothing）**：只要有一个 cited token 在错误侧就触发，但有时只有少数 token 跑偏，整体 routing 是正确的。

### 改动

**`ProveTok_Main_experiment/config.py`**

- 新增 `r1_min_same_side_ratio: float = 1.0`（默认 1.0 = 原有严格行为）

**`ProveTok_Main_experiment/stage4_verifier.py`**

- R1 block 改为 ratio 模式：
  - 将 cited tokens 按 `token_side()` 分类为 `same / opp / cross`
  - cross tokens 排除在比例计算之外
  - `same_side_ratio = len(same) / (len(same) + len(opp))`
  - 仅当 `same_side_ratio < r1_min_same_side_ratio` 时触发 R1
  - violation message 带 ratio 数值

**`run_mini_experiment.py`**

- 新增 `--r1_min_same_side_ratio` flag（float，默认 None → 不改配置，等于 1.0 严格）
- config wiring + run_meta 记录

### 验证策略

50-case 先跑两组对比：

```bash
# 当前基线（ratio=1.0 严格）
python run_mini_experiment.py ... --anatomy_spatial_routing --r2_skip_bilateral --r1_negation_exempt --r1_skip_midline

# ratio 模式（推荐阈值 0.6）
python run_mini_experiment.py ... --anatomy_spatial_routing --r2_skip_bilateral --r1_negation_exempt --r1_skip_midline --r1_min_same_side_ratio 0.6
```

预期：R1 从 90 明显下降（目标 < 40），R2 保持稳定（~42）。

---

## 第五轮改进：R1 优化（否定句豁免 + 中线解剖词豁免）

### 背景

450/450 全量运行显示 R1_LATERALITY 占总违规约 81%（1770/2190），且从 50-case（12.5%/sentence）到 450（24.6%/sentence）几乎翻倍。根因分析：

1. **否定句误触发**：`"no left pleural effusion"` 中 `parse_laterality` 检测到 "left" → R1 触发，但否定句 cite 的是背景/正常组织 token，不要求在左侧，是 false positive。
2. **中线解剖词误触发**：`mediastinum / trachea / aorta` 等结构天生跨中线，cited token 覆盖双侧，R1 必然触发，是结构性误报。

### 改动

**`ProveTok_Main_experiment/config.py`**

- 新增 `r1_negation_exempt: bool = False`
- 新增 `r1_skip_midline_keywords: set = field(default_factory=set)`

**`ProveTok_Main_experiment/stage4_verifier.py`**

- 在 R1 块前计算 `negated`（复用给 R5，去掉 R5 处的重复计算）
- 新增 `r1_skipped` guard：`negation_exempt and negated` 或 `anatomy_keyword in r1_skip_midline_keywords`

**`run_mini_experiment.py`**

- 新增 `--r1_negation_exempt` flag
- 新增 `--r1_skip_midline` flag（覆盖词表：mediastinum/trachea/carina/esophagus/aorta/spine/vertebra/sternum）
- `run_meta.json` 记录两个新字段

### 验证策略

先在 50-case smoke 上跑回归：

```bash
python run_mini_experiment.py ... \
  --r2_skip_bilateral \
  --r1_negation_exempt \
  --r1_skip_midline
```

预期：R1 明显下降（50-case 基线 100 → 目标 < 60），R2 保持稳定（~42）。回归通过后再上 450/450。

---

## 本次更新做了什么

这次没有继续改主实验逻辑，重点是把文档和分析入口统一到 `450/450` 主实验口径。

### 1. 锁定 450/450 主实验配置

当前推荐配置固定为：

- `--cp_strict`
- `--r2_mode ratio`
- `--r2_min_support_ratio 0.8`
- `--tau_iou 0.05`
- `--anatomy_spatial_routing`
- `--r2_skip_bilateral`
- `--r4_disabled`
- `--r5_fallback_disabled`

锁定依据是最新 `50/50` smoke：

- `Validated: 100, Passed: 100, Failed: 0`
- 总违规句率：`42.00% -> 17.375%`
- `R2_ANATOMY: 239 -> 42`
- `R1_LATERALITY: 100 -> 100`

### 2. 更新运行说明

文件：

- [RUN_GUIDE_CN.md](c:\Users\34228\Desktop\ACM\RUN_GUIDE_CN.md)

更新内容：

- 改成以 `450/450` 主实验为中心
- 保留 Linux/Colab Shell 和 Windows PowerShell 两套命令
- 明确写出结构验收命令
- 明确写出 Colab notebook 怎么做验收和分析

### 3. 更新结果验收说明

文件：

- [OUTPUT_ANALYSIS_GUIDE_CN.md](c:\Users\34228\Desktop\ACM\OUTPUT_ANALYSIS_GUIDE_CN.md)

更新内容：

- 不再停留在旧 baseline 的描述
- 新增 `50/50` smoke 改善结果
- 新增 `450/450` 硬性验收标准
- 新增 `450/450` 软性目标区间

### 4. 更新 Colab 分析 notebook

文件：

- [OUTPUT_ANALYSIS_COLAB.ipynb](c:\Users\34228\Desktop\ACM\Smoke_analysis\OUTPUT_ANALYSIS_COLAB.ipynb)

更新目标：

- 默认切到 `450/450` 单次运行分析
- 结构验收强制 `expected_cases_map = ctrate=450,radgenome=450`
- 直接显示主实验状态检查
- 导出文件名与文档统一

## 现在你朋友该怎么做

5090 跑实验的人：

1. 按 [RUN_GUIDE_CN.md](c:\Users\34228\Desktop\ACM\RUN_GUIDE_CN.md) 的 `450/450` 主实验命令运行
2. 再运行同一份 guide 里的结构验收命令
3. 把结果目录交给分析同学

Colab 做分析的人：

1. 打开 [OUTPUT_ANALYSIS_COLAB.ipynb](c:\Users\34228\Desktop\ACM\Smoke_analysis\OUTPUT_ANALYSIS_COLAB.ipynb)
2. 把 `OUT_DIR` 改成这次 `450/450` 结果目录
3. 运行全部 cells
4. 根据 [OUTPUT_ANALYSIS_GUIDE_CN.md](c:\Users\34228\Desktop\ACM\OUTPUT_ANALYSIS_GUIDE_CN.md) 判断是否达标

## 当前判断

现在最重要的不是再改 bilateral 逻辑，而是先把 `450/450` 主实验跑出来并完成标准化验收。
