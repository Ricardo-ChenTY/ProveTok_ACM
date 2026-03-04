# Colab 450/450 重跑指南（CP strict）

这份文档给协作者在 **Google Colab** 直接重跑你们的 450/450 实验，覆盖以下内容：

1. 本次代码改了什么
2. Colab 需要准备什么
3. 先做 ckpt 可用性探测（强制）
4. 怎么重跑 450/450（CP strict）
5. 怎么验收输出是否达标

## 1) 本次改动摘要（你朋友需要知道）

核心目标：按 CP 口径把 Stage0-4 跑得更严格、更可追溯。

- `run_mini_experiment.py`
  - 新增 `--cp_strict`
  - 新增 `--text_encoder`（`hash|semantic`）和 semantic 模型参数
  - strict 模式下强制要求 `--encoder_ckpt`
  - 新增 `case_id` 唯一性检查（防覆盖写盘）
  - 新增样本数硬校验 `--expected_cases_per_dataset`
  - 新增 `run_meta.json`（记录本次运行配置和样本计数）
- `validate_stage0_4_outputs.py`
  - 新增 `--expected_cases_map`（如 `ctrate=450,radgenome=450`）
- `stage4_verifier.py`
  - R2 从“单个 max IoU 放行”改为“支持比例”判定（默认更严格）
  - R1 中线可用体素全局中心（由 runner 传入体素尺寸）
  - R5 增加 fallback（当 `negation_conflict` 不可用时，词典触发）
- `preprocess.py`
  - 非 `.npy` 医学影像读取后统一到 `LPS` 方向
- `stage2_octree_splitter.py` / `stage0_artifacts.py`
  - Stage2 复用 Stage0 缓存统计（减少冗余计算）
  - token metadata 补 `negation_conflict` 字段（默认 0）
- `simple_modules.py`
  - 扩展 R3/R4 触发词典与 negation 词典
- `environment.yaml`
  - 增加 `sentence-transformers`

## 2) Colab 准备

### 必需输入

- CT-RATE manifest CSV（至少含列：`case_id, volume_path, report_text`）
- RadGenome manifest CSV（至少含列：`case_id, volume_path, report_text`）
- SwinUNETR checkpoint（`--encoder_ckpt`）
  - 注意：`有链接 != 可用`，必须先跑 probe（见第 3 节）

### 建议目录（Google Drive）

- `/content/drive/MyDrive/Data/manifests/ctrate_manifest.csv`
- `/content/drive/MyDrive/Data/manifests/radgenome_manifest.csv`
- `/content/drive/MyDrive/Data/checkpoints/swinunetr.ckpt`
- 输出：`/content/drive/MyDrive/Data/outputs_stage0_4_full450_cp_strict`

## 3) 先做 ckpt 可用性探测（强制）

你们当前代码固定模型签名是：

- `SwinUNETR(in_channels=1, out_channels=2, feature_size=48)`

很多公开权重是 `out_channels=14` 或 `in_channels=4`，下载后会在加载时报 shape mismatch。
所以在 450/450 前，先跑一次 probe；probe 不通过就不要跑主实验。

```bash
# 3.1 先探测 checkpoint（本地路径版本）
!python Scripts/ckpt_probe.py \
  --ckpt_path /content/drive/MyDrive/Data/checkpoints/swinunetr.ckpt \
  --in_channels 1 \
  --out_channels 2 \
  --feature_size 48 \
  --save_report /content/drive/MyDrive/Data/ckpt_probe_report.json
```

如果你只有 URL，也可以：

```bash
!python Scripts/ckpt_probe.py \
  --ckpt_url "https://<your_ckpt_link>" \
  --in_channels 1 \
  --out_channels 2 \
  --feature_size 48 \
  --save_report /content/drive/MyDrive/Data/ckpt_probe_report.json
```

判定标准：

- 返回码 `0`，且 JSON 里 `compatible=true` 才能继续 `--cp_strict`
- 如果 `compatible=false`，先换权重，不要直接跑 450

## 4) Colab 一键步骤

在 Colab 中新开代码单元，按顺序运行：

```bash
# 1) 挂载 Drive
from google.colab import drive
drive.mount('/content/drive')
```

```bash
# 2) 拉代码
%cd /content
git clone https://github.com/Ricardo-ChenTY/ProveTok_ACM.git
%cd /content/ProveTok_ACM
```

```bash
# 3) 安装依赖（Colab 用 pip）
!pip install -U pip
!pip install numpy pandas scipy simpleitk monai "sentence-transformers>=3.0"
```

```bash
# 4) 运行 450/450（CP strict）
!python run_mini_experiment.py \
  --ctrate_csv /content/drive/MyDrive/Data/manifests/ctrate_manifest.csv \
  --radgenome_csv /content/drive/MyDrive/Data/manifests/radgenome_manifest.csv \
  --out_dir /content/drive/MyDrive/Data/outputs_stage0_4_full450_cp_strict \
  --max_cases 450 \
  --expected_cases_per_dataset 450 \
  --cp_strict \
  --encoder_ckpt /content/drive/MyDrive/Data/checkpoints/swinunetr.ckpt \
  --text_encoder semantic \
  --text_encoder_model sentence-transformers/all-MiniLM-L6-v2 \
  --text_encoder_device cuda \
  --device cuda \
  --token_budget_b 64 \
  --k_per_sentence 8 \
  --lambda_spatial 0.3 \
  --tau_iou 0.1 \
  --beta 0.1
```

```bash
# 5) 验收 450/450
!python validate_stage0_4_outputs.py \
  --out_dir /content/drive/MyDrive/Data/outputs_stage0_4_full450_cp_strict \
  --datasets ctrate,radgenome \
  --expected_cases_map ctrate=450,radgenome=450 \
  --save_report /content/drive/MyDrive/Data/outputs_stage0_4_full450_cp_strict/validation_report.json
```

## 5) 期望输出结构

输出目录下至少有：

- `summary.csv`
- `ctrate_case_summary.csv`
- `radgenome_case_summary.csv`
- `run_meta.json`
- `ckpt_probe_report.json`（建议保留，证明 ckpt 通过探测）
- `cases/ctrate/<case_id>/{tokens.npy,tokens.pt,tokens.json,bank_meta.json,trace.jsonl}`
- `cases/radgenome/<case_id>/{tokens.npy,tokens.pt,tokens.json,bank_meta.json,trace.jsonl}`

## 6) 如何判断达标

- `validate_stage0_4_outputs.py` 退出码为 0
- 输出里没有 `expected ... found ...` 的 dataset 级错误
- `ckpt_probe_report.json` 里 `compatible=true`
- `run_meta.json` 中：
  - `ctrate.selected_rows == 450`
  - `radgenome.selected_rows == 450`
  - `cp_strict == true`
  - `text_encoder == "semantic"`

## 7) 常见报错快速定位

- `CP strict mode requires --encoder_ckpt`
  - checkpoint 路径不对或没传
- `ckpt_probe.py` 返回 `compatible=false`
  - 这个 checkpoint 和当前模型签名不匹配（常见是输入/输出通道或 key 前缀不一致）
- `duplicated case_id detected`
  - manifest 的 `case_id` 有重复
- `expected 450 cases, found ...`
  - manifest 实际可用样本不足，或中途跑失败未落完整
- `sentence-transformers is required`
  - 依赖未装，重新执行 pip 安装单元

---

如需只做 smoke（10/10），把 `--max_cases 450` 和 `--expected_cases_per_dataset 450` 改成 10。
