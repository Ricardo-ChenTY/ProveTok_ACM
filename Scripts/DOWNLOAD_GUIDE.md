# ProveTok Data Download Guide / ProveTok 数据下载指南

## Directory Structure / 目录结构

```
ProveTok_ACM/
+-- ProveTok_main_exp/          # Main experiment scripts / 主要实验脚本
+-- ProveTok_data_download/     # Downloaded .nii.gz (auto-created)
|   +-- CT-RATE/
|   |   +-- train/
|   |   +-- valid/
|   +-- RadGenome-ChestCT/
|       +-- train/
|       +-- valid/
+-- Scripts/
|   +-- _download_core.py       # Shared logic (do not run directly)
|   +-- download_smoke_450.py   # Smoke test (~450 samples)
|   +-- download_full_3000.py   # Full download (~3000 samples)
|   +-- DOWNLOAD_GUIDE.md       # This file
+-- ...
```

---

## Prerequisites / 前置条件

### 1. HuggingFace Token / HuggingFace 令牌

You need a HuggingFace access token with read permission for:

你需要一个拥有以下数据集读取权限的 HuggingFace 令牌：

- `ibrahimhamamci/CT-RATE`
- `RadGenome/RadGenome-ChestCT`

Set it as an environment variable before running / 运行前设置环境变量：

```bash
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxx"
```

### 2. Python Dependencies / Python 依赖

```bash
pip install huggingface_hub
```

---

## Quick Start / 快速开始

### Option A: Smoke Test - ~450 Samples / 小样本测试 - 约 450 个

For quick validation. Logs progress every 10 files.

用于快速验证。每 10 个文件打印一次进度。

```bash
cd ProveTok_ACM/Scripts

# Both datasets / 两个数据集
python download_smoke_450.py \
    --jobs ct,rad \
    --ct_csv  ../CT-RATE/outputs/smoke_dataset_450_broad.csv \
    --rad_csv ../RadGenome-ChestCT/outputs/smoke_450_broad.csv

# CT-RATE only / 仅 CT-RATE
python download_smoke_450.py \
    --jobs ct \
    --ct_csv ../CT-RATE/outputs/smoke_dataset_450_broad.csv
```

### Option B: Full Download - ~3000 Samples / 全量下载 - 约 3000 个

For full experiments. Logs progress every 100 files.

用于正式实验。每 100 个文件打印一次进度。

```bash
cd ProveTok_ACM/Scripts

python download_full_3000.py \
    --jobs ct,rad \
    --ct_csv  ../CT-RATE/outputs/full_dataset_3000.csv \
    --rad_csv ../RadGenome-ChestCT/outputs/full_3000.csv
```

---

## Arguments / 参数说明

Both scripts accept the same arguments. The only built-in difference is the default log interval (10 vs 100).

两个脚本参数完全一致，唯一内置区别是默认日志间隔（10 vs 100）。

| Argument | Description / 说明 | Default / 默认 |
|---|---|---|
| `--jobs` | Which datasets: `ct`, `rad`, or `ct,rad` / 下载哪些数据集 | `ct,rad` |
| `--ct_csv` | Path to CT-RATE input CSV / CT-RATE 输入 CSV 路径 | (required if ct) |
| `--rad_csv` | Path to RadGenome input CSV / RadGenome 输入 CSV 路径 | (required if rad) |
| `--out_dir` | Override output directory / 覆盖输出目录 | `../ProveTok_data_download` |
| `--log_interval` | Print progress every N files / 每 N 个文件打印进度 | 10 (450) / 100 (3000) |

---

## How It Works / 工作原理

1. **Clean CSV / 清洗 CSV** - Deduplicates rows, normalizes case IDs, assigns train/valid splits. Outputs `*_clean.csv`.

   去重、归一化 CaseID、标记 train/valid 分组。生成 `*_clean.csv`。

2. **Match remote files / 匹配远程文件** - Lists all `.nii.gz` in the HuggingFace repo and matches to CSV entries. Prefers `*_fixed/` or `*_preprocessed/` versions.

   列出仓库中所有 `.nii.gz` 文件并匹配。优先选择 `*_fixed/` 或 `*_preprocessed/` 版本。

3. **Download .nii.gz / 下载** - Downloads **directly** to target dir using `local_dir` (no cache duplication, no double disk usage). Skips existing files (resumable).

   使用 `local_dir` **直接**下载到目标目录（无缓存副本，避免双倍磁盘占用）。已存在的文件自动跳过。

4. **Generate manifest / 生成清单** - Outputs `*_manifest.csv` tracking status. If any files are missing or failed, the script prints guidance pointing to the manifest.

   输出 `*_manifest.csv`。如有缺失或失败，脚本会打印提示并指向清单文件。

---

## Troubleshooting / 常见问题

**Q: `HF_TOKEN not set` error**

Run `export HF_TOKEN="hf_..."` before the script. On servers, add to `~/.bashrc`.

请先执行 `export HF_TOKEN="hf_..."`。服务器上建议写入 `~/.bashrc`。

**Q: Some files show `missing_remote`**

Check the manifest CSV for specific missing items and verify spelling in the input CSV.

请查看 manifest CSV 定位具体缺失项，并检查输入 CSV 拼写。

**Q: Download is slow / 下载速度慢**

```bash
pip install hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1
```

**Q: Want to retry failed downloads / 重试失败的下载**

Just re-run the script. Already-downloaded files are skipped automatically.

直接重新运行即可，已下载的文件会自动跳过。

---

## Code Structure / 代码结构

The two entry scripts (`download_smoke_450.py`, `download_full_3000.py`) are thin wrappers. All logic lives in `_download_core.py`. To modify download behavior, only edit `_download_core.py`.

两个入口脚本是薄封装，所有逻辑在 `_download_core.py` 中。如需修改下载行为，只编辑 `_download_core.py`。
