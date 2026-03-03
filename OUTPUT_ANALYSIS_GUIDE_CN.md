# 输出分析与换机操作指南（Stage0-4）

这份文档给“换电脑继续分析”使用。目标是让你或协作者在新机器上快速判断：

1. 当前程序在做什么  
2. 输出是否符合预期  
3. 怎么在本地/Colab做分析  

---

## 1. 当前程序形态（你们现在跑的是什么）

当前主流程是 **Stage0-4（无LLM生成）**：

1. Stage 0-2：确定性 token bank（SwinUNETR + 八叉树分裂）
2. Stage 3：router（teacher-forcing，用参考报告句子做查询）
3. Stage 4：verifier（R1-R5规则审计）
4. Stage 6：日志落盘（trace.jsonl）

核心入口脚本：

- `run_mini_experiment.py`
- `validate_stage0_4_outputs.py`

---

## 2. 输出目录应该长什么样（450 小样本）

假设 `--out_dir outputs_stage0_4_450`，典型结构如下：

```text
outputs_stage0_4_450/
  summary.csv
  ctrate_case_summary.csv
  radgenome_case_summary.csv
  cases/
    ctrate/
      <case_id>/
        tokens.npy
        tokens.pt
        tokens.json
        bank_meta.json
        trace.jsonl
    radgenome/
      <case_id>/
        tokens.npy
        tokens.pt
        tokens.json
        bank_meta.json
        trace.jsonl
  cache/
    ctrate/stage0/*.npz
    ctrate/stage1/*.npy
    radgenome/stage0/*.npz
    radgenome/stage1/*.npy
```

---

## 3. 按你们预设参数时，应期待的结果（450/450）

常见预设（你们之前在用）：

- `B=64`
- `k=8`
- `lambda_spatial=0.3`
- `tau_iou=0.1`
- `beta=0.1`
- `resize=(128,128,128)`

在该预设下，**每个 case 的预期**：

1. `bank_meta.json["B"]` 通常等于 `64`
2. `tokens.npy` 形状应为 `[B, d]`（`B` 与 `bank_meta` 一致）
3. `trace.jsonl` 第一行 `case_meta` 中应包含  
   `B, k, B_plan, lambda_spatial, tau_IoU, ell_coarse, beta`
4. 每个 `sentence` 行：
   - `topk_token_ids` 长度通常为 `k=8`（除非候选不足）
   - `topk_scores` 与 `topk_token_ids` 等长
   - `violations` 为数组（可空）

此外对 450 小样本，若你是 `--build_mini --max_cases 450`：

1. `summary.csv` 中 `ctrate` 的 `cases` 应接近/等于 `450`
2. `summary.csv` 中 `radgenome` 的 `cases` 应接近/等于 `450`
3. 两个数据集的 case 目录数应与上面一致

---

## 4. 快速验收（强烈建议每次都跑）

```powershell
python validate_stage0_4_outputs.py `
  --out_dir outputs_stage0_4_450 `
  --datasets ctrate,radgenome `
  --expected_cases_per_dataset 450 `
  --save_report outputs_stage0_4_450\validation_report.json
```

结果解释：

- `Failed: 0`：结构层面通过
- `Failed > 0`：先看 `validation_report.json` 的 `errors`

---

## 5. 本地快速分析（不读影像，只读结果）

```python
import pandas as pd, json, glob, os

out_dir = "outputs_stage0_4_450"
summary = pd.read_csv(os.path.join(out_dir, "summary.csv"))
print(summary.groupby("dataset", as_index=False).agg(
    cases=("case_id","count"),
    mean_tokens=("n_tokens","mean"),
    mean_sentences=("n_sentences","mean"),
    total_violations=("n_violations","sum"),
))

# 统计每类规则触发次数
rule_count = {}
for p in glob.glob(os.path.join(out_dir, "cases", "*", "*", "trace.jsonl")):
    with open(p, "r", encoding="utf-8") as f:
        lines = [json.loads(x) for x in f if x.strip()]
    for row in lines[1:]:
        for v in row.get("violations", []):
            rid = v.get("rule_id", "UNKNOWN")
            rule_count[rid] = rule_count.get(rid, 0) + 1
print(rule_count)
```

---

## 6. Colab 分析建议（你要出门换电脑时）

建议上传“轻量分析包”到 Drive：

- 必留：`summary.csv`, `*_case_summary.csv`, `trace.jsonl`, `bank_meta.json`
- 可选：`tokens.json`
- 可不上传：`tokens.npy`, `tokens.pt`, 原始 `.nii.gz`

挂载 Drive 示例：

```python
from google.colab import drive
drive.mount('/content/drive')
```

如果你上传的是 `.rar`：

```bash
!apt-get -y install unrar
!unrar x "/content/drive/MyDrive/<path>/outputs_stage0_4_450.rar" /content/data/
```

---

## 7. 换机 Checklist（10分钟版）

1. `git pull` 拉最新代码  
2. 创建环境：`conda env create -f environment.yaml`  
3. `python run_mini_experiment.py --help` 确认入口正常  
4. 跑一小批（`--max_cases 5`）做 smoke test  
5. 跑验收脚本确认结构正确  
6. 再跑完整分析

---

## 8. 常见异常与定位

1. `ModuleNotFoundError`  
通常是包名/目录名不一致，先执行 `python run_mini_experiment.py --help` 复测。

2. `Missing column 'volume_path'`  
manifest 列名不一致，使用：
`--volume_col --report_col --case_id_col` 指定。

3. `Failed > 0` 但程序跑完了  
说明格式或字段异常，按 `validation_report.json` 定位具体 case 修正。
