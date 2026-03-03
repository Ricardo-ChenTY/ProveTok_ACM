# ProveTok_ACM 运行说明（给协作者）

这份说明对应当前代码的目标范围：**Stage 0-4**（不调用 LLM 生成）。

## 1. 项目里有什么

当前主流程只做四件事：

1. Stage 0-2：确定性 token bank 构建（SwinUNETR + 八叉树分裂）
2. Stage 3：Router（teacher-forcing，用参考报告句子做查询）
3. Stage 4：Verifier 规则审计（R1-R5）
4. Stage 6：结构化日志落盘（每例可追溯）

关键入口：

- `run_mini_experiment.py`：主运行脚本
- `validate_stage0_4_outputs.py`：结果验收脚本

---

## 2. 数据接口（你要准备什么）

需要两个 manifest CSV（CT-RATE / RadGenome 各一个）。

### 必需列

- `case_id`：病例唯一 ID
- `volume_path`：体数据路径（支持医学影像文件；也支持 `.npy`）
- `report_text`：参考报告文本（用于 teacher-forcing 分句）

### 可选列

- `split`：当你加 `--build_mini` 时用于分层抽样（450/450）

### 体数据说明

- 若是医学影像文件：自动从元数据读取 spacing
- 若是 `.npy`：默认 spacing = `(1.0, 1.0, 1.0)` mm

---

## 3. 怎么运行

## 3.1 一次性跑（含 450/450 mini 构建）

```powershell
python run_mini_experiment.py `
  --ctrate_csv <path_to_ctrate_manifest.csv> `
  --radgenome_csv <path_to_radgenome_manifest.csv> `
  --build_mini `
  --out_dir outputs_stage0_4 `
  --max_cases 450 `
  --device cuda `
  --token_budget_b 64 `
  --k_per_sentence 8 `
  --lambda_spatial 0.3 `
  --tau_iou 0.1 `
  --beta 0.1
```

如果没有 GPU，可把 `--device cpu`。

## 3.2 只跑已有 manifest（不重建 mini）

去掉 `--build_mini` 即可。

---

## 4. 能得到什么（输出文件）

所有输出在 `--out_dir` 下（默认 `outputs_stage0_4`）。

## 4.1 全局汇总

- `summary.csv`
- `ctrate_case_summary.csv`
- `radgenome_case_summary.csv`

## 4.2 每个 case 的核心产物

路径：`outputs_stage0_4/cases/<dataset>/<case_id>/`

- `tokens.npy`：`[B, d]` token 特征
- `tokens.pt`：同上（PyTorch tensor）
- `tokens.json`：每个 token 的结构信息
- `bank_meta.json`：token bank 级元信息
- `trace.jsonl`：逐句路由 + verifier 结果

---

## 5. 每一步的 Input / Output

## Stage 0-2（Deterministic Token Bank）

输入：

- `volume_path` 指向的 3D 体数据
- 配置：`B, depth_max, beta` 等

输出（按 case 落盘）：

- `tokens.npy` / `tokens.pt`
- `tokens.json`（`token_id, level, bbox_3d_voxel, bbox_3d_mm, cached_boundary_flag/params`）
- `bank_meta.json`（`B, depth_max, beta, encoder_name, voxel_spacing, global_bbox`）

## Stage 3（Router, Teacher-Forcing）

输入：

- 参考报告分句得到 `s`
- token bank 的 `f_i` 与 `bbox_i`
- 超参数：`k, lambda_spatial`

计算：

- `v_i = normalize(W_proj f_i)`
- `r_i^(s) = cos(q_s, v_i)`
- 可选空间先验：`+ lambda_spatial * IoU(bbox_i, bbox_anatomy)`
- 取 top-k citations

输出：

- 每句 `topk_token_ids` 和 `topk_scores`（在 `trace.jsonl`）

## Stage 4（Verifier）

输入：

- 原句文本
- 每句 citations（token id）
- token 几何信息（bbox / level）

规则：

- `R1` 侧别一致性
- `R2` 解剖区域一致性
- `R3` 深度层级一致性
- `R4` 大小/范围合理性
- `R5` 否定处理

输出：

- 每句 `violations[]`（在 `trace.jsonl`）

## Stage 6（统一日志）

输入：

- Stage 0-4 全部中间结果

输出：

- `trace.jsonl` 第一行为 case_meta：
  - `B, k, B_plan, lambda_spatial, tau_IoU, ell_coarse, beta`
- 后续每句记录：
  - `sentence_text, q_s, topk_token_ids, topk_scores, violations`

---

## 6. 如何验收“达到预期”

运行验收脚本：

```powershell
python validate_stage0_4_outputs.py `
  --out_dir outputs_stage0_4 `
  --datasets ctrate,radgenome `
  --save_report outputs_stage0_4\validation_report.json
```

通过标准（脚本自动检查）：

1. 每 case 文件齐全：`tokens.* + bank_meta + trace`
2. `tokens.json / bank_meta.json` 字段完整
3. `trace.jsonl` 有 case_meta + sentence 记录
4. `topk_token_ids` 合法且与 token bank 对齐
5. `topk_scores` 数值格式正确（并做排序检查）

---

## 7. 常见问题

1. 报错 `Missing column ...`  
说明 manifest 缺少必需列，请检查 `case_id, volume_path, report_text`。

2. 跑得慢  
先把 `--max_cases` 调小（如 10）做 smoke test，再放到 450。

3. 显存不足  
改 `--device cpu`，或减小 `--resize_d/h/w`。

4. `.npy` spacing 不准确  
当前默认 `(1,1,1)` mm；如要真实 mm，请用带元数据的医学影像格式。
