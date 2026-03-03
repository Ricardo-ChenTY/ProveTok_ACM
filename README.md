# ProveTok Stage0-4 Only (No LLM Generation)

中文可执行说明：[`RUN_GUIDE_CN.md`](./RUN_GUIDE_CN.md)

This codebase is intentionally narrowed to your requested scope:

1. Stage 0-2 deterministic token bank
2. Stage 3 router with teacher-forcing sentences
3. Stage 4 verifier
4. Stage 6 structured reproducibility logs

No LLM generation is used in the main run path.

## Contract Priority

- `provetok_v3.0_oral_rewrite_v3.4.2_oral_ready.tex`
- `CP_priorjudgement (2).pdf`

If math differs, CP wins.

## A) Stage 0-2 Outputs Per Case

Saved under:

- `outputs_stage0_4/cases/<dataset>/<case_id>/tokens.npy`
- `outputs_stage0_4/cases/<dataset>/<case_id>/tokens.pt`
- `outputs_stage0_4/cases/<dataset>/<case_id>/tokens.json`
- `outputs_stage0_4/cases/<dataset>/<case_id>/bank_meta.json`

`tokens.json` fields include:

- `token_id`
- `level`
- `bbox_3d_voxel`
- `bbox_3d_mm`
- `cached_boundary_flag`
- `cached_boundary_params`

`bank_meta.json` includes:

- `B`
- `depth_max`
- `beta`
- `encoder_name`
- `voxel_spacing_mm_xyz`
- `global_bbox_voxel`
- `global_bbox_mm`

## B) Stage 3 Router (Teacher-Forcing)

For each sentence from reference report:

- `v_i = normalize(W_proj f_i)`
- `r_i^(s) = cos(q_s, v_i)`
- optional `+ lambda_spatial * IoU(bbox_i, bbox_anatomy)`
- select top-k (default `k=8`)

No LLM call is needed.

## C) Stage 4 Verifier

Rules enabled:

- `R1` laterality consistency
- `R2` anatomy IoU consistency
- `R3` depth-level consistency
- `R4` size/range consistency
- `R5` negation handling

## D) Stage 6 Unified Logs

Per case:

- `trace.jsonl` with
  - case-level config line: `B, k, B_plan, lambda_spatial, tau_IoU, ell_coarse, beta`
  - sentence lines: `sentence_text, q_s, topk_token_ids, topk_scores, violations`

## Acceptance Script

- `validate_stage0_4_outputs.py`

Run:

```powershell
python validate_stage0_4_outputs.py `
  --out_dir outputs_stage0_4 `
  --datasets ctrate,radgenome `
  --save_report outputs_stage0_4\validation_report.json
```

## Main Entry

- `run_mini_experiment.py`

Default expected manifest columns:

- `case_id`
- `volume_path`
- `report_text`

Optional:

- `split` (used for stratified mini subset when `--build_mini` is set)
- `spacing_x`, `spacing_y`, `spacing_z` are not required because spacing is read from image metadata.
  If input is `.npy`, spacing defaults to `(1.0, 1.0, 1.0)` mm.

## Run

```powershell
python run_mini_experiment.py `
  --ctrate_csv path\\to\\ctrate_manifest.csv `
  --radgenome_csv path\\to\\radgenome_manifest.csv `
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
