"""
Evaluate trained W_proj vs identity baseline on test set.

Usage:
    python Scripts/eval_wprojection_test.py \
        --cases_dir outputs/stage0_5_llama_450/cases \
        --test_manifest manifests/split_seed42/test.txt \
        --w_proj_path outputs_wprojection/w_proj.pt \
        --device cuda
"""
from __future__ import annotations

import argparse
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn

from train_wprojection import build_dataset, infonce_batch, _eval_loss
from ProveTok_Main_experiment.text_encoder import make_text_encoder


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate W_proj on test set.")
    parser.add_argument("--cases_dir", type=str, default="outputs/stage0_5_llama_450/cases")
    parser.add_argument("--test_manifest", type=str, default="manifests/split_seed42/test.txt")
    parser.add_argument("--w_proj_path", type=str, default="outputs_wprojection/w_proj.pt")
    parser.add_argument("--text_encoder_model", type=str,
                        default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--text_encoder_device", type=str, default="cpu")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--tau", type=float, default=0.07)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    # Load text encoder
    print("Loading text encoder...")
    enc = make_text_encoder("semantic", st_model_name=args.text_encoder_model,
                            st_device=args.text_encoder_device)

    # Load test data
    print(f"Loading test data from manifest: {args.test_manifest}")
    test_pairs = build_dataset(args.cases_dir, enc, manifest=args.test_manifest)
    if not test_pairs:
        raise RuntimeError("No test pairs found.")
    print(f"Test pairs: {len(test_pairs)}")

    # Probe dims
    d_q = test_pairs[0][0].shape[0]
    d_v = test_pairs[0][1].shape[-1]
    print(f"Dims: d_q={d_q}, d_v={d_v}")

    # Load trained W_proj
    w = torch.load(args.w_proj_path, weights_only=True)
    w_proj = nn.Linear(d_v, d_q, bias=False).to(args.device)
    w_proj.weight.data.copy_(w)
    print(f"Loaded trained W_proj from {args.w_proj_path} (shape {list(w.shape)})")

    # Build identity baseline
    w_id = nn.Linear(d_v, d_q, bias=False).to(args.device)
    nn.init.eye_(w_id.weight[:min(d_q, d_v), :min(d_q, d_v)])
    print("Built identity baseline")

    # Evaluate
    loss_trained = _eval_loss(test_pairs, w_proj, args.batch_size, args.tau, args.device)
    loss_identity = _eval_loss(test_pairs, w_id, args.batch_size, args.tau, args.device)

    print(f"\n{'='*50}")
    print(f"Test loss (identity W_proj):  {loss_identity:.4f}")
    print(f"Test loss (trained W_proj):   {loss_trained:.4f}")
    print(f"Improvement: {loss_identity - loss_trained:.4f} ({(loss_identity - loss_trained) / loss_identity * 100:.1f}%)")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
