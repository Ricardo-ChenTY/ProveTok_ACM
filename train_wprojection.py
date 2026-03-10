"""
Train W_proj via InfoNCE (CP .tex Eq.12) for Stage 3c Token-Gated Generation.

Uses existing experiment outputs (trace.jsonl + tokens.pt) as training data.
No new CT data needed — leverages token banks already built by Stage 0-4.

Usage:
    python train_wprojection.py \
        --cases_dir outputs_stage0_4/cases \
        --out_dir outputs_wprojection \
        --text_encoder semantic \
        --epochs 50 \
        --lr 1e-3

Output:
    outputs_wprojection/w_proj.pt   -- torch tensor [text_dim, token_dim]
    outputs_wprojection/w_proj.npy  -- numpy array
    outputs_wprojection/train_log.json
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ProveTok_Main_experiment.text_encoder import make_text_encoder


# ──────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────

def _load_case(case_dir: Path):
    """Load (sentence_text, topk_token_ids, token_features) for one case."""
    trace_file = case_dir / "trace.jsonl"
    tokens_pt = case_dir / "tokens.pt"
    if not trace_file.exists() or not tokens_pt.exists():
        return None

    token_feats = torch.load(tokens_pt, weights_only=True)  # [B, d_v]
    sentences = []
    with trace_file.open(encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("type") != "sentence":
                continue
            text = str(obj.get("sentence_text", ""))
            topk_ids = [int(x) for x in obj.get("topk_token_ids", [])]
            if text and topk_ids:
                sentences.append((text, topk_ids))
    return token_feats, sentences


def build_dataset(cases_dir: str, text_encoder) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Returns list of (query_emb [d_q], pos_token_feats [k, d_v]) pairs.
    """
    root = Path(cases_dir)
    pairs: List[Tuple[torch.Tensor, torch.Tensor]] = []

    case_dirs = [d for d in root.rglob("trace.jsonl")]
    print(f"Found {len(case_dirs)} trace files under {cases_dir}")

    for trace_path in case_dirs:
        case_dir = trace_path.parent
        result = _load_case(case_dir)
        if result is None:
            continue
        token_feats, sentences = result

        for text, topk_ids in sentences:
            # Filter valid token ids
            valid_ids = [i for i in topk_ids if i < len(token_feats)]
            if not valid_ids:
                continue
            q = torch.tensor(text_encoder(text), dtype=torch.float32)  # [d_q]
            pos_feats = token_feats[valid_ids]  # [k, d_v]
            pairs.append((q, pos_feats))

    print(f"Built {len(pairs)} (query, positive_tokens) training pairs.")
    return pairs


# ──────────────────────────────────────────────
# InfoNCE loss (batch version)
# ──────────────────────────────────────────────

def infonce_batch(
    queries: torch.Tensor,        # [N, d_q]
    pos_feats: torch.Tensor,      # [N, d_v]  (mean of positive tokens per query)
    w_proj: nn.Linear,
    tau: float = 0.07,
) -> torch.Tensor:
    """
    Multi-positive InfoNCE (CP .tex Eq.12).
    Treats the mean of cited tokens as the positive for each query.
    All other queries in the batch are negatives.
    """
    proj = w_proj(pos_feats)                              # [N, d_q]
    proj = nn.functional.normalize(proj, dim=-1)
    q_norm = nn.functional.normalize(queries, dim=-1)    # [N, d_q]
    logits = (q_norm @ proj.T) / tau                      # [N, N]
    labels = torch.arange(len(queries), device=queries.device)
    return nn.functional.cross_entropy(logits, labels)


# ──────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────

def train(
    pairs: List[Tuple[torch.Tensor, torch.Tensor]],
    d_q: int,
    d_v: int,
    epochs: int,
    batch_size: int,
    lr: float,
    tau: float,
    device: str,
    seed: int,
) -> Tuple[nn.Linear, List[float]]:
    torch.manual_seed(seed)
    random.seed(seed)

    w_proj = nn.Linear(d_v, d_q, bias=False).to(device)
    nn.init.orthogonal_(w_proj.weight)

    optimizer = optim.AdamW(w_proj.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    loss_log: List[float] = []

    for epoch in range(1, epochs + 1):
        random.shuffle(pairs)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, len(pairs), batch_size):
            batch = pairs[i : i + batch_size]
            if len(batch) < 2:
                continue  # InfoNCE needs at least 2 samples

            queries = torch.stack([p[0] for p in batch]).to(device)          # [B, d_q]
            pos_means = torch.stack([p[1].mean(0) for p in batch]).to(device)  # [B, d_v]

            optimizer.zero_grad()
            loss = infonce_batch(queries, pos_means, w_proj, tau=tau)
            loss.backward()
            nn.utils.clip_grad_norm_(w_proj.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg = epoch_loss / max(n_batches, 1)
        loss_log.append(avg)
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:4d}/{epochs} | loss={avg:.4f} | lr={scheduler.get_last_lr()[0]:.2e}")

    return w_proj, loss_log


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Train W_proj (InfoNCE) for Stage 3c.")
    parser.add_argument("--cases_dir", type=str, required=True,
                        help="Root dir containing per-case subdirs with trace.jsonl + tokens.pt.")
    parser.add_argument("--out_dir", type=str, default="outputs_wprojection")
    parser.add_argument("--text_encoder", type=str, default="semantic",
                        choices=("hash", "semantic"),
                        help="Must match the encoder used during Stage 0-4 run.")
    parser.add_argument("--text_encoder_model", type=str,
                        default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--text_encoder_device", type=str, default="cpu")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--tau", type=float, default=0.07, help="InfoNCE temperature.")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading text encoder: {args.text_encoder}")
    text_encoder = make_text_encoder(
        encoder_type=args.text_encoder,
        st_model_name=args.text_encoder_model,
        st_device=args.text_encoder_device,
    )

    # Probe dims
    _probe_q = text_encoder("hello world")
    d_q = len(_probe_q)

    pairs = build_dataset(args.cases_dir, text_encoder)
    if not pairs:
        raise RuntimeError(
            f"No training pairs found under {args.cases_dir}. "
            "Run Stage 0-4 first to generate trace.jsonl + tokens.pt files."
        )

    # Probe d_v from first pair
    d_v = pairs[0][1].shape[-1]
    print(f"Dims: d_q={d_q}, d_v={d_v}, n_pairs={len(pairs)}")

    w_proj, loss_log = train(
        pairs=pairs,
        d_q=d_q,
        d_v=d_v,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        tau=args.tau,
        device=args.device,
        seed=args.seed,
    )

    # Save
    torch.save(w_proj.weight.data.cpu(), out_dir / "w_proj.pt")
    np.save(str(out_dir / "w_proj.npy"), w_proj.weight.data.cpu().numpy())
    train_log = {
        "d_q": d_q,
        "d_v": d_v,
        "n_pairs": len(pairs),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "tau": args.tau,
        "final_loss": loss_log[-1] if loss_log else None,
        "loss_curve": loss_log,
        "text_encoder": args.text_encoder,
        "text_encoder_model": args.text_encoder_model,
    }
    with (out_dir / "train_log.json").open("w", encoding="utf-8") as f:
        json.dump(train_log, f, ensure_ascii=False, indent=2)

    print(f"\nSaved W_proj to {out_dir}/w_proj.pt  (shape {list(w_proj.weight.shape)})")
    print(f"Final InfoNCE loss: {loss_log[-1]:.4f}")
    print(
        f"\nTo use in Stage 0-4, pass w_proj to Router:\n"
        f"  import torch\n"
        f"  w = torch.load('{out_dir}/w_proj.pt').tolist()\n"
        f"  router = Router(cfg=cfg.router, text_encoder=..., w_proj=w)"
    )


if __name__ == "__main__":
    main()
