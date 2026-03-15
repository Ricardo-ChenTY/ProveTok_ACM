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


def build_dataset(
    cases_dir: str,
    text_encoder,
    manifest: Optional[str] = None,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Returns list of (query_emb [d_q], pos_token_feats [k, d_v]) pairs.

    If manifest is provided, only load cases listed in the manifest file.
    Each line in manifest: dataset/case_id (e.g. ctrate/train_10123_a_1)
    """
    root = Path(cases_dir)
    pairs: List[Tuple[torch.Tensor, torch.Tensor]] = []

    if manifest:
        manifest_path = Path(manifest)
        with manifest_path.open() as f:
            case_keys = [line.strip() for line in f if line.strip()]
        case_dirs = [root / key for key in case_keys]
        print(f"Manifest {manifest}: {len(case_dirs)} cases")
    else:
        case_dirs = [d.parent for d in root.rglob("trace.jsonl")]
        print(f"Found {len(case_dirs)} trace files under {cases_dir}")

    for case_dir in case_dirs:
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

def _eval_loss(
    pairs: List[Tuple[torch.Tensor, torch.Tensor]],
    w_proj: nn.Linear,
    batch_size: int,
    tau: float,
    device: str,
) -> float:
    """Compute average InfoNCE loss on a dataset (no grad)."""
    w_proj.eval()
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i : i + batch_size]
            if len(batch) < 2:
                continue
            queries = torch.stack([p[0] for p in batch]).to(device)
            pos_means = torch.stack([p[1].mean(0) for p in batch]).to(device)
            loss = infonce_batch(queries, pos_means, w_proj, tau=tau)
            total_loss += loss.item()
            n_batches += 1
    w_proj.train()
    return total_loss / max(n_batches, 1)


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
    val_pairs: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    patience: int = 10,
    out_dir: Optional[Path] = None,
) -> Tuple[nn.Linear, List[float], List[float]]:
    torch.manual_seed(seed)
    random.seed(seed)

    w_proj = nn.Linear(d_v, d_q, bias=False).to(device)
    nn.init.orthogonal_(w_proj.weight)

    optimizer = optim.AdamW(w_proj.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_log: List[float] = []
    val_log: List[float] = []
    best_val_loss = float("inf")
    best_epoch = 0
    no_improve = 0

    for epoch in range(1, epochs + 1):
        random.shuffle(pairs)
        epoch_loss = 0.0
        n_batches = 0

        w_proj.train()
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
        avg_train = epoch_loss / max(n_batches, 1)
        train_log.append(avg_train)

        # Validation
        avg_val = 0.0
        if val_pairs:
            avg_val = _eval_loss(val_pairs, w_proj, batch_size, tau, device)
            val_log.append(avg_val)

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                best_epoch = epoch
                no_improve = 0
                # Save best checkpoint
                if out_dir:
                    torch.save(w_proj.weight.data.cpu(), out_dir / "w_proj_best.pt")
            else:
                no_improve += 1
        else:
            val_log.append(avg_train)

        if epoch % 5 == 0 or epoch == 1:
            val_str = f" | val={avg_val:.4f}" if val_pairs else ""
            best_str = f" | best_val={best_val_loss:.4f}@{best_epoch}" if val_pairs else ""
            print(
                f"Epoch {epoch:4d}/{epochs} | train={avg_train:.4f}{val_str}"
                f" | lr={scheduler.get_last_lr()[0]:.2e}{best_str}"
            )

        # Early stopping
        if val_pairs and no_improve >= patience:
            print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    # Load best checkpoint if available
    if val_pairs and out_dir and (out_dir / "w_proj_best.pt").exists():
        best_weight = torch.load(out_dir / "w_proj_best.pt", weights_only=True)
        w_proj.weight.data.copy_(best_weight.to(device))
        print(f"Loaded best checkpoint from epoch {best_epoch} (val_loss={best_val_loss:.4f})")

    return w_proj, train_log, val_log


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Train W_proj (InfoNCE) for Stage 3c.")
    parser.add_argument("--cases_dir", type=str, required=True,
                        help="Root dir containing per-case subdirs with trace.jsonl + tokens.pt.")
    parser.add_argument("--train_manifest", type=str, default=None,
                        help="Manifest file listing train cases (one per line: dataset/case_id).")
    parser.add_argument("--val_manifest", type=str, default=None,
                        help="Manifest file listing val cases for early stopping.")
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
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience (epochs without val improvement).")
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

    # Build train dataset
    train_pairs = build_dataset(args.cases_dir, text_encoder, manifest=args.train_manifest)
    if not train_pairs:
        raise RuntimeError(
            f"No training pairs found. "
            "Run Stage 0-4 first to generate trace.jsonl + tokens.pt files."
        )

    # Build val dataset (optional)
    val_pairs = None
    if args.val_manifest:
        print("Loading validation set...")
        val_pairs = build_dataset(args.cases_dir, text_encoder, manifest=args.val_manifest)
        print(f"Val pairs: {len(val_pairs)}")

    # Probe d_v from first pair
    d_v = train_pairs[0][1].shape[-1]
    print(f"Dims: d_q={d_q}, d_v={d_v}, n_train={len(train_pairs)}")

    w_proj, train_losses, val_losses = train(
        pairs=train_pairs,
        d_q=d_q,
        d_v=d_v,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        tau=args.tau,
        device=args.device,
        seed=args.seed,
        val_pairs=val_pairs,
        patience=args.patience,
        out_dir=out_dir,
    )

    # Save final
    torch.save(w_proj.weight.data.cpu(), out_dir / "w_proj.pt")
    np.save(str(out_dir / "w_proj.npy"), w_proj.weight.data.cpu().numpy())
    log_data = {
        "d_q": d_q,
        "d_v": d_v,
        "n_train_pairs": len(train_pairs),
        "n_val_pairs": len(val_pairs) if val_pairs else 0,
        "epochs_run": len(train_losses),
        "epochs_max": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "tau": args.tau,
        "patience": args.patience,
        "final_train_loss": train_losses[-1] if train_losses else None,
        "final_val_loss": val_losses[-1] if val_losses else None,
        "best_val_loss": min(val_losses) if val_losses else None,
        "train_loss_curve": train_losses,
        "val_loss_curve": val_losses,
        "text_encoder": args.text_encoder,
        "text_encoder_model": args.text_encoder_model,
        "train_manifest": args.train_manifest,
        "val_manifest": args.val_manifest,
    }
    with (out_dir / "train_log.json").open("w", encoding="utf-8") as f:
        json.dump(log_data, f, ensure_ascii=False, indent=2)

    print(f"\nSaved W_proj to {out_dir}/w_proj.pt  (shape {list(w_proj.weight.shape)})")
    print(f"Final train loss: {train_losses[-1]:.4f}")
    if val_losses:
        print(f"Best val loss: {min(val_losses):.4f}")
    print(
        f"\nTo use in Stage 0-4, pass w_proj to Router:\n"
        f"  import torch\n"
        f"  w = torch.load('{out_dir}/w_proj.pt').tolist()\n"
        f"  router = Router(cfg=cfg.router, text_encoder=..., w_proj=w)"
    )


if __name__ == "__main__":
    main()
