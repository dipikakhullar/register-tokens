#!/usr/bin/env python3
"""
train_overcomplete_sae.py

Train an overcomplete TopK SAE (KempnerInstitute/overcomplete) on extracted DINOv2 tokens.

Inputs:
- extract_dir containing shard files saved by your extractor:
    extract_hookXX_shardYYYYY.pt
  Each shard must include:
    - tokens: FloatTensor [T, D] (CPU)
    - image_ix: LongTensor [T] (CPU)  (used for train/val split)
    - token_pos: IntTensor [T] (CPU)  (we drop CLS where token_pos == 0)

Split:
- Deterministic 80/20 split by image id:
    val if (image_ix % 5 == 0), else train

Normalization:
- Compute mean/std over TRAIN tokens only (streaming, one pass over shards).
- Apply the same mean/std to both train and val.
- Save stats to output_dir/norm_stats.pt

Training:
- SAE: TopKSAE(D, 4096, top_k=20)
- Optim: AdamW
- Loss: MSE reconstruction
- Streaming over shards; never loads all tokens at once.
- Saves best checkpoint by validation MSE.

Outputs (output_dir):
- config.json
- norm_stats.pt
- checkpoints/
    - best.pt
    - last.pt
- train_log.jsonl  (step-level metrics)
- split_rule.json  (records %5 rule, and CLS-drop rule)

Usage:
python3 train_overcomplete_sae.py \
  --extract_dir /lambda/nfs/neel/Research/runs/dinov2/imagenet1k_base_reg4_hook06_extract \
  --output_dir  /lambda/nfs/neel/Research/runs/sae/imagenet1k_base_reg4_hook06_overcomplete_topk4096_k20 \
  --device cuda

Dependency:
pip install overcomplete
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from tqdm.auto import tqdm


# ----------------------------
# IO helpers
# ----------------------------
def list_shards(extract_dir: Path) -> List[Path]:
    shards = sorted(extract_dir.glob("extract_hook*_shard*.pt"))
    if not shards:
        raise FileNotFoundError(f"No shards found in {extract_dir} matching extract_hook*_shard*.pt")
    return shards


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, obj: Dict) -> None:
    path.write_text(json.dumps(obj, indent=2))


def append_jsonl(path: Path, obj: Dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(obj) + "\n")


# ----------------------------
# Split + filtering
# ----------------------------
def make_masks(image_ix: torch.Tensor, token_pos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns boolean masks:
      train_mask, val_mask on tokens
    Rules:
      - drop CLS: token_pos != 0
      - val: (image_ix % 5 == 0)
    """
    keep = token_pos != 0
    val = (image_ix % 5 == 0) & keep
    train = (~val) & keep
    return train, val


# ----------------------------
# Streaming mean/std (train only)
# ----------------------------
@torch.no_grad()
def compute_train_mean_std(
    shards: List[Path],
    max_train_tokens_for_stats: Optional[int],
    eps: float,
) -> Dict[str, torch.Tensor]:
    """
    Streaming mean/std over train tokens only using Welford.
    Works on CPU (fast enough, avoids GPU memory).
    Optionally cap number of tokens used for stats for speed (None => all).
    """
    n = 0
    mean = None
    m2 = None

    pbar = tqdm(shards, desc="Norm stats: shards")
    for sp in pbar:
        obj = torch.load(sp, map_location="cpu")
        tokens = obj["tokens"].to(torch.float32)  # [T, D]
        image_ix = obj["image_ix"].to(torch.long)
        token_pos = obj["token_pos"].to(torch.int64)

        train_mask, _val_mask = make_masks(image_ix, token_pos)
        x = tokens[train_mask]  # [Nt, D]
        if x.numel() == 0:
            continue

        if max_train_tokens_for_stats is not None and n >= max_train_tokens_for_stats:
            break
        if max_train_tokens_for_stats is not None:
            remaining = max_train_tokens_for_stats - n
            if x.shape[0] > remaining:
                x = x[:remaining]

        # Welford update in batch form
        if mean is None:
            mean = torch.zeros((x.shape[1],), dtype=torch.float32)
            m2 = torch.zeros((x.shape[1],), dtype=torch.float32)

        batch_n = x.shape[0]
        batch_mean = x.mean(dim=0)
        batch_m2 = ((x - batch_mean) ** 2).sum(dim=0)

        if n == 0:
            mean.copy_(batch_mean)
            m2.copy_(batch_m2)
            n = batch_n
        else:
            delta = batch_mean - mean
            total_n = n + batch_n
            mean += delta * (batch_n / total_n)
            m2 += batch_m2 + (delta ** 2) * (n * batch_n / total_n)
            n = total_n

        pbar.set_postfix({"train_tokens_seen": n})

    if mean is None or m2 is None or n == 0:
        raise RuntimeError("No train tokens found for normalization stats.")

    var = m2 / n
    std = torch.sqrt(var + eps)
    return {"n": torch.tensor(n, dtype=torch.long), "mean": mean, "std": std}


def normalize(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (x - mean) / std


# ----------------------------
# Batching tokens from a shard
# ----------------------------
def iter_token_batches(
    tokens: torch.Tensor,
    token_batch_size: int,
    shuffle: bool,
    rng: np.random.Generator,
) -> Iterable[torch.Tensor]:
    """
    tokens: CPU float32 tensor [N, D]
    yields CPU float32 [B, D]
    """
    n = tokens.shape[0]
    if n == 0:
        return
    if shuffle:
        idx = rng.permutation(n)
        tokens = tokens[idx]
    for i in range(0, n, token_batch_size):
        yield tokens[i : i + token_batch_size]


# ----------------------------
# Training / validation
# ----------------------------
def mse_loss(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    return (x - x_hat).pow(2).mean()


def unpack_overcomplete_output(out):
    """
    overcomplete TopKSAE forward commonly returns SAEOuput with:
      - z_pre, z, x_hat
    Handle tuple/list or attribute variants.
    """
    if isinstance(out, (tuple, list)) and len(out) == 3:
        z_pre, z, x_hat = out
        return x_hat, z_pre, z

    # attribute fallbacks
    def _get_attr(o, names):
        for n in names:
            if hasattr(o, n):
                return getattr(o, n)
        return None

    x_hat = _get_attr(out, ["x_hat", "xhat", "recons", "reconstruction"])
    z_pre = _get_attr(out, ["pre_codes", "z_pre", "precode", "zpre"])
    z = _get_attr(out, ["codes", "z", "code"])

    if x_hat is None or z_pre is None or z is None:
        raise TypeError(
            "Could not unpack overcomplete output. Expected (z_pre, z, x_hat) or attrs like pre_codes/codes/x_hat."
        )
    return x_hat, z_pre, z


@torch.no_grad()
def run_validation_epoch(
    sae,
    shards: List[Path],
    mean: torch.Tensor,
    std: torch.Tensor,
    token_batch_size: int,
    device: str,
    max_val_tokens: Optional[int],
) -> Tuple[float, int]:
    sae.eval()
    total_loss = 0.0
    total_tokens = 0

    pbar = tqdm(shards, desc="Val: shards", leave=False)
    for sp in pbar:
        obj = torch.load(sp, map_location="cpu")
        tokens = obj["tokens"].to(torch.float32)
        image_ix = obj["image_ix"].to(torch.long)
        token_pos = obj["token_pos"].to(torch.int64)

        _train_mask, val_mask = make_masks(image_ix, token_pos)
        x = tokens[val_mask]
        if x.numel() == 0:
            continue

        if max_val_tokens is not None and total_tokens >= max_val_tokens:
            break
        if max_val_tokens is not None:
            remaining = max_val_tokens - total_tokens
            if x.shape[0] > remaining:
                x = x[:remaining]

        for xb_cpu in iter_token_batches(x, token_batch_size, shuffle=False, rng=np.random.default_rng(0)):
            xb = normalize(xb_cpu, mean, std).to(device, non_blocking=True)
            out = sae(xb)
            x_hat, _z_pre, _z = unpack_overcomplete_output(out)
            loss = mse_loss(xb, x_hat).item()

            total_loss += loss * xb.shape[0]
            total_tokens += xb.shape[0]

        pbar.set_postfix({"val_tokens": total_tokens})

    if total_tokens == 0:
        raise RuntimeError("Validation saw 0 tokens. Check split rule and shard contents.")

    return total_loss / total_tokens, total_tokens


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--extract_dir", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)

    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--nb_concepts", type=int, default=4096)
    ap.add_argument("--top_k", type=int, default=20)

    ap.add_argument("--token_batch_size", type=int, default=8192)
    ap.add_argument("--epochs", type=int, default=1)

    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)

    ap.add_argument("--eps", type=float, default=1e-6)

    ap.add_argument("--max_train_tokens_for_stats", type=int, default=0,
                    help="0 means use all train tokens; else cap for faster stats")
    ap.add_argument("--max_val_tokens", type=int, default=0,
                    help="0 means use all val tokens; else cap validation cost")

    ap.add_argument("--val_every_steps", type=int, default=2000)
    ap.add_argument("--save_every_steps", type=int, default=5000)

    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()

    # Import overcomplete
    try:
        from overcomplete.sae import TopKSAE
    except Exception:
        from overcomplete import TopKSAE  # type: ignore

    extract_dir = Path(args.extract_dir)
    output_dir = Path(args.output_dir)
    ckpt_dir = output_dir / "checkpoints"
    safe_mkdir(output_dir)
    safe_mkdir(ckpt_dir)

    shards = list_shards(extract_dir)

    config = {
        "extract_dir": str(extract_dir),
        "output_dir": str(output_dir),
        "device": args.device,
        "nb_concepts": args.nb_concepts,
        "top_k": args.top_k,
        "token_batch_size": args.token_batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "val_every_steps": args.val_every_steps,
        "save_every_steps": args.save_every_steps,
        "seed": args.seed,
        "split_rule": "val if image_ix % 5 == 0 (after dropping CLS token_pos==0)",
        "normalization": "train-only mean/std via streaming Welford; applied to train+val",
    }
    save_json(output_dir / "config.json", config)
    save_json(
        output_dir / "split_rule.json",
        {"val_rule": "image_ix % 5 == 0", "drop_cls_rule": "token_pos != 0", "val_frac": 0.2},
    )

    # Normalization stats
    max_stats = None if args.max_train_tokens_for_stats == 0 else int(args.max_train_tokens_for_stats)
    print("[1/4] Computing train normalization stats ...", flush=True)
    t_stats0 = time.time()
    stats = compute_train_mean_std(shards, max_stats, eps=args.eps)
    t_stats = time.time() - t_stats0
    torch.save(stats, output_dir / "norm_stats.pt")
    print(
        f"[1/4] Done. train_tokens_seen={int(stats['n'])} "
        f"(elapsed {t_stats/60:.1f} min)  std_mean={float(stats['std'].mean()):.6f}",
        flush=True,
    )

    mean = stats["mean"]
    std = stats["std"]

    # Infer D from meta or from first shard
    first = torch.load(shards[0], map_location="cpu")
    D = int(first["tokens"].shape[1])

    print("[2/4] Building model + optimizer ...", flush=True)
    sae = TopKSAE(D, args.nb_concepts, top_k=args.top_k, device=args.device).to(args.device)
    opt = torch.optim.AdamW(sae.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    log_path = output_dir / "train_log.jsonl"
    if log_path.exists():
        log_path.unlink()

    global_step = 0
    best_val = float("inf")

    max_val_tokens = None if args.max_val_tokens == 0 else int(args.max_val_tokens)
    rng = np.random.default_rng(args.seed)

    print("[3/4] Training ...", flush=True)
    t_train0 = time.time()

    for epoch in range(1, args.epochs + 1):
        sae.train()
        epoch_pbar = tqdm(shards, desc=f"Train: shards (epoch {epoch}/{args.epochs})")
        for sp in epoch_pbar:
            obj = torch.load(sp, map_location="cpu")
            tokens = obj["tokens"].to(torch.float32)
            image_ix = obj["image_ix"].to(torch.long)
            token_pos = obj["token_pos"].to(torch.int64)

            train_mask, _val_mask = make_masks(image_ix, token_pos)
            x_train = tokens[train_mask]
            if x_train.numel() == 0:
                continue

            # Shuffle within shard for SGD
            for xb_cpu in iter_token_batches(x_train, args.token_batch_size, shuffle=True, rng=rng):
                global_step += 1

                xb = normalize(xb_cpu, mean, std).to(args.device, non_blocking=True)

                opt.zero_grad(set_to_none=True)
                out = sae(xb)
                x_hat, _z_pre, z = unpack_overcomplete_output(out)
                loss = mse_loss(xb, x_hat)

                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite loss at step {global_step}: {loss.item()}")

                loss.backward()
                opt.step()

                with torch.no_grad():
                    mean_nnz = float((z > 0).sum(dim=1).float().mean().item())

                # log
                append_jsonl(
                    log_path,
                    {
                        "time": time.time(),
                        "epoch": epoch,
                        "step": global_step,
                        "train_loss": float(loss.item()),
                        "mean_nnz": mean_nnz,
                        "batch_tokens": int(xb.shape[0]),
                    },
                )

                if global_step % 200 == 0:
                    epoch_pbar.set_postfix({"step": global_step, "loss": f"{loss.item():.4f}", "nnz": f"{mean_nnz:.2f}"})

                # periodic validation
                if args.val_every_steps > 0 and (global_step % args.val_every_steps == 0):
                    print(f"\n[VAL] step {global_step} ...", flush=True)
                    t_val0 = time.time()
                    val_loss, val_tokens = run_validation_epoch(
                        sae=sae,
                        shards=shards,
                        mean=mean,
                        std=std,
                        token_batch_size=args.token_batch_size,
                        device=args.device,
                        max_val_tokens=max_val_tokens,
                    )
                    t_val = time.time() - t_val0
                    print(
                        f"[VAL] step {global_step}  val_loss={val_loss:.6f}  val_tokens={val_tokens} "
                        f"(elapsed {t_val/60:.1f} min)",
                        flush=True,
                    )
                    append_jsonl(
                        log_path,
                        {"time": time.time(), "epoch": epoch, "step": global_step, "val_loss": float(val_loss)},
                    )

                    # save best
                    if val_loss < best_val:
                        best_val = val_loss
                        torch.save(
                            {"sae_state": sae.state_dict(), "opt_state": opt.state_dict(), "step": global_step, "best_val": best_val},
                            ckpt_dir / "best.pt",
                        )
                        print(f"[CKPT] saved best.pt (best_val={best_val:.6f})", flush=True)

                # periodic last checkpoint
                if args.save_every_steps > 0 and (global_step % args.save_every_steps == 0):
                    torch.save(
                        {"sae_state": sae.state_dict(), "opt_state": opt.state_dict(), "step": global_step, "best_val": best_val},
                        ckpt_dir / "last.pt",
                    )
                    print(f"[CKPT] saved last.pt (step={global_step})", flush=True)

        # end epoch checkpoint
        torch.save(
            {"sae_state": sae.state_dict(), "opt_state": opt.state_dict(), "step": global_step, "best_val": best_val},
            ckpt_dir / "last.pt",
        )
        print(f"[CKPT] end-epoch saved last.pt (epoch={epoch}, step={global_step})", flush=True)

    t_train = time.time() - t_train0
    print(f"[3/4] Training done. elapsed {t_train/60:.1f} min", flush=True)

    # Final validation (full or capped)
    print("[4/4] Final validation ...", flush=True)
    val_loss, val_tokens = run_validation_epoch(
        sae=sae,
        shards=shards,
        mean=mean,
        std=std,
        token_batch_size=args.token_batch_size,
        device=args.device,
        max_val_tokens=max_val_tokens,
    )
    append_jsonl(log_path, {"time": time.time(), "epoch": args.epochs, "step": global_step, "final_val_loss": float(val_loss)})
    print(f"[4/4] Final val_loss={val_loss:.6f}  val_tokens={val_tokens}", flush=True)

    # Always save final last.pt
    torch.save(
        {"sae_state": sae.state_dict(), "opt_state": opt.state_dict(), "step": global_step, "best_val": best_val},
        ckpt_dir / "last.pt",
    )
    print("[DONE] Wrote checkpoints and logs to:", output_dir, flush=True)


if __name__ == "__main__":
    main()
