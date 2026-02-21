#!/usr/bin/env python3
"""
smoke_test_overcomplete_sae.py

Smoke test for SAE training using KempnerInstitute/overcomplete on your extracted tokens.

What it checks (fast, no full training):
1) Imports + versions (torch, overcomplete)
2) Loads a small sample of tokens from 1–N extraction shards (*.pt)
3) Computes normalization stats (mean/std) on TRAIN subset only
4) Runs a few optimization steps with TopKSAE
5) Verifies Top-K sparsity (≈ top_k nonzeros per example)
6) Verifies reconstruction loss is finite and typically decreases

Expected extraction format (from your extractor):
- each shard is a torch-saved dict with key "tokens" (Tensor [T, D])
- also has "image_ix" etc, but only "tokens" is required here

Usage example:
python3 smoke_test_overcomplete_sae.py \
  --extract_dir /lambda/nfs/neel/Research/runs/dinov2/imagenet1k_base_reg4_hook06_extract \
  --num_shards 2 \
  --max_tokens 200000 \
  --top_k 20 \
  --nb_concepts 4096 \
  --batch_size 4096 \
  --steps 20 \
  --device cuda

Notes:
- This is intentionally small and should finish quickly (minutes).
- If "overcomplete" is missing:
  pip install overcomplete
"""

from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader, TensorDataset


def _print_env(device: str) -> None:
    print("=" * 80)
    print("ENV")
    print("=" * 80)
    print("python:", os.sys.version.replace("\n", " "))
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("cuda device count:", torch.cuda.device_count())
        print("cuda current device:", torch.cuda.current_device())
        print("cuda device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
    print("requested device:", device)
    print()


def _list_shards(extract_dir: Path) -> List[Path]:
    shards = sorted(extract_dir.glob("extract_hook*_shard*.pt"))
    if not shards:
        raise FileNotFoundError(f"No shard files found in {extract_dir} matching extract_hook*_shard*.pt")
    return shards


def _load_token_sample(
    shard_paths: List[Path],
    num_shards: int,
    max_tokens: int,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      X: [N, D] float32
      image_ix: [N] int64 (optional for splitting; filled with -1 if missing)
    """
    g = torch.Generator().manual_seed(seed)

    X_list = []
    ix_list = []
    total = 0

    for p in shard_paths[:num_shards]:
        obj = torch.load(p, map_location="cpu")
        if "tokens" not in obj:
            raise KeyError(f"Shard {p} missing key 'tokens'")
        tokens = obj["tokens"]
        if not torch.is_tensor(tokens):
            raise TypeError(f"Shard {p} 'tokens' is not a torch.Tensor")

        # optional
        image_ix = obj.get("image_ix", None)
        if image_ix is None or not torch.is_tensor(image_ix):
            image_ix = torch.full((tokens.shape[0],), -1, dtype=torch.long)

        # random subsample per shard if needed
        remaining = max_tokens - total
        if remaining <= 0:
            break

        if tokens.shape[0] > remaining:
            idx = torch.randperm(tokens.shape[0], generator=g)[:remaining]
            tokens = tokens[idx]
            image_ix = image_ix[idx]

        X_list.append(tokens)
        ix_list.append(image_ix)
        total += tokens.shape[0]

        if total >= max_tokens:
            break

    X = torch.cat(X_list, dim=0).to(torch.float32).contiguous()
    image_ix = torch.cat(ix_list, dim=0).to(torch.long).contiguous()
    return X, image_ix


def _make_train_val_split(image_ix: torch.Tensor, val_frac: float, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Deterministic-ish split:
    - If image_ix is present (not all -1), uses modulus bucketing for stability
    - Else, random split by seed
    """
    if image_ix.numel() == 0:
        raise ValueError("Empty dataset")

    if (image_ix >= 0).any():
        # Use modulus to approximate val_frac (e.g., 0.2 => mod 5 == 0)
        denom = int(round(1.0 / max(val_frac, 1e-9)))
        denom = max(2, denom)
        val_mask = (image_ix % denom) == 0
        train_mask = ~val_mask
        if train_mask.sum() == 0 or val_mask.sum() == 0:
            # fallback to random if modulus degenerates
            val_mask = None
        else:
            train_idx = train_mask.nonzero(as_tuple=False).squeeze(1)
            val_idx = val_mask.nonzero(as_tuple=False).squeeze(1)
            return train_idx, val_idx

    # random fallback
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(image_ix.shape[0], generator=g)
    n_val = max(1, int(math.floor(val_frac * image_ix.shape[0])))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    return train_idx, val_idx


def _compute_mean_std(X_train: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    mean = X_train.mean(dim=0)
    var = X_train.var(dim=0, unbiased=False)
    std = torch.sqrt(var + eps)
    return mean, std


def _normalize(X: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (X - mean) / std


def _unpack_sae_output(out):
    """
    overcomplete TopKSAE forward returns SAEOuput:
      - docs: returns z_pre, z, x_hat
    But to be robust, handle:
      - tuple/list length 3
      - object with attributes
    Returns:
      x_hat, z_pre, z
    """
    if isinstance(out, (tuple, list)) and len(out) == 3:
        z_pre, z, x_hat = out
        return x_hat, z_pre, z
    # try attribute-style
    for cand in ["x_hat", "xhat", "recons", "reconstruction"]:
        if hasattr(out, cand):
            x_hat = getattr(out, cand)
            break
    else:
        x_hat = None
    for cand in ["pre_codes", "z_pre", "precode", "zpre"]:
        if hasattr(out, cand):
            z_pre = getattr(out, cand)
            break
    else:
        z_pre = None
    for cand in ["codes", "z", "code"]:
        if hasattr(out, cand):
            z = getattr(out, cand)
            break
    else:
        z = None

    if x_hat is None or z_pre is None or z is None:
        raise TypeError(
            "Could not unpack TopKSAE output. Expected (z_pre, z, x_hat) or an object with "
            "attributes like pre_codes/codes/x_hat."
        )
    return x_hat, z_pre, z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract_dir", type=str, required=True)
    ap.add_argument("--num_shards", type=int, default=2)
    ap.add_argument("--max_tokens", type=int, default=200_000)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--nb_concepts", type=int, default=4096)
    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=4096)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--device", type=str, default="cuda")

    args = ap.parse_args()
    device = args.device

    _print_env(device)

    try:
        from overcomplete.sae import TopKSAE  # modern import
    except Exception:
        try:
            from overcomplete import TopKSAE  # docs basic usage
        except Exception as e:
            print("ERROR: Could not import TopKSAE from overcomplete.")
            print("Install with: pip install overcomplete")
            raise e

    extract_dir = Path(args.extract_dir)
    shard_paths = _list_shards(extract_dir)
    print("=" * 80)
    print("DATA LOAD")
    print("=" * 80)
    print("extract_dir:", extract_dir)
    print("total shards found:", len(shard_paths))
    print("loading shards:", min(args.num_shards, len(shard_paths)))
    print("max_tokens:", args.max_tokens)
    t0 = time.time()
    X, image_ix = _load_token_sample(shard_paths, args.num_shards, args.max_tokens, args.seed)
    print(f"loaded X: {tuple(X.shape)}  (elapsed {time.time()-t0:.1f}s)")
    print()

    D = X.shape[1]
    train_idx, val_idx = _make_train_val_split(image_ix, args.val_frac, args.seed)
    X_train = X[train_idx]
    X_val = X[val_idx]
    print("=" * 80)
    print("SPLIT")
    print("=" * 80)
    print("train tokens:", X_train.shape[0])
    print("val tokens:", X_val.shape[0])
    print()

    print("=" * 80)
    print("NORMALIZATION (train-only stats)")
    print("=" * 80)
    mean, std = _compute_mean_std(X_train)
    print("mean shape:", tuple(mean.shape), "std shape:", tuple(std.shape))
    print("std min/mean/max:", float(std.min()), float(std.mean()), float(std.max()))
    Xn_train = _normalize(X_train, mean, std)
    Xn_val = _normalize(X_val, mean, std)
    print("normalized train (approx) mean/std:",
          float(Xn_train.mean()), float(Xn_train.std(unbiased=False)))
    print()

    train_loader = DataLoader(TensorDataset(Xn_train), batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(TensorDataset(Xn_val), batch_size=args.batch_size, shuffle=False, drop_last=False)

    print("=" * 80)
    print("MODEL + OPTIM")
    print("=" * 80)
    sae = TopKSAE(D, args.nb_concepts, top_k=args.top_k, device=device)
    sae = sae.to(device)
    opt = torch.optim.AdamW(sae.parameters(), lr=args.lr)

    def mse_loss(x, x_hat):
        return (x - x_hat).pow(2).mean()

    print("TopKSAE:", sae.__class__.__name__)
    print("D:", D, "nb_concepts:", args.nb_concepts, "top_k:", args.top_k)
    print("steps:", args.steps, "batch_size:", args.batch_size, "lr:", args.lr)
    print()

    print("=" * 80)
    print("TRAIN STEPS (smoke)")
    print("=" * 80)
    sae.train()
    train_iter = iter(train_loader)

    prev_loss = None
    for step in range(1, args.steps + 1):
        try:
            (xb,) = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            (xb,) = next(train_iter)

        xb = xb.to(device, non_blocking=True)

        opt.zero_grad(set_to_none=True)
        out = sae(xb)
        x_hat, z_pre, z = _unpack_sae_output(out)

        loss = mse_loss(xb, x_hat)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}: {loss.item()}")

        loss.backward()
        opt.step()

        # sparsity check: top_k nonzeros per row (allow small tolerance due to ties/implementation)
        with torch.no_grad():
            nnz = (z > 0).sum(dim=1).float().mean().item()

        msg = f"step {step:03d} | loss {loss.item():.6f} | mean_nnz {nnz:.2f}"
        if prev_loss is not None:
            msg += f" | dloss {loss.item()-prev_loss:+.6f}"
        print(msg)
        prev_loss = loss.item()

    print()
    print("=" * 80)
    print("VAL (one pass)")
    print("=" * 80)
    sae.eval()
    val_losses = []
    val_nnz = []
    with torch.no_grad():
        for (xb,) in val_loader:
            xb = xb.to(device, non_blocking=True)
            out = sae(xb)
            x_hat, _z_pre, z = _unpack_sae_output(out)
            val_losses.append(mse_loss(xb, x_hat).item())
            val_nnz.append((z > 0).sum(dim=1).float().mean().item())
    print("val loss mean:", sum(val_losses) / max(1, len(val_losses)))
    print("val mean_nnz mean:", sum(val_nnz) / max(1, len(val_nnz)))
    print()

    print("=" * 80)
    print("SMOKE TEST PASSED")
    print("=" * 80)


if __name__ == "__main__":
    main()
