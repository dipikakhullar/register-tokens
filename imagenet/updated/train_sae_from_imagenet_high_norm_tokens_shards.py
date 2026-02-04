#!/usr/bin/env python3
"""
train_sae_from_imagenet_high_norm_tokens_shards.py

Paper-aligned training script for an SAE on token vectors extracted from ImageNet-1k
using the "high-norm patch token" (paper "outlier") definition.

Input shards directory (expected):
  /lambda/nfs/neel/Research/runs/dinov2/imagenet1k_sae/layer6_high_norm_tokens/shards

Outputs:
  /lambda/nfs/neel/Research/runs/dinov2/imagenet1k_sae/layer6_high_norm_tokens/sae_<NUM_FEATURES>_<TOKEN_SET>/sae.pt
  /lambda/nfs/neel/Research/runs/dinov2/imagenet1k_sae/layer6_high_norm_tokens/sae_<NUM_FEATURES>_<TOKEN_SET>/train_metrics.jsonl

Scope constraints (paper-aligned):
- "High-norm patch tokens" are the paper-defined outliers (upper tail of patch token L2 norms).
- "Normal patch tokens" are the remaining patch tokens (non-high-norm).
- Register tokens are saved separately (and are where high-norm behavior is expected to move when registers are used).

This script can train on:
- all tokens (register + high-norm patches + normal patches)
- only registers
- only high-norm patches
- only normal patches

Shard keys supported:
- Required: vecs, image_id, token_type, token_pos
- Optional (if present): norms  (per-token L2 norm; logged as stats if available)
"""

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# Token type IDs (must match extraction)
TT_REG = 0
TT_HIGH_NORM = 1
TT_NORMAL = 2


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run_dir",
        type=str,
        default="/lambda/nfs/neel/Research/runs/dinov2/imagenet1k_sae/layer6_high_norm_tokens",
        help="Directory containing shards/ from the extraction step.",
    )
    ap.add_argument(
        "--token_set",
        type=str,
        default="all",
        choices=["all", "reg", "high_norm", "normal"],
        help="Which token subset to train on.",
    )

    ap.add_argument("--num_features", type=int, default=1024)
    ap.add_argument("--batch_size", type=int, default=2048)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--l1_lambda", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])
    return ap.parse_args()


def dtype_from_str(s: str) -> torch.dtype:
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[s]


def load_all_shards(shards_dir: Path) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    shard_paths = sorted(shards_dir.glob("shard_*.pt"))
    if not shard_paths:
        raise FileNotFoundError(f"No shards found in {shards_dir}")

    vecs_all = []
    imgid_all = []
    ttype_all = []
    tpos_all = []
    norms_all = []

    required = {"vecs", "image_id", "token_type", "token_pos"}

    for p in tqdm(shard_paths, desc="load shards", dynamic_ncols=True):
        d = torch.load(p, map_location="cpu")
        missing = required - set(d.keys())
        if missing:
            raise KeyError(f"{p} missing keys: {sorted(missing)}")

        vecs_all.append(d["vecs"].to(torch.float32))  # train in fp32
        imgid_all.append(d["image_id"].to(torch.int64))
        ttype_all.append(d["token_type"].to(torch.int8))
        tpos_all.append(d["token_pos"].to(torch.int16))

        if "norms" in d:
            norms_all.append(d["norms"].to(torch.float32))

    vecs = torch.cat(vecs_all, 0)
    imgid = torch.cat(imgid_all, 0)
    ttype = torch.cat(ttype_all, 0)
    tpos = torch.cat(tpos_all, 0)
    norms = torch.cat(norms_all, 0) if norms_all else None
    return vecs, imgid, ttype, tpos, norms


def subset_mask(token_type: torch.Tensor, token_set: str) -> torch.Tensor:
    if token_set == "all":
        return torch.ones_like(token_type, dtype=torch.bool)
    if token_set == "reg":
        return token_type == TT_REG
    if token_set == "high_norm":
        return token_type == TT_HIGH_NORM
    if token_set == "normal":
        return token_type == TT_NORMAL
    raise ValueError(f"Unknown token_set: {token_set}")


class SAE(nn.Module):
    def __init__(self, d_in: int, n_feat: int):
        super().__init__()
        self.enc = nn.Linear(d_in, n_feat, bias=True)
        self.dec = nn.Linear(n_feat, d_in, bias=True)

        nn.init.normal_(self.enc.weight, std=0.02)
        nn.init.zeros_(self.enc.bias)
        nn.init.normal_(self.dec.weight, std=0.02)
        nn.init.zeros_(self.dec.bias)

    def forward(self, x):
        a = F.relu(self.enc(x))
        x_hat = self.dec(a)
        return x_hat, a


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    shards_dir = run_dir / "shards"
    if not shards_dir.exists():
        raise FileNotFoundError(f"Missing shards dir: {shards_dir}")

    device = args.device
    dtype = dtype_from_str(args.dtype)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    token_set = args.token_set
    out_dir = run_dir / f"sae_{args.num_features}_{token_set}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write run config once
    (out_dir / "config.json").write_text(
        json.dumps(
            {
                "paper_aligned": True,
                "token_set": token_set,
                "num_features": args.num_features,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "lr": args.lr,
                "l1_lambda": args.l1_lambda,
                "weight_decay": args.weight_decay,
                "seed": args.seed,
                "device": device,
                "dtype": args.dtype,
                "run_dir": str(run_dir),
                "shards_dir": str(shards_dir),
                "token_type_ids": {"reg": TT_REG, "high_norm": TT_HIGH_NORM, "normal": TT_NORMAL},
            },
            indent=2,
        )
    )

    X, image_id, token_type, token_pos, norms = load_all_shards(shards_dir)
    m = subset_mask(token_type, token_set)

    X = X[m]
    image_id = image_id[m]
    token_type = token_type[m]
    token_pos = token_pos[m]
    norms = norms[m] if norms is not None else None

    n, d = X.shape
    print("loaded vecs:", X.shape, "D=", d, "token_set=", token_set)

    # Normalize per-dimension for training stability
    mu = X.mean(dim=0, keepdim=True)
    sigma = X.std(dim=0, keepdim=True).clamp_min(1e-6)
    Xn = (X - mu) / sigma

    sae = SAE(d_in=d, n_feat=args.num_features).to(device)
    opt = torch.optim.AdamW(sae.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    idx = torch.arange(n)
    best_loss = float("inf")

    metrics_path = out_dir / "train_metrics.jsonl"
    model_path = out_dir / "sae.pt"

    # Optional: log norm stats (diagnostic only, no extra interpretation)
    norm_stats = None
    if norms is not None and norms.numel() > 0:
        norm_stats = {
            "norm_mean": float(norms.mean().item()),
            "norm_std": float(norms.std().item()),
            "norm_min": float(norms.min().item()),
            "norm_max": float(norms.max().item()),
        }

    for ep in range(args.epochs):
        perm = idx[torch.randperm(n)]

        tot_recon = 0.0
        tot_l1 = 0.0
        tot = 0.0
        count = 0

        sae.train()
        for s in tqdm(range(0, n, args.batch_size), desc=f"train ep {ep+1}/{args.epochs}", dynamic_ncols=True):
            b_ix = perm[s : s + args.batch_size]
            xb = Xn[b_ix].to(device, dtype=dtype)

            opt.zero_grad(set_to_none=True)
            x_hat, a = sae(xb)

            recon = F.mse_loss(x_hat, xb)
            l1 = a.abs().mean()
            loss = recon + args.l1_lambda * l1

            loss.backward()
            opt.step()

            bs = xb.size(0)
            tot_recon += recon.item() * bs
            tot_l1 += l1.item() * bs
            tot += loss.item() * bs
            count += bs

        avg_recon = tot_recon / count
        avg_l1 = tot_l1 / count
        avg_loss = tot / count

        sae.eval()
        with torch.inference_mode():
            m_eval = min(8192, n)
            j = torch.randint(0, n, (m_eval,))
            xb = Xn[j].to(device, dtype=dtype)
            _, a = sae(xb)
            active = (a > 1e-6).float().sum(dim=1).mean().item()

        metrics = {
            "epoch": ep + 1,
            "avg_loss": avg_loss,
            "avg_recon_mse": avg_recon,
            "avg_l1": avg_l1,
            "avg_active_features_per_token": active,
            "n_tokens": n,
            "d_model": d,
            "num_features": args.num_features,
            "l1_lambda": args.l1_lambda,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "token_set": token_set,
            "model_path": str(model_path),
        }
        if norm_stats is not None:
            metrics.update(norm_stats)

        with metrics_path.open("a") as f:
            f.write(json.dumps(metrics) + "\n")

        print(metrics)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                {
                    "state_dict": sae.state_dict(),
                    "mu": mu,
                    "sigma": sigma,
                    "config": metrics,
                },
                model_path,
            )

    print("saved:", model_path)


if __name__ == "__main__":
    main()
