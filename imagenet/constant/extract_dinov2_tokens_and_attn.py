#!/usr/bin/env python3
"""
Minimal extraction for SAE training + attention summaries (DINOv2 base + reg4).

Defaults:
- model_name: vit_base_patch14_reg4_dinov2
- hook_block: 6 (0-indexed)

Parquet schema:
  image.bytes : binary
  image.path  : string (optional)
  label       : int64

Pipeline:
A) Estimate ONE global patch-norm cutoff (top outlier_pct% are "high-norm") via reservoir sampling.
B) Extract tokens at hook_block, label token_bucket, optionally store attention summaries, and shard outputs.

Important implementation notes:
- Pass B buffers are kept on CPU to avoid CUDA OOM during flush().
- You can skip Pass A by providing --cutoff or --reuse_existing_cutoff (loads from output_dir/extraction_meta.pt).
- Progress + timing: tqdm progress bars + images/sec during both passes.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from io import BytesIO
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from PIL import Image

import torch
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from tqdm.auto import tqdm


# ----------------------------
# Utils
# ----------------------------
def get_torch_dtype(dtype_str: str) -> torch.dtype:
    return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[dtype_str]


def infer_prefix_and_register_counts(model) -> Tuple[int, int]:
    """
    timm ViT convention:
      - model.num_prefix_tokens includes CLS + registers (if present)
      - CLS is typically present -> 1 token
      - registers = num_prefix_tokens - 1
    """
    num_prefix = int(getattr(model, "num_prefix_tokens", 1))
    has_cls = bool(getattr(model, "cls_token", None) is not None)
    cls_count = 1 if has_cls else 0
    num_reg = max(0, num_prefix - cls_count)
    return num_prefix, num_reg


def load_image_from_bytes(b) -> Image.Image:
    if isinstance(b, memoryview):
        b = b.tobytes()
    return Image.open(BytesIO(b)).convert("RGB")


def parquet_paths_from_glob(glob_str: str) -> List[Path]:
    g = Path(glob_str)
    if any(ch in glob_str for ch in ["*", "?", "["]):
        return sorted(g.parent.glob(g.name))
    return [g]


# ----------------------------
# Attention capture (simple)
# ----------------------------
class AttnCapture:
    """
    Monkeypatch timm Attention forward to store attention probabilities.
    Accepts attn_mask kwarg for API compatibility.

    Stores:
      attn_probs: [B, H, N, N]

    We later summarize:
      CLS->patch (avg heads): [B, P]
      REG->patch (avg heads): [B, R, P]
    """

    def __init__(self):
        self.attn_probs: Optional[torch.Tensor] = None
        self._attn_module = None
        self._orig_forward = None

    def install(self, attn_module):
        self._attn_module = attn_module
        self._orig_forward = attn_module.forward

        def forward_with_attn(x, attn_mask=None):
            B, N, C = x.shape

            qkv = attn_module.qkv(x)  # [B, N, 3*C]
            qkv = qkv.reshape(B, N, 3, attn_module.num_heads, C // attn_module.num_heads)
            qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, N, Dh]
            q, k, v = qkv[0], qkv[1], qkv[2]  # each [B, H, N, Dh]

            attn = (q @ k.transpose(-2, -1)) * attn_module.scale  # [B, H, N, N]

            if attn_mask is not None:
                if attn_mask.dtype != torch.bool:
                    attn_mask = attn_mask.to(torch.bool)
                while attn_mask.ndim < attn.ndim:
                    attn_mask = attn_mask.unsqueeze(1)
                attn = attn.masked_fill(~attn_mask, float("-inf"))

            attn = attn.softmax(dim=-1)
            self.attn_probs = attn

            attn = attn_module.attn_drop(attn)
            x_out = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x_out = attn_module.proj(x_out)
            x_out = attn_module.proj_drop(x_out)
            return x_out

        attn_module.forward = forward_with_attn

    def uninstall(self):
        if self._attn_module is not None and self._orig_forward is not None:
            self._attn_module.forward = self._orig_forward

    def pop(self) -> Optional[torch.Tensor]:
        a = self.attn_probs
        self.attn_probs = None
        return a


# ----------------------------
# Pass A: estimate global cutoff (reservoir)
# ----------------------------
@torch.no_grad()
def estimate_global_patch_norm_cutoff(
    parquet_paths: List[Path],
    model_name: str,
    hook_block: int,
    device: str,
    dtype: str,
    batch_size: int,
    outlier_pct: float,
    max_norm_samples: int,
    seed: int,
) -> Tuple[float, int, int, int]:
    """
    Returns:
      cutoff (float): global patch L2 norm threshold (top outlier_pct% are high-norm)
      num_prefix, num_reg, N_tokens
    """
    rng = np.random.default_rng(seed)
    torch_dtype = get_torch_dtype(dtype)

    model = timm.create_model(model_name, pretrained=True)
    model.eval().to(device)
    model.to(dtype=torch_dtype)

    cfg = resolve_data_config({}, model=model)
    transform = create_transform(**cfg, is_training=False)

    num_prefix, num_reg = infer_prefix_and_register_counts(model)

    blocks = getattr(model, "blocks", None)
    if blocks is None or hook_block < 0 or hook_block >= len(blocks):
        raise ValueError(
            f"Invalid hook_block={hook_block}. Model has {len(blocks) if blocks is not None else 'no'} blocks."
        )

    captured = {"x": None}

    def hook_fn(_m, _inp, out):
        captured["x"] = out

    handle = blocks[hook_block].register_forward_hook(hook_fn)

    reservoir = np.empty((max_norm_samples,), dtype=np.float32)
    filled = 0
    seen = 0
    total_images = 0
    n_tokens_seen: Optional[int] = None

    t0 = time.time()
    for pq_path in tqdm(parquet_paths, desc="Pass A: shards (cutoff)"):
        df = pd.read_parquet(pq_path, engine="pyarrow", columns=["image"])
        images = df["image"].tolist()

        bbar = tqdm(range(0, len(images), batch_size), desc=f"Pass A: batches ({pq_path.name})", leave=False)
        for start in bbar:
            batch = images[start : start + batch_size]

            pil_list = []
            for im in batch:
                b = im.get("bytes", None)
                if b is None:
                    raise ValueError("Missing image.bytes in parquet row.")
                pil_list.append(load_image_from_bytes(b))

            x = torch.stack([transform(img) for img in pil_list], dim=0).to(
                device=device, dtype=torch_dtype, non_blocking=True
            )

            _ = model(x)

            h = captured["x"]
            if h is None:
                raise RuntimeError("Hook did not capture activations.")
            h = h.to(torch.float32)

            B, N, _D = h.shape
            if n_tokens_seen is None:
                n_tokens_seen = N

            patch_start = num_prefix
            patch = h[:, patch_start:, :]
            norms = torch.linalg.vector_norm(patch, ord=2, dim=-1)
            norms_np = norms.reshape(-1).detach().cpu().numpy().astype(np.float32)

            for v in norms_np:
                seen += 1
                if filled < max_norm_samples:
                    reservoir[filled] = v
                    filled += 1
                else:
                    j = rng.integers(0, seen)
                    if j < max_norm_samples:
                        reservoir[j] = v

            total_images += B
            elapsed = time.time() - t0
            ips = total_images / max(elapsed, 1e-6)
            bbar.set_postfix({"imgs": total_images, "imgs/s": f"{ips:.2f}", "res": filled})

    handle.remove()

    if filled == 0:
        raise RuntimeError("No patch norms sampled. Check input parquet.")

    sample = reservoir[:filled]
    q = 1.0 - (outlier_pct / 100.0)
    cutoff = float(np.quantile(sample, q))

    return cutoff, num_prefix, num_reg, int(n_tokens_seen or 0)


# ----------------------------
# Pass B: extract tokens + labels + attention summaries
# ----------------------------
@torch.no_grad()
def extract_with_labels_and_attention(
    parquet_paths: List[Path],
    output_dir: Path,
    model_name: str,
    hook_block: int,
    device: str,
    dtype: str,
    batch_size: int,
    max_images_per_shard: int,
    cutoff: float,
    store_attention: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    torch_dtype = get_torch_dtype(dtype)

    model = timm.create_model(model_name, pretrained=True)
    model.eval().to(device)
    model.to(dtype=torch_dtype)

    cfg = resolve_data_config({}, model=model)
    transform = create_transform(**cfg, is_training=False)

    num_prefix, num_reg = infer_prefix_and_register_counts(model)

    blocks = getattr(model, "blocks", None)
    if blocks is None or hook_block < 0 or hook_block >= len(blocks):
        raise ValueError(
            f"Invalid hook_block={hook_block}. Model has {len(blocks) if blocks is not None else 'no'} blocks."
        )

    captured = {"x": None}

    def hook_fn(_m, _inp, out):
        captured["x"] = out

    handle_block = blocks[hook_block].register_forward_hook(hook_fn)

    attn_cap = AttnCapture()
    if store_attention:
        attn_cap.install(blocks[hook_block].attn)

    # shard buffers (CPU)
    shard_idx = 0
    images_in_shard = 0

    tok_list: List[torch.Tensor] = []
    img_ix_list: List[torch.Tensor] = []
    pos_list: List[torch.Tensor] = []
    bucket_list: List[torch.Tensor] = []
    label_list: List[torch.Tensor] = []

    src_parquet_list: List[str] = []
    src_row_list: List[int] = []
    img_path_list: List[str] = []

    cls2patch_list: List[torch.Tensor] = []
    reg2patch_list: List[torch.Tensor] = []

    global_image_counter = 0
    total_images = 0
    t0 = time.time()

    def flush():
        nonlocal shard_idx, images_in_shard
        if images_in_shard == 0:
            return

        # All lists contain CPU tensors; concat happens on CPU.
        tokens = torch.cat(tok_list, dim=0).contiguous()
        image_ix = torch.cat(img_ix_list, dim=0).contiguous()
        token_pos = torch.cat(pos_list, dim=0).contiguous()
        token_bucket = torch.cat(bucket_list, dim=0).contiguous()
        labels = torch.cat(label_list, dim=0).contiguous()

        payload = {
            "model_name": model_name,
            "hook_block": int(hook_block),
            "d_model": int(tokens.shape[-1]),
            "num_prefix_tokens": int(num_prefix),
            "num_register_tokens": int(num_reg),
            "patch_norm_cutoff": float(cutoff),
            "tokens": tokens,               # CPU
            "image_ix": image_ix,           # CPU
            "token_pos": token_pos,         # CPU
            "token_bucket": token_bucket,   # CPU: 0=reg, 1=normal_patch, 2=high_norm_patch
            "label": labels,                # CPU
            "src_parquet": src_parquet_list,
            "src_row": src_row_list,
            "image_path": img_path_list,
        }

        if store_attention:
            payload["cls_to_patch_attn"] = torch.cat(cls2patch_list, dim=0)  # CPU float16
            if num_reg > 0:
                payload["reg_to_patch_attn"] = torch.cat(reg2patch_list, dim=0)  # CPU float16

        out_path = output_dir / f"extract_hook{hook_block:02d}_shard{shard_idx:05d}.pt"
        torch.save(payload, out_path)

        shard_idx += 1
        images_in_shard = 0

        tok_list.clear()
        img_ix_list.clear()
        pos_list.clear()
        bucket_list.clear()
        label_list.clear()

        src_parquet_list.clear()
        src_row_list.clear()
        img_path_list.clear()

        cls2patch_list.clear()
        reg2patch_list.clear()

    for pq_path in tqdm(parquet_paths, desc="Pass B: shards (extract)"):
        df = pd.read_parquet(pq_path, engine="pyarrow", columns=["image", "label"])
        images = df["image"].tolist()
        labels_img = df["label"].astype("int64").tolist()

        bbar = tqdm(range(0, len(images), batch_size), desc=f"Pass B: batches ({pq_path.name})", leave=False)
        for start in bbar:
            batch = images[start : start + batch_size]
            batch_labels = labels_img[start : start + batch_size]

            pil_list = []
            batch_paths = []
            for im in batch:
                b = im.get("bytes", None)
                if b is None:
                    raise ValueError("Missing image.bytes in parquet row.")
                pil_list.append(load_image_from_bytes(b))
                batch_paths.append(str(im.get("path", "")) if im.get("path", None) is not None else "")

            x = torch.stack([transform(img) for img in pil_list], dim=0).to(
                device=device, dtype=torch_dtype, non_blocking=True
            )

            _ = model(x)

            h = captured["x"]
            if h is None:
                raise RuntimeError("Hook did not capture activations.")
            # Keep norms stable; do norm computations in fp32.
            h_fp32 = h.to(torch.float32)

            B, N, D = h_fp32.shape
            P = N - num_prefix
            if P <= 0:
                raise RuntimeError(f"Unexpected token counts: N={N}, num_prefix={num_prefix}")

            # global image indices for this batch
            batch_image_ix = torch.arange(
                global_image_counter, global_image_counter + B, device=h_fp32.device, dtype=torch.long
            )
            global_image_counter += B

            # per-image pointers (CPU python lists)
            for bi in range(B):
                src_parquet_list.append(str(pq_path))
                src_row_list.append(int(start + bi))
                img_path_list.append(batch_paths[bi])

            # token positions
            token_pos = torch.arange(N, device=h_fp32.device, dtype=torch.int32).unsqueeze(0).expand(B, N)

            # token_bucket: 0=register, 1=normal_patch, 2=high_norm_patch
            token_bucket = torch.ones((B, N), device=h_fp32.device, dtype=torch.uint8)

            # registers occupy positions 1..num_reg (CLS at 0)
            if num_reg > 0:
                token_bucket[:, 1 : 1 + num_reg] = 0

            # apply global cutoff to patch tokens only
            patch_start = num_prefix
            patch = h_fp32[:, patch_start:, :]  # [B, P, D]
            patch_norm = torch.linalg.vector_norm(patch, ord=2, dim=-1)  # [B, P]
            high_mask = patch_norm > cutoff

            token_bucket[:, patch_start:] = torch.where(
                high_mask,
                torch.tensor(2, device=h_fp32.device, dtype=torch.uint8),
                torch.tensor(1, device=h_fp32.device, dtype=torch.uint8),
            )

            # Flatten per-token (move to CPU immediately to avoid CUDA OOM at flush)
            h_flat_cpu = h_fp32.reshape(B * N, D).detach().cpu()
            pos_flat_cpu = token_pos.reshape(B * N).detach().cpu()
            bucket_flat_cpu = token_bucket.reshape(B * N).detach().cpu()
            img_ix_flat_cpu = batch_image_ix.unsqueeze(1).expand(B, N).reshape(B * N).detach().cpu()

            lab_cpu = torch.tensor(batch_labels, device="cpu", dtype=torch.int64)
            lab_flat_cpu = lab_cpu.unsqueeze(1).expand(B, N).reshape(B * N)

            tok_list.append(h_flat_cpu)
            img_ix_list.append(img_ix_flat_cpu)
            pos_list.append(pos_flat_cpu)
            bucket_list.append(bucket_flat_cpu)
            label_list.append(lab_flat_cpu)

            # attention summaries (optional)
            if store_attention:
                attn = attn_cap.pop()
                if attn is None:
                    raise RuntimeError("Attention capture returned None.")

                attn_avg = attn.mean(dim=1)  # [B, N, N]
                patch_cols = slice(patch_start, N)

                cls_to_patch = attn_avg[:, 0, patch_cols]  # [B, P]
                cls2patch_list.append(cls_to_patch.to(torch.float16).detach().cpu())

                if num_reg > 0:
                    reg_to_patch = attn_avg[:, 1 : 1 + num_reg, patch_cols]  # [B, R, P]
                    reg2patch_list.append(reg_to_patch.to(torch.float16).detach().cpu())

                # free large tensors promptly
                del attn, attn_avg

            images_in_shard += B
            total_images += B

            elapsed = time.time() - t0
            ips = total_images / max(elapsed, 1e-6)
            bbar.set_postfix({"imgs": total_images, "imgs/s": f"{ips:.2f}", "shard_imgs": images_in_shard})

            if images_in_shard >= max_images_per_shard:
                flush()

    flush()

    handle_block.remove()
    if store_attention:
        attn_cap.uninstall()


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--input_glob", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)

    ap.add_argument("--model_name", type=str, default="vit_base_patch14_reg4_dinov2")
    ap.add_argument("--hook_block", type=int, default=6, help="0-indexed block index")

    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, choices=["fp16", "bf16", "fp32"], default="fp16")
    ap.add_argument("--batch_size", type=int, default=32)

    ap.add_argument("--outlier_pct", type=float, default=2.37)
    ap.add_argument("--max_norm_samples", type=int, default=5_000_000)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--max_images_per_shard", type=int, default=4096)
    ap.add_argument("--store_attention", action="store_true")

    # New: skip Pass A
    ap.add_argument(
        "--cutoff",
        type=float,
        default=None,
        help="If provided, skip Pass A and use this global patch-norm cutoff.",
    )
    ap.add_argument(
        "--reuse_existing_cutoff",
        action="store_true",
        help="If set, try to load cutoff from output_dir/extraction_meta.pt and skip Pass A.",
    )

    args = ap.parse_args()

    parquet_paths = parquet_paths_from_glob(args.input_glob)
    if not parquet_paths:
        raise FileNotFoundError(f"No parquet files matched: {args.input_glob}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Decide cutoff source
    cutoff = None
    num_prefix = None
    num_reg = None
    n_tokens = None

    meta_path = output_dir / "extraction_meta.pt"
    if args.cutoff is not None:
        cutoff = float(args.cutoff)
    elif args.reuse_existing_cutoff and meta_path.exists():
        meta = torch.load(meta_path, map_location="cpu")
        cutoff = float(meta["patch_norm_cutoff"])
        num_prefix = int(meta.get("num_prefix_tokens", 0))
        num_reg = int(meta.get("num_register_tokens", 0))
        n_tokens = int(meta.get("num_tokens_total", 0))

    # Pass A if needed
    if cutoff is None:
        print(f"[Pass A] Estimating global cutoff (outlier_pct={args.outlier_pct}%) ...", flush=True)
        tA0 = time.time()
        cutoff, num_prefix, num_reg, n_tokens = estimate_global_patch_norm_cutoff(
            parquet_paths=parquet_paths,
            model_name=args.model_name,
            hook_block=args.hook_block,
            device=args.device,
            dtype=args.dtype,
            batch_size=args.batch_size,
            outlier_pct=args.outlier_pct,
            max_norm_samples=args.max_norm_samples,
            seed=args.seed,
        )
        tA = time.time() - tA0
        print(f"[Pass A] Done. cutoff={cutoff:.6f}  (elapsed={tA/60:.1f} min)", flush=True)
    else:
        # For completeness, if we skipped A but did not load prefix/reg/tokens, infer quickly.
        if num_prefix is None or num_reg is None or n_tokens is None or n_tokens == 0:
            model = timm.create_model(args.model_name, pretrained=True).eval()
            num_prefix, num_reg = infer_prefix_and_register_counts(model)
            blocks = getattr(model, "blocks", None)
            n_tokens = 0
            if blocks is not None and 0 <= args.hook_block < len(blocks):
                # cheap approximate token count by running 1 dummy forward
                cfg = resolve_data_config({}, model=model)
                transform = create_transform(**cfg, is_training=False)
                dummy = Image.new("RGB", (224, 224), color=(0, 0, 0))
                x = torch.stack([transform(dummy)], dim=0)
                model.to(args.device)
                model.to(dtype=get_torch_dtype(args.dtype))
                captured = {"x": None}

                def hook_fn(_m, _inp, out):
                    captured["x"] = out

                hdl = model.blocks[args.hook_block].register_forward_hook(hook_fn)
                with torch.no_grad():
                    _ = model(x.to(args.device, dtype=get_torch_dtype(args.dtype)))
                hdl.remove()
                if captured["x"] is not None:
                    n_tokens = int(captured["x"].shape[1])

        print(f"[Pass A] Skipped. Using cutoff={cutoff:.6f}", flush=True)

    # Save meta (always overwrite with current settings)
    meta = {
        "model_name": args.model_name,
        "hook_block": int(args.hook_block),
        "dtype": args.dtype,
        "batch_size": int(args.batch_size),
        "outlier_pct": float(args.outlier_pct),
        "patch_norm_cutoff": float(cutoff),
        "num_prefix_tokens": int(num_prefix or 0),
        "num_register_tokens": int(num_reg or 0),
        "num_tokens_total": int(n_tokens or 0),
        "store_attention": bool(args.store_attention),
    }
    torch.save(meta, meta_path)

    print(f"[Pass B] Extracting + sharding (store_attention={args.store_attention}) ...", flush=True)
    tB0 = time.time()
    extract_with_labels_and_attention(
        parquet_paths=parquet_paths,
        output_dir=output_dir,
        model_name=args.model_name,
        hook_block=args.hook_block,
        device=args.device,
        dtype=args.dtype,
        batch_size=args.batch_size,
        max_images_per_shard=args.max_images_per_shard,
        cutoff=cutoff,
        store_attention=args.store_attention,
    )
    tB = time.time() - tB0
    print(f"[Pass B] Done. (elapsed={tB/60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()

