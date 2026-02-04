#!/usr/bin/env python3
"""
viz_top_attn_imagenet.py

Outputs, for each (feature, token_type), a grid where each cell has 3 rows:
  row 1: original image
  row 2: attention overlay on image
  row 3: attention map only

Notes:
- NO black borders.
- NO SAE heatmap row.
- Attention map is rendered at the SAME size as the image row (cell_size x cell_size),
  using NEAREST so patch structure stays crisp (no blur).
- Attention is "routing" (reg_mean->patches or cls->patches), not an explanation.

Inputs:
- Shards: vecs, image_id, token_type
- SAE ckpt: state_dict, mu, sigma  (only used to pick top features + top activating tokens)
"""

import argparse
import bisect
import io
import math
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
import matplotlib
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, Dinov2WithRegistersModel

# Token type IDs (must match extraction)
TT_REG = 0
TT_HIGH_NORM = 1
TT_NORMAL = 2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--run_dir",
        type=str,
        default="/lambda/nfs/neel/Research/runs/dinov2/imagenet1k_sae/layer6_high_norm_tokens",
        help="Run directory containing shards/ and SAE directories.",
    )
    ap.add_argument(
        "--sae_dir",
        type=str,
        default="sae_1024_all",
        help="SAE subdirectory inside run_dir (contains sae.pt).",
    )
    ap.add_argument(
        "--parquet_dir",
        type=str,
        default="/lambda/nfs/neel/Research/datasets/imagenet1k/data",
        help="Directory containing ImageNet-1k parquet shards.",
    )

    # selection
    ap.add_argument("--top_n", type=int, default=5)
    ap.add_argument("--num_features_to_viz", type=int, default=80)
    ap.add_argument("--batch", type=int, default=5000, help="Batch size for scanning stored vectors.")
    ap.add_argument("--grid_cols", type=int, default=5)
    ap.add_argument("--cell_size", type=int, default=224)

    # model
    ap.add_argument(
        "--model_id",
        type=str,
        default="facebook/dinov2-with-registers-base",
        help="DINOv2-with-registers model id.",
    )

    # attention
    ap.add_argument(
        "--attn_query",
        type=str,
        default="reg_mean",
        choices=["reg_mean", "cls"],
        help="Attention query token: reg_mean (avg over registers) or cls.",
    )
    ap.add_argument(
        "--attn_layer_1_indexed",
        type=int,
        default=6,
        help="Which attention layer (1-indexed) to visualize.",
    )
    ap.add_argument(
        "--overlay_alpha",
        type=float,
        default=0.45,
        help="Blend alpha for overlay (0=just image, 1=just heat).",
    )
    ap.add_argument(
        "--clip_q",
        type=float,
        default=0.99,
        help="Quantile clip for attention map normalization.",
    )

    # quick test
    ap.add_argument("--trial", action="store_true", help="Quick sanity-check subset.")
    ap.add_argument("--limit_shards", type=int, default=0, help="If >0, only scan the first N shards.")
    ap.add_argument(
        "--features",
        type=str,
        default="",
        help="Comma-separated feature indices to visualize (skips Pass 1). Example: 19,37,105",
    )

    return ap.parse_args()


def unpack_image_id(packed: int) -> Tuple[int, int]:
    file_idx = (int(packed) >> 32) & 0xFFFFFFFF
    row_idx = int(packed) & 0xFFFFFFFF
    return file_idx, row_idx


def _bytes_from_cell(cell: Any) -> Optional[bytes]:
    if cell is None:
        return None
    if isinstance(cell, (bytes, bytearray)):
        return bytes(cell)
    if isinstance(cell, dict):
        if "bytes" in cell and isinstance(cell["bytes"], (bytes, bytearray)):
            return bytes(cell["bytes"])
        if "data" in cell and isinstance(cell["data"], (bytes, bytearray)):
            return bytes(cell["data"])
    return None


class ParquetIndexCache:
    def __init__(self, parquet_files: List[str]):
        self.parquet_files = parquet_files
        self._pf: Dict[int, pq.ParquetFile] = {}
        self._rg_prefix: Dict[int, List[int]] = {}
        self._nrows: Dict[int, int] = {}

    def get_pf(self, file_idx: int) -> pq.ParquetFile:
        if file_idx not in self._pf:
            self._pf[file_idx] = pq.ParquetFile(self.parquet_files[file_idx])
        return self._pf[file_idx]

    def get_total_rows(self, file_idx: int) -> int:
        if file_idx in self._nrows:
            return self._nrows[file_idx]
        pf = self.get_pf(file_idx)
        self._nrows[file_idx] = int(pf.metadata.num_rows)
        return self._nrows[file_idx]

    def get_rowgroup_prefix(self, file_idx: int) -> List[int]:
        if file_idx in self._rg_prefix:
            return self._rg_prefix[file_idx]
        pf = self.get_pf(file_idx)
        prefix = [0]
        running = 0
        for rg in range(pf.num_row_groups):
            running += int(pf.metadata.row_group(rg).num_rows)
            prefix.append(running)
        self._rg_prefix[file_idx] = prefix
        return prefix


def pil_from_parquet(cache: ParquetIndexCache, file_idx: int, row_idx: int, fallback_size: int = 224) -> Image.Image:
    if file_idx < 0 or file_idx >= len(cache.parquet_files):
        return Image.new("RGB", (fallback_size, fallback_size), (128, 128, 128))

    total_rows = cache.get_total_rows(file_idx)
    if row_idx < 0 or row_idx >= total_rows:
        return Image.new("RGB", (fallback_size, fallback_size), (128, 128, 128))

    pf = cache.get_pf(file_idx)
    prefix = cache.get_rowgroup_prefix(file_idx)

    rg = bisect.bisect_right(prefix, row_idx) - 1
    rg = max(0, min(rg, pf.num_row_groups - 1))
    in_rg = row_idx - prefix[rg]

    table = pf.read_row_group(rg)

    if "image" in table.column_names:
        img_col = "image"
    elif "bytes" in table.column_names:
        img_col = "bytes"
    else:
        return Image.new("RGB", (fallback_size, fallback_size), (128, 128, 128))

    if in_rg < 0 or in_rg >= table.num_rows:
        return Image.new("RGB", (fallback_size, fallback_size), (128, 128, 128))

    arr = table[img_col].combine_chunks()
    cell = arr[in_rg].as_py()
    b = _bytes_from_cell(cell)
    if b is None:
        return Image.new("RGB", (fallback_size, fallback_size), (128, 128, 128))

    try:
        with Image.open(io.BytesIO(b)) as im:
            return im.convert("RGB").copy()
    except Exception:
        return Image.new("RGB", (fallback_size, fallback_size), (128, 128, 128))


def shard_paths(shards_dir: Path) -> List[Path]:
    paths = sorted(shards_dir.glob("shard_*.pt"))
    if not paths:
        raise FileNotFoundError(f"No shards found in {shards_dir}")
    return paths


def _reshape_patch_grid(vals_1d: torch.Tensor) -> torch.Tensor:
    N = int(vals_1d.numel())
    g = int(math.isqrt(N))
    if g * g == N:
        return vals_1d.view(g, g)
    return vals_1d.view(1, N)


def _grid_to_color_image(grid: torch.Tensor, out_wh: Tuple[int, int], clip_q: float) -> Image.Image:
    """
    Render grid as viridis heatmap, resized to out_wh using NEAREST for crisp patch blocks.
    """
    g = grid.detach().float().cpu().clamp_min(0.0)
    if g.numel() == 0:
        g = torch.zeros((1, 1), dtype=torch.float32)

    flat = g.flatten()
    vmax = torch.quantile(flat, clip_q).item() if flat.numel() > 0 else 0.0
    if vmax <= 1e-12:
        norm = torch.zeros_like(g)
    else:
        norm = (g / vmax).clamp(0.0, 1.0)

    cmap = matplotlib.colormaps["viridis"]
    rgba = cmap(norm.numpy())  # (H,W,4) in [0,1]
    rgb = (rgba[..., :3] * 255).astype(np.uint8)
    im = Image.fromarray(rgb, mode="RGB")
    im = im.resize(out_wh, resample=Image.NEAREST)  # crisp
    return im


def _overlay(image_rgb: Image.Image, heat_rgb: Image.Image, alpha: float) -> Image.Image:
    """
    image_rgb and heat_rgb must be same size. alpha blends heat onto image.
    """
    a = image_rgb.convert("RGB")
    b = heat_rgb.convert("RGB")
    if a.size != b.size:
        b = b.resize(a.size, resample=Image.NEAREST)
    return Image.blend(a, b, alpha=max(0.0, min(1.0, float(alpha))))


def make_grid_cells(triplets: List[Tuple[Image.Image, Image.Image, Image.Image]], cell_size: int) -> List[Image.Image]:
    """
    Each cell is 3 stacked squares (all exactly cell_size x cell_size).
    No borders.
    """
    cells: List[Image.Image] = []

    def _resize_sq(x: Image.Image) -> Image.Image:
        try:
            return x.convert("RGB").resize((cell_size, cell_size), resample=Image.BICUBIC)
        except Exception:
            return Image.new("RGB", (cell_size, cell_size), (128, 128, 128))

    for (im, overlay_im, attn_im) in triplets:
        a = _resize_sq(im)
        b = _resize_sq(overlay_im)
        c = _resize_sq(attn_im)

        cell = Image.new("RGB", (cell_size, cell_size * 3))
        cell.paste(a, (0, 0))
        cell.paste(b, (0, cell_size))
        cell.paste(c, (0, cell_size * 2))
        cells.append(cell)

    return cells


def save_grid(cells: List[Image.Image], out_path: Path, cols: int) -> None:
    if not cells:
        return
    w, h = cells[0].size
    rows = math.ceil(len(cells) / cols)
    grid = Image.new("RGB", (cols * w, rows * h), (255, 255, 255))
    for i, cell in enumerate(cells):
        r = i // cols
        c = i % cols
        grid.paste(cell, (c * w, r * h))
    grid.save(out_path)


def _force_eager_attention(model: Dinov2WithRegistersModel) -> None:
    """
    Needed for output_attentions=True on many transformer backends.
    """
    try:
        if hasattr(model, "set_attn_implementation"):
            model.set_attn_implementation("eager")
            print("[info] Attention backend set to eager.")
        else:
            print("[warn] model.set_attn_implementation not available; attentions may be unavailable.")
    except Exception as e:
        print(f"[warn] Could not set eager attention backend: {e}")


@torch.inference_mode()
def compute_attention_patch_grid(
    model: Dinov2WithRegistersModel,
    processor: AutoImageProcessor,
    pil: Image.Image,
    attn_layer_idx0: int,
    query_mode: str,
) -> torch.Tensor:
    """
    Returns (H,W) attention grid over PATCH tokens (routing weights).
    """
    inputs = processor(images=[pil], return_tensors="pt").to(DEVICE)
    out = model(**inputs, output_attentions=True, return_dict=True)

    if not hasattr(out, "attentions") or out.attentions is None:
        raise RuntimeError("Model returned no attentions. Ensure eager attention backend is enabled.")

    if attn_layer_idx0 < 0 or attn_layer_idx0 >= len(out.attentions):
        raise ValueError(f"attn_layer_idx0 out of range: {attn_layer_idx0} (have {len(out.attentions)} layers)")

    attn = out.attentions[attn_layer_idx0]  # (B, heads, T, T)
    attn = attn[0]                          # (heads, T, T)
    attn_mean = attn.mean(dim=0)            # (T, T)

    R = int(model.config.num_register_tokens)
    patch_start = 1 + R

    if query_mode == "cls":
        w = attn_mean[0, patch_start:]  # (Npatch,)
    else:
        reg_q = attn_mean[1:1 + R, patch_start:]  # (R, Npatch)
        w = reg_q.mean(dim=0)  # (Npatch,)

    return _reshape_patch_grid(w.clamp_min(0.0))


def main():
    args = parse_args()

    run_dir = Path(args.run_dir)
    shards_dir = run_dir / "shards"
    if not shards_dir.exists():
        raise FileNotFoundError(f"Missing shards dir: {shards_dir}")

    sae_dir = run_dir / args.sae_dir
    sae_path = sae_dir / "sae.pt"
    if not sae_path.exists():
        raise FileNotFoundError(f"Missing SAE checkpoint: {sae_path}")

    parquet_dir = Path(args.parquet_dir)
    parquet_files = sorted(glob(str(parquet_dir / "*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {parquet_dir}")

    out_dir = sae_dir / "viz_attention_only_overlay"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- load SAE (only for selecting features and top tokens) ----
    ckpt = torch.load(sae_path, map_location="cpu")
    sd = ckpt["state_dict"]
    mu = ckpt["mu"].to(torch.float32).to(DEVICE)
    sigma = ckpt["sigma"].to(torch.float32).to(DEVICE)

    W = sd["enc.weight"].to(torch.float32).to(DEVICE)  # (F,D)
    b = sd["enc.bias"].to(torch.float32).to(DEVICE)    # (F,)
    num_features = int(W.shape[0])
    print("loaded SAE, num_features =", num_features)

    # ---- load model for attention ----
    processor = AutoImageProcessor.from_pretrained(args.model_id, use_fast=True)
    model = Dinov2WithRegistersModel.from_pretrained(args.model_id).eval().to(DEVICE)
    _force_eager_attention(model)

    attn_layer_idx0 = args.attn_layer_1_indexed - 1
    if attn_layer_idx0 < 0 or attn_layer_idx0 >= len(model.encoder.layer):
        raise ValueError("attn_layer_1_indexed out of range for this model")

    # ---- shard paths ----
    paths = shard_paths(shards_dir)
    if args.limit_shards and args.limit_shards > 0:
        paths = paths[: args.limit_shards]
    if args.trial:
        paths = paths[:2]
        args.num_features_to_viz = min(args.num_features_to_viz, 5)
        args.top_n = min(args.top_n, 5)

    # ---- choose features ----
    if args.features.strip():
        feats: List[int] = []
        for x in args.features.split(","):
            x = x.strip()
            if not x:
                continue
            try:
                feats.append(int(x))
            except ValueError:
                continue
        feats = [f for f in feats if 0 <= f < num_features]
        if not feats:
            raise ValueError("No valid feature indices parsed from --features")
        top_feats = feats
        print("using provided features:", top_feats)
    else:
        feat_sum = torch.zeros(num_features, device=DEVICE)
        feat_cnt = 0
        for p in tqdm(paths, desc="score features", dynamic_ncols=True):
            shard = torch.load(p, map_location="cpu")
            X = shard["vecs"].to(torch.float32)
            n = X.shape[0]
            for s in range(0, n, args.batch):
                xb = X[s:s + args.batch].to(DEVICE)
                xb = (xb - mu) / sigma
                a = F.relu(xb @ W.t() + b)  # (B,F)
                feat_sum += a.sum(dim=0)
                feat_cnt += a.size(0)
        feat_mean = feat_sum / max(1, feat_cnt)
        top_feats = torch.topk(feat_mean, k=min(args.num_features_to_viz, num_features)).indices.tolist()
        print("selected features:", top_feats[:10], "...")

    sel = torch.tensor(top_feats, device=DEVICE, dtype=torch.long)

    # ---- collect TOP_N tokens per feature + token type ----
    top: Dict[int, Dict[int, List[Tuple[float, int]]]] = {
        ft: {TT_REG: [], TT_HIGH_NORM: [], TT_NORMAL: []} for ft in top_feats
    }

    def push(ft: int, ttype: int, act_val: float, packed_id: int):
        arr = top[ft][ttype]
        arr.append((act_val, packed_id))
        arr.sort(key=lambda x: x[0], reverse=True)
        if len(arr) > args.top_n:
            arr.pop()

    for p in tqdm(paths, desc="collect tops", dynamic_ncols=True):
        shard = torch.load(p, map_location="cpu")
        X = shard["vecs"].to(torch.float32)
        imgid = shard["image_id"].to(torch.int64)
        ttype = shard["token_type"].to(torch.int64)

        n = X.shape[0]
        for s in range(0, n, args.batch):
            xb = X[s:s + args.batch].to(DEVICE)
            xb = (xb - mu) / sigma
            a = F.relu(xb @ W[sel].t() + b[sel])  # (B, F_sel)

            img_b = imgid[s:s + args.batch]
            tt_b = ttype[s:s + args.batch]

            for j, ft in enumerate(top_feats):
                acts = a[:, j].detach().cpu()
                for i in range(acts.numel()):
                    tt = int(tt_b[i].item())
                    if tt not in (TT_REG, TT_HIGH_NORM, TT_NORMAL):
                        continue
                    push(ft, tt, float(acts[i].item()), int(img_b[i].item()))

    # ---- image cache ----
    cache = ParquetIndexCache(parquet_files)
    img_cache: Dict[int, Image.Image] = {}

    def get_image(pid: int) -> Image.Image:
        if pid in img_cache:
            return img_cache[pid]
        file_idx, row_idx = unpack_image_id(pid)
        im = pil_from_parquet(cache, file_idx, row_idx, fallback_size=args.cell_size)
        img_cache[pid] = im
        return im

    # ---- attention cache ----
    attn_img_cache: Dict[Tuple[int, str, int], Image.Image] = {}

    labels = [
        (TT_REG, "reg"),
        (TT_HIGH_NORM, "high_norm_patch"),
        (TT_NORMAL, "normal_patch"),
    ]

    for ft in tqdm(top_feats, desc="write grids", dynamic_ncols=True):
        for tt, name in labels:
            packed_ids = [pid for _, pid in top[ft][tt]]
            triplets: List[Tuple[Image.Image, Image.Image, Image.Image]] = []

            for pid in packed_ids:
                im = get_image(pid)
                im_sq = im.convert("RGB").resize((args.cell_size, args.cell_size), resample=Image.BICUBIC)

                ak = (pid, args.attn_query, attn_layer_idx0)
                if ak in attn_img_cache:
                    attn_map = attn_img_cache[ak]
                else:
                    attn_grid = compute_attention_patch_grid(
                        model=model,
                        processor=processor,
                        pil=im,
                        attn_layer_idx0=attn_layer_idx0,
                        query_mode=args.attn_query,
                    )
                    # IMPORTANT: attention map rendered at SAME size as image row, NEAREST for crispness
                    attn_map = _grid_to_color_image(attn_grid, out_wh=(args.cell_size, args.cell_size), clip_q=args.clip_q)
                    attn_img_cache[ak] = attn_map

                overlay_im = _overlay(im_sq, attn_map, alpha=args.overlay_alpha)
                triplets.append((im_sq, overlay_im, attn_map))

            cells = make_grid_cells(triplets, cell_size=args.cell_size)
            out_path = out_dir / f"feat_{ft:04d}_{name}.png"
            save_grid(cells, out_path, cols=args.grid_cols)

    print("saved grids to:", out_dir)


if __name__ == "__main__":
    main()


