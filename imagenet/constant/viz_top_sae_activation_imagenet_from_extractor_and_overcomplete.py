#!/usr/bin/env python3
"""
viz_top_sae_activation_imagenet_from_extractor_and_overcomplete.py

Visualize SAE FEATURE ACTIVATION overlays (not attention) on top-ranked ImageNet images
using extractor shards + overcomplete TopKSAE checkpoints.

Grid layout per output image:
  row 1: original image
  row 2: SAE activation overlay
  row 3: SAE activation map only

Works with extractor shards produced by your extractor (fields like):
  - tokens        [T, D]
  - image_ix      [T]
  - token_pos     [T]
  - token_bucket  [T]   (0=reg, 1=normal_patch, 2=high_norm_patch)
  - src_parquet   list[str] per image
  - src_row       list[int] per image

And SAE training outputs from your train_overcomplete_sae.py:
  <sae_dir>/
    norm_stats.pt
    checkpoints/best.pt or last.pt

This version:
- keeps original functionality (exact rankings) while speeding up the hot path
- reduces activations by (token_type, image_ix) per batch before Python dict updates
- explicitly filters ranking tokens to valid token_bucket values (reg/normal/high_norm)
- avoids a second full shard scan to build image->shard ranges (ranges collected during ranking)
- uses binary search for shard lookup during visualization
- preserves spatial patch layout for subset heatmaps (normal/high_norm) by zero-filling others
- writes full top-N grids for each ranking type
- writes one compact "one_of_each_type" grid per feature (top-1 reg / high_norm / normal)
"""

from __future__ import annotations

import argparse
import bisect
import io
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

# extractor token_bucket ids (from your extractor)
TT_REG = 0
TT_NORMAL = 1
TT_HIGH_NORM = 2
VALID_TT_VALUES = (TT_REG, TT_NORMAL, TT_HIGH_NORM)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------
# Args
# ----------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()

    # extractor / data
    ap.add_argument(
        "--extract_dir",
        type=str,
        required=True,
        help="Directory containing extractor shards: extract_hookXX_shardYYYYY.pt",
    )
    ap.add_argument(
        "--parquet_dir",
        type=str,
        default="/lambda/nfs/neel/Research/datasets/imagenet1k/data",
        help="Directory containing ImageNet parquet files (fallback if src_parquet paths are relative/missing).",
    )

    # SAE
    ap.add_argument(
        "--sae_dir",
        type=str,
        required=True,
        help="Absolute path to SAE run dir (contains norm_stats.pt and checkpoints/).",
    )
    ap.add_argument(
        "--sae_ckpt",
        type=str,
        default="best",
        choices=["best", "last"],
        help="Which checkpoint from checkpoints/ to load.",
    )

    # ranking / selection
    ap.add_argument("--top_n", type=int, default=10)
    ap.add_argument("--num_features_to_viz", type=int, default=50)
    ap.add_argument("--batch", type=int, default=5000, help="Batch size for scanning tokens.")
    ap.add_argument(
        "--agg_mode",
        type=str,
        default="max",
        choices=["max", "mean"],
        help="Image-level aggregation of token activations within each token type.",
    )
    ap.add_argument(
        "--features",
        type=str,
        default="",
        help="Comma-separated feature indices to visualize (skips feature selection pass), e.g. 0,1,7",
    )

    # visualization
    ap.add_argument("--grid_cols", type=int, default=5)
    ap.add_argument("--cell_size", type=int, default=224)
    ap.add_argument("--overlay_alpha", type=float, default=0.45)
    ap.add_argument("--clip_q", type=float, default=0.99)

    # debugging / smoke
    ap.add_argument("--limit_shards", type=int, default=0, help="If >0, only scan first N extractor shards.")
    ap.add_argument("--trial", action="store_true", help="Quick subset sanity run.")
    ap.add_argument("--print_shard_timing", action="store_true", help="Print per-shard ranking timing.")

    return ap.parse_args()


# ----------------------------
# Image loading from parquet
# ----------------------------
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
    def __init__(self):
        self._pf: Dict[str, pq.ParquetFile] = {}
        self._rg_prefix: Dict[str, List[int]] = {}
        self._nrows: Dict[str, int] = {}

    def get_pf(self, path: str) -> pq.ParquetFile:
        if path not in self._pf:
            self._pf[path] = pq.ParquetFile(path)
        return self._pf[path]

    def get_total_rows(self, path: str) -> int:
        if path in self._nrows:
            return self._nrows[path]
        pf = self.get_pf(path)
        self._nrows[path] = int(pf.metadata.num_rows)
        return self._nrows[path]

    def get_rowgroup_prefix(self, path: str) -> List[int]:
        if path in self._rg_prefix:
            return self._rg_prefix[path]
        pf = self.get_pf(path)
        prefix = [0]
        running = 0
        for rg in range(pf.num_row_groups):
            running += int(pf.metadata.row_group(rg).num_rows)
            prefix.append(running)
        self._rg_prefix[path] = prefix
        return prefix


def pil_from_parquet_path_row(
    cache: ParquetIndexCache,
    parquet_path: str,
    row_idx: int,
    fallback_size: int = 224,
) -> Image.Image:
    try:
        total_rows = cache.get_total_rows(parquet_path)
        if row_idx < 0 or row_idx >= total_rows:
            return Image.new("RGB", (fallback_size, fallback_size), (128, 128, 128))

        pf = cache.get_pf(parquet_path)
        prefix = cache.get_rowgroup_prefix(parquet_path)

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

        with Image.open(io.BytesIO(b)) as im:
            return im.convert("RGB").copy()
    except Exception:
        return Image.new("RGB", (fallback_size, fallback_size), (128, 128, 128))


# ----------------------------
# Rendering helpers
# ----------------------------
def _reshape_patch_grid(vals_1d: torch.Tensor) -> torch.Tensor:
    n = int(vals_1d.numel())
    g = int(math.isqrt(n))
    if g * g == n:
        return vals_1d.view(g, g)
    return vals_1d.view(1, n)


def _grid_to_color_image(grid: torch.Tensor, out_wh: Tuple[int, int], clip_q: float) -> Image.Image:
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
    rgba = cmap(norm.numpy())
    rgb = (rgba[..., :3] * 255).astype(np.uint8)
    im = Image.fromarray(rgb, mode="RGB")
    im = im.resize(out_wh, resample=Image.NEAREST)  # keep patch blocks crisp
    return im


def _overlay(image_rgb: Image.Image, heat_rgb: Image.Image, alpha: float) -> Image.Image:
    a = image_rgb.convert("RGB")
    b = heat_rgb.convert("RGB")
    if a.size != b.size:
        b = b.resize(a.size, resample=Image.NEAREST)
    alpha = max(0.0, min(1.0, float(alpha)))
    return Image.blend(a, b, alpha=alpha)


def make_grid_cells(triplets: List[Tuple[Image.Image, Image.Image, Image.Image]], cell_size: int) -> List[Image.Image]:
    cells: List[Image.Image] = []

    def _resize_sq(x: Image.Image) -> Image.Image:
        try:
            return x.convert("RGB").resize((cell_size, cell_size), resample=Image.BICUBIC)
        except Exception:
            return Image.new("RGB", (cell_size, cell_size), (128, 128, 128))

    for orig, over, heat in triplets:
        a = _resize_sq(orig)
        b = _resize_sq(over)
        c = _resize_sq(heat)

        cell = Image.new("RGB", (cell_size, cell_size * 3))
        cell.paste(a, (0, 0))
        cell.paste(b, (0, cell_size))
        cell.paste(c, (0, 2 * cell_size))
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


# ----------------------------
# Extractor shard helpers
# ----------------------------
def list_extractor_shards(extract_dir: Path) -> List[Path]:
    paths = sorted(extract_dir.glob("extract_hook*_shard*.pt"))
    if not paths:
        raise FileNotFoundError(f"No extractor shards found in {extract_dir}")
    return paths


def _sort_ranges_for_lookup(ranges: List[Tuple[int, int, Path]]) -> List[Tuple[int, int, Path]]:
    return sorted(ranges, key=lambda x: (x[0], x[1], str(x[2])))


def _build_range_starts(ranges: List[Tuple[int, int, Path]]) -> List[int]:
    return [r[0] for r in ranges]


def find_extractor_shard_for_image_ix(
    image_ix: int,
    ranges: List[Tuple[int, int, Path]],
    starts: List[int],
) -> Optional[Path]:
    if not ranges:
        return None

    idx = bisect.bisect_right(starts, image_ix) - 1
    if idx >= 0:
        mn, mx, p = ranges[idx]
        if mn <= image_ix <= mx:
            return p

    for j in (idx - 1, idx + 1):
        if 0 <= j < len(ranges):
            mn, mx, p = ranges[j]
            if mn <= image_ix <= mx:
                return p

    for mn, mx, p in ranges:
        if mn <= image_ix <= mx:
            return p
    return None


def build_image_meta_index_from_shard(obj: Dict[str, Any]) -> Dict[int, Tuple[str, int]]:
    """
    Map extractor image_ix -> (src_parquet_path, src_row).
    Uses token_pos==0 rows (CLS) to align with per-image src_parquet/src_row arrays.
    """
    image_ix = obj["image_ix"].to(torch.int64)
    token_pos = obj["token_pos"].to(torch.int64)

    src_parquet = obj.get("src_parquet", None)
    src_row = obj.get("src_row", None)
    if src_parquet is None or src_row is None:
        return {}

    cls_mask = (token_pos == 0)
    cls_img_ids = image_ix[cls_mask].tolist()

    meta: Dict[int, Tuple[str, int]] = {}
    n = min(len(cls_img_ids), len(src_parquet), len(src_row))
    for i in range(n):
        iid = int(cls_img_ids[i])
        if iid not in meta:
            meta[iid] = (str(src_parquet[i]), int(src_row[i]))
    return meta


def find_image_meta_for_image_ix(
    image_ix: int,
    parquet_dir: Path,
    shard_meta_map: Dict[int, Tuple[str, int]],
) -> Optional[Tuple[str, int]]:
    if image_ix not in shard_meta_map:
        return None

    p, r = shard_meta_map[image_ix]
    pp = Path(p)
    if pp.exists():
        return (str(pp), r)

    alt = parquet_dir / pp.name
    if alt.exists():
        return (str(alt), r)

    return None


# ----------------------------
# SAE helpers
# ----------------------------
def _find_key(sd: Dict[str, torch.Tensor], candidates: List[str]) -> Optional[str]:
    for k in candidates:
        if k in sd:
            return k
    return None


def _infer_enc_keys_by_shape(sd: Dict[str, torch.Tensor]) -> Tuple[str, str]:
    """
    Infer encoder weight/bias from state_dict shapes.

    Expected:
      - encoder weight: 2D [F, D]
      - encoder bias:   1D [F]
    """
    tensor_items = [(k, v) for k, v in sd.items() if torch.is_tensor(v)]

    w_candidates = [(k, v) for k, v in tensor_items if v.ndim == 2]
    w_candidates_sorted = sorted(
        w_candidates,
        key=lambda kv: (
            0 if ("enc" in kv[0].lower() or "encoder" in kv[0].lower()) else 1,
            kv[0],
        ),
    )

    for wk, wv in w_candidates_sorted:
        fdim = int(wv.shape[0])

        b_matches = []
        for bk, bv in tensor_items:
            if bv.ndim == 1 and int(bv.shape[0]) == fdim:
                b_matches.append((bk, bv))

        if not b_matches:
            continue

        b_matches = sorted(
            b_matches,
            key=lambda kv: (
                0 if ("enc" in kv[0].lower() or "encoder" in kv[0].lower() or "bias" in kv[0].lower()) else 1,
                kv[0],
            ),
        )
        bk, _ = b_matches[0]
        return wk, bk

    all_keys = list(sd.keys())
    preview = "\n".join(all_keys[:200])
    raise KeyError(
        "Could not infer encoder weight/bias from checkpoint state_dict.\n"
        f"Available keys (first {min(len(all_keys), 200)}):\n{preview}"
    )


def load_sae_and_norm(sae_dir: Path, ckpt_name: str):
    norm_path = sae_dir / "norm_stats.pt"
    ckpt_path = sae_dir / "checkpoints" / f"{ckpt_name}.pt"

    if not norm_path.exists():
        raise FileNotFoundError(f"Missing norm_stats.pt: {norm_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    norm = torch.load(norm_path, map_location="cpu")
    mean = norm["mean"].to(torch.float32).to(DEVICE)
    std = norm["std"].to(torch.float32).to(DEVICE)
    std = torch.where(std.abs() < 1e-12, torch.ones_like(std), std)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    sae_state = ckpt.get("sae_state", ckpt)

    w_key = _find_key(
        sae_state,
        [
            "enc.weight",
            "encoder.weight",
            "sae.encoder.weight",
            "model.encoder.weight",
            "W_enc",
            "weight_enc",
        ],
    )
    b_key = _find_key(
        sae_state,
        [
            "enc.bias",
            "encoder.bias",
            "sae.encoder.bias",
            "model.encoder.bias",
            "b_enc",
            "bias_enc",
        ],
    )

    if w_key is None or b_key is None:
        print("[warn] Could not find canonical encoder keys. Inferring from state_dict shapes...")
        w_key2, b_key2 = _infer_enc_keys_by_shape(sae_state)
        if w_key is None:
            w_key = w_key2
        if b_key is None:
            b_key = b_key2

    print(f"[info] using encoder weight key: {w_key}")
    print(f"[info] using encoder bias key:   {b_key}")

    W = sae_state[w_key].to(torch.float32).to(DEVICE)
    b = sae_state[b_key].to(torch.float32).to(DEVICE)

    if W.ndim != 2 or b.ndim != 1:
        raise ValueError(f"Unexpected shapes: W {tuple(W.shape)}, b {tuple(b.shape)}")
    if W.shape[0] != b.shape[0]:
        raise ValueError(f"Encoder shape mismatch: W {tuple(W.shape)}, b {tuple(b.shape)}")

    return W, b, mean, std, ckpt_path


@torch.inference_mode()
def compute_sae_activation_patch_grid_from_shard(
    *,
    shard_obj: Dict[str, Any],
    image_ix_target: int,
    feature_idx: int,
    W: torch.Tensor,      # (F,D), DEVICE
    b: torch.Tensor,      # (F,), DEVICE
    mu: torch.Tensor,     # (D,), DEVICE
    sigma: torch.Tensor,  # (D,), DEVICE
    num_prefix_tokens_fallback: int = 5,
    patch_subset: str = "all",  # "all" | "normal" | "high_norm"
) -> torch.Tensor:
    """
    Direct SAE activation map for one image and feature from extractor tokens.

    Preserves spatial patch layout even when visualizing only a subset of patch types
    (normal / high_norm) by zero-filling excluded patch positions.
    """
    tokens = shard_obj["tokens"].to(torch.float32)       # CPU [T,D]
    image_ix = shard_obj["image_ix"].to(torch.int64)     # CPU [T]
    token_pos = shard_obj["token_pos"].to(torch.int64)   # CPU [T]
    token_bucket = shard_obj.get("token_bucket", None)
    if token_bucket is not None:
        token_bucket = token_bucket.to(torch.int64)

    mask_img = (image_ix == int(image_ix_target))
    if mask_img.sum().item() == 0:
        return torch.zeros((1, 1), dtype=torch.float32)

    x_img = tokens[mask_img]
    pos_img = token_pos[mask_img]
    tb_img = token_bucket[mask_img] if token_bucket is not None else None

    # sort by token position so patch order is spatially correct
    order = torch.argsort(pos_img)
    x_img = x_img[order]
    pos_img = pos_img[order]
    if tb_img is not None:
        tb_img = tb_img[order]

    num_prefix = int(shard_obj.get("num_prefix_tokens", num_prefix_tokens_fallback))
    patch_mask_all = pos_img >= num_prefix
    if patch_mask_all.sum().item() == 0:
        return torch.zeros((1, 1), dtype=torch.float32)

    x_patch_all = x_img[patch_mask_all]
    x_patch_all = x_patch_all.to(DEVICE)
    x_patch_all = (x_patch_all - mu) / sigma

    wf = W[feature_idx]
    bf = b[feature_idx]
    acts_all = F.relu(x_patch_all @ wf + bf).detach().cpu()  # [P_all]

    # preserve full patch grid positions
    if tb_img is not None and patch_subset in ("normal", "high_norm"):
        tb_patch_all = tb_img[patch_mask_all]
        if patch_subset == "normal":
            keep = (tb_patch_all == TT_NORMAL)
        else:
            keep = (tb_patch_all == TT_HIGH_NORM)

        if keep.sum().item() == 0:
            acts_all = torch.zeros_like(acts_all)
        else:
            acts_all = acts_all * keep.to(acts_all.dtype)

    elif patch_subset != "all":
        raise ValueError(f"Invalid patch_subset: {patch_subset}")

    return _reshape_patch_grid(acts_all)


def patch_subset_for_rank_type(tt: int) -> str:
    # Register-ranked images do not form a patch grid, so visualize all patches for that image.
    if tt == TT_REG:
        return "all"
    if tt == TT_NORMAL:
        return "normal"
    if tt == TT_HIGH_NORM:
        return "high_norm"
    return "all"


# ----------------------------
# Fast batch reduction helpers
# ----------------------------
@torch.inference_mode()
def _reduce_by_image_per_token_type_max(
    img_b: torch.Tensor,    # CPU [B]
    tt_b: torch.Tensor,     # CPU [B]
    a_cpu: torch.Tensor,    # CPU [B, F_sel]
) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Returns per token type:
      tt -> (uniq_img_ids [U], reduced_max [U, F_sel])
    """
    out: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

    for tt in VALID_TT_VALUES:
        m = (tt_b == tt)
        if not bool(m.any()):
            continue

        img_tt = img_b[m]
        act_tt = a_cpu[m]
        uniq, inv = torch.unique(img_tt, sorted=False, return_inverse=True)
        U = int(uniq.numel())
        F_sel = int(act_tt.shape[1])
        if U == 0:
            continue

        try:
            red = torch.full((U, F_sel), float("-inf"), dtype=act_tt.dtype)
            red = red.scatter_reduce(
                0,
                inv.view(-1, 1).expand(-1, F_sel),
                act_tt,
                reduce="amax",
                include_self=True,
            )
        except Exception:
            rows = [act_tt[inv == u].max(dim=0).values for u in range(U)]
            red = torch.stack(rows, dim=0)

        out[tt] = (uniq, red)

    return out


@torch.inference_mode()
def _reduce_by_image_per_token_type_sum_count(
    img_b: torch.Tensor,    # CPU [B]
    tt_b: torch.Tensor,     # CPU [B]
    a_cpu: torch.Tensor,    # CPU [B, F_sel]
) -> Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Returns per token type:
      tt -> (uniq_img_ids [U], reduced_sum [U, F_sel], counts [U])
    """
    out: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    for tt in VALID_TT_VALUES:
        m = (tt_b == tt)
        if not bool(m.any()):
            continue

        img_tt = img_b[m]
        act_tt = a_cpu[m]
        uniq, inv = torch.unique(img_tt, sorted=False, return_inverse=True)
        U = int(uniq.numel())
        F_sel = int(act_tt.shape[1])
        if U == 0:
            continue

        red_sum = torch.zeros((U, F_sel), dtype=act_tt.dtype)
        red_sum.index_add_(0, inv, act_tt)
        counts = torch.bincount(inv, minlength=U).to(torch.int64)
        out[tt] = (uniq, red_sum, counts)

    return out


# ----------------------------
# Main
# ----------------------------
def main():
    args = parse_args()

    extract_dir = Path(args.extract_dir)
    if not extract_dir.exists():
        raise FileNotFoundError(f"Missing extract_dir: {extract_dir}")

    sae_dir = Path(args.sae_dir)
    if not sae_dir.exists():
        raise FileNotFoundError(f"Missing sae_dir: {sae_dir}")

    parquet_dir = Path(args.parquet_dir)
    if not parquet_dir.exists():
        print(f"[warn] parquet_dir does not exist: {parquet_dir} (will rely on src_parquet absolute paths)")

    out_dir = sae_dir / "viz_sae_activation_overlay"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load SAE + normalization
    W, b, mu, sigma, loaded_ckpt_path = load_sae_and_norm(sae_dir, args.sae_ckpt)
    num_features = int(W.shape[0])
    d_model = int(W.shape[1])
    print(f"[info] loaded SAE checkpoint: {loaded_ckpt_path}")
    print(f"[info] num_features={num_features}, d_model={d_model}, device={DEVICE}")

    # Extractor shards
    paths = list_extractor_shards(extract_dir)
    if args.limit_shards and args.limit_shards > 0:
        paths = paths[: args.limit_shards]
    if args.trial:
        paths = paths[:2]
        args.top_n = min(args.top_n, 5)
        args.num_features_to_viz = min(args.num_features_to_viz, 5)
    print(f"[info] using {len(paths)} extractor shards")

    # Feature selection (if not provided)
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
        top_feats = [f for f in feats if 0 <= f < num_features]
        if not top_feats:
            raise ValueError("No valid feature indices in --features")
        print(f"[info] using provided features: {top_feats}")
    else:
        print("[info] selecting high-mean features over scanned tokens ...")
        feat_sum = torch.zeros(num_features, device=DEVICE)
        feat_cnt = 0

        # Reuse transpose for faster matmul: [B,D] @ [D,F]
        WT_all = W.t().contiguous()

        for p in tqdm(paths, desc="feature selection pass", dynamic_ncols=True):
            obj = torch.load(p, map_location="cpu")
            X = obj["tokens"].to(torch.float32)

            n = int(X.shape[0])
            for s in range(0, n, args.batch):
                xb = X[s:s + args.batch].to(DEVICE, non_blocking=True)
                xb = (xb - mu) / sigma
                a = F.relu(xb @ WT_all + b)  # (B,F)
                feat_sum += a.sum(dim=0)
                feat_cnt += int(a.shape[0])

        feat_mean = feat_sum / max(1, feat_cnt)
        k = min(args.num_features_to_viz, num_features)
        top_feats = torch.topk(feat_mean, k=k).indices.tolist()
        print(f"[info] selected features (first 10): {top_feats[:10]}")

        # Save selected features to make resume runs fast and reproducible
        try:
            (out_dir / "selected_features.txt").write_text(",".join(map(str, top_feats)))
            torch.save({"selected_features": top_feats}, out_dir / "selected_features.pt")
        except Exception as e:
            print(f"[warn] failed to save selected features list: {e}")

    fsel = len(top_feats)
    if fsel == 0:
        raise ValueError("No features selected")

    sel = torch.tensor(top_feats, device=DEVICE, dtype=torch.long)
    W_sel = W[sel].contiguous()           # [F_sel, D]
    WT_sel = W_sel.t().contiguous()       # [D, F_sel]
    b_sel = b[sel].contiguous()           # [F_sel]
    print(f"[info] ranking/visualizing {fsel} features")

    # Collect image-level scores by token type (exact behavior)
    if args.agg_mode == "max":
        image_scores: Dict[int, Dict[int, Dict[int, float]]] = {
            ft: {TT_REG: {}, TT_NORMAL: {}, TT_HIGH_NORM: {}} for ft in top_feats
        }
        score_dicts_by_tt: Dict[int, List[Dict[int, float]]] = {
            TT_REG:       [image_scores[ft][TT_REG] for ft in top_feats],
            TT_NORMAL:    [image_scores[ft][TT_NORMAL] for ft in top_feats],
            TT_HIGH_NORM: [image_scores[ft][TT_HIGH_NORM] for ft in top_feats],
        }
    else:
        image_sum: Dict[int, Dict[int, Dict[int, float]]] = {
            ft: {TT_REG: {}, TT_NORMAL: {}, TT_HIGH_NORM: {}} for ft in top_feats
        }
        image_cnt: Dict[int, Dict[int, Dict[int, int]]] = {
            ft: {TT_REG: {}, TT_NORMAL: {}, TT_HIGH_NORM: {}} for ft in top_feats
        }

        sum_dicts_by_tt: Dict[int, List[Dict[int, float]]] = {
            TT_REG:       [image_sum[ft][TT_REG] for ft in top_feats],
            TT_NORMAL:    [image_sum[ft][TT_NORMAL] for ft in top_feats],
            TT_HIGH_NORM: [image_sum[ft][TT_HIGH_NORM] for ft in top_feats],
        }
        cnt_dicts_by_tt: Dict[int, List[Dict[int, int]]] = {
            TT_REG:       [image_cnt[ft][TT_REG] for ft in top_feats],
            TT_NORMAL:    [image_cnt[ft][TT_NORMAL] for ft in top_feats],
            TT_HIGH_NORM: [image_cnt[ft][TT_HIGH_NORM] for ft in top_feats],
        }

    print(f"[info] collecting image-level scores ({args.agg_mode}) ...")
    extractor_ranges_unsorted: List[Tuple[int, int, Path]] = []

    for shard_idx, p in enumerate(tqdm(paths, desc="ranking pass", dynamic_ncols=True), start=1):
        t0 = time.time()

        obj = torch.load(p, map_location="cpu")
        if "token_bucket" not in obj:
            raise KeyError(f"{p} missing token_bucket; rerun extractor or adapt script.")

        X_all = obj["tokens"].to(torch.float32)             # CPU
        imgid_all = obj["image_ix"].to(torch.int64)         # CPU
        ttype_all = obj["token_bucket"].to(torch.int64)     # CPU

        if imgid_all.numel() > 0:
            extractor_ranges_unsorted.append(
                (int(imgid_all.min().item()), int(imgid_all.max().item()), p)
            )

        # Exclude CLS/unknown buckets from ranking (matches intended semantics)
        valid_mask = (ttype_all == TT_REG) | (ttype_all == TT_NORMAL) | (ttype_all == TT_HIGH_NORM)
        if not bool(valid_mask.any()):
            if args.print_shard_timing:
                print(f"[info] ranking shard {shard_idx}/{len(paths)} {p.name}: no valid token_bucket rows")
            continue

        X = X_all[valid_mask]
        imgid = imgid_all[valid_mask]
        ttype = ttype_all[valid_mask]

        n = int(X.shape[0])
        for s in range(0, n, args.batch):
            xb = X[s:s + args.batch].to(DEVICE, non_blocking=True)
            xb = (xb - mu) / sigma
            a = F.relu(xb @ WT_sel + b_sel)          # GPU [B, F_sel]
            a_cpu = a.detach().cpu()                 # CPU [B, F_sel]

            img_b = imgid[s:s + args.batch]
            tt_b = ttype[s:s + args.batch]

            if args.agg_mode == "max":
                reduced = _reduce_by_image_per_token_type_max(img_b, tt_b, a_cpu)
                for tt, (uniq_ids, red_max) in reduced.items():
                    dicts = score_dicts_by_tt[tt]
                    U = int(uniq_ids.numel())
                    for u in range(U):
                        pid = int(uniq_ids[u].item())
                        vals = red_max[u].tolist()
                        for j, val in enumerate(vals):
                            d = dicts[j]
                            fv = float(val)
                            prev = d.get(pid)
                            if (prev is None) or (fv > prev):
                                d[pid] = fv
            else:
                reduced = _reduce_by_image_per_token_type_sum_count(img_b, tt_b, a_cpu)
                for tt, (uniq_ids, red_sum, counts) in reduced.items():
                    sum_dicts = sum_dicts_by_tt[tt]
                    cnt_dicts = cnt_dicts_by_tt[tt]
                    U = int(uniq_ids.numel())
                    for u in range(U):
                        pid = int(uniq_ids[u].item())
                        cnt_u = int(counts[u].item())
                        if cnt_u <= 0:
                            continue
                        vals = red_sum[u].tolist()
                        for j, sval in enumerate(vals):
                            dsum = sum_dicts[j]
                            dcnt = cnt_dicts[j]
                            dsum[pid] = dsum.get(pid, 0.0) + float(sval)
                            dcnt[pid] = dcnt.get(pid, 0) + cnt_u

        if args.print_shard_timing:
            print(f"[info] ranking shard {shard_idx}/{len(paths)} {p.name} done in {time.time() - t0:.1f}s")

    extractor_ranges = _sort_ranges_for_lookup(extractor_ranges_unsorted)
    extractor_range_starts = _build_range_starts(extractor_ranges)

    # Final top-N
    top: Dict[int, Dict[int, List[Tuple[float, int]]]] = {
        ft: {TT_REG: [], TT_NORMAL: [], TT_HIGH_NORM: []} for ft in top_feats
    }

    if args.agg_mode == "max":
        for ft in top_feats:
            for tt in VALID_TT_VALUES:
                items = [(score, iid) for iid, score in image_scores[ft][tt].items()]
                items.sort(key=lambda x: (-x[0], x[1]))
                top[ft][tt] = items[: args.top_n]
    else:
        for ft in top_feats:
            for tt in VALID_TT_VALUES:
                items = []
                for iid, ssum in image_sum[ft][tt].items():
                    cnt = image_cnt[ft][tt].get(iid, 0)
                    if cnt > 0:
                        items.append((ssum / cnt, iid))
                items.sort(key=lambda x: (-x[0], x[1]))
                top[ft][tt] = items[: args.top_n]

    # Caches for visualization
    parquet_cache = ParquetIndexCache()
    extractor_shard_cache: Dict[str, Dict[str, Any]] = {}
    extractor_meta_cache: Dict[str, Dict[int, Tuple[str, int]]] = {}
    image_rgb_cache: Dict[int, Image.Image] = {}
    act_heat_cache: Dict[Tuple[int, int, str], Image.Image] = {}

    def get_shard_path_and_obj_for_image_ix(image_ix: int) -> Tuple[Optional[Path], Optional[Dict[str, Any]]]:
        sp = find_extractor_shard_for_image_ix(image_ix, extractor_ranges, extractor_range_starts)
        if sp is None:
            return None, None
        key = str(sp)
        if key not in extractor_shard_cache:
            extractor_shard_cache[key] = torch.load(sp, map_location="cpu")
        return sp, extractor_shard_cache[key]

    def get_meta_map_for_shard(shard_path: Path, shard_obj: Dict[str, Any]) -> Dict[int, Tuple[str, int]]:
        sk = str(shard_path)
        if sk not in extractor_meta_cache:
            extractor_meta_cache[sk] = build_image_meta_index_from_shard(shard_obj)
        return extractor_meta_cache[sk]

    def get_image_for_extractor_image_ix(image_ix: int) -> Image.Image:
        if image_ix in image_rgb_cache:
            return image_rgb_cache[image_ix]

        shard_path, shard_obj = get_shard_path_and_obj_for_image_ix(image_ix)
        if shard_obj is None or shard_path is None:
            im = Image.new("RGB", (args.cell_size, args.cell_size), (128, 128, 128))
            image_rgb_cache[image_ix] = im
            return im

        meta_map = get_meta_map_for_shard(shard_path, shard_obj)
        meta = find_image_meta_for_image_ix(image_ix, parquet_dir, meta_map)

        if meta is None:
            im = Image.new("RGB", (args.cell_size, args.cell_size), (128, 128, 128))
        else:
            parquet_path, row_idx = meta
            im = pil_from_parquet_path_row(parquet_cache, parquet_path, row_idx, fallback_size=args.cell_size)

        image_rgb_cache[image_ix] = im
        return im

    labels = [
        (TT_REG, "reg"),
        (TT_HIGH_NORM, "high_norm_patch"),
        (TT_NORMAL, "normal_patch"),
    ]

    print("[info] writing activation overlay grids ...")
    for ft in tqdm(top_feats, desc="write grids", dynamic_ncols=True):
        # Full top-N per type
        for tt, name in labels:
            ranked = top[ft][tt]
            if not ranked:
                continue

            triplets: List[Tuple[Image.Image, Image.Image, Image.Image]] = []
            patch_subset = patch_subset_for_rank_type(tt)

            for _score, iid in ranked:
                im = get_image_for_extractor_image_ix(iid)
                im_sq = im.convert("RGB").resize((args.cell_size, args.cell_size), resample=Image.BICUBIC)

                cache_key = (iid, ft, patch_subset)
                if cache_key in act_heat_cache:
                    heat_im = act_heat_cache[cache_key]
                else:
                    _shard_path, shard_obj = get_shard_path_and_obj_for_image_ix(iid)
                    if shard_obj is None:
                        act_grid = torch.zeros((1, 1), dtype=torch.float32)
                    else:
                        act_grid = compute_sae_activation_patch_grid_from_shard(
                            shard_obj=shard_obj,
                            image_ix_target=iid,
                            feature_idx=ft,
                            W=W,
                            b=b,
                            mu=mu,
                            sigma=sigma,
                            num_prefix_tokens_fallback=5,
                            patch_subset=patch_subset,
                        )
                    heat_im = _grid_to_color_image(
                        act_grid,
                        out_wh=(args.cell_size, args.cell_size),
                        clip_q=args.clip_q,
                    )
                    act_heat_cache[cache_key] = heat_im

                overlay_im = _overlay(im_sq, heat_im, alpha=args.overlay_alpha)
                triplets.append((im_sq, overlay_im, heat_im))

            cells = make_grid_cells(triplets, cell_size=args.cell_size)
            out_path = out_dir / f"feat_{ft:04d}_{name}.png"
            save_grid(cells, out_path, cols=args.grid_cols)

        # One-of-each-type summary grid (top-1 reg / high_norm / normal)
        summary_triplets: List[Tuple[Image.Image, Image.Image, Image.Image]] = []
        summary_order = [
            (TT_REG, "reg"),
            (TT_HIGH_NORM, "high_norm_patch"),
            (TT_NORMAL, "normal_patch"),
        ]

        for tt_summary, _name_summary in summary_order:
            ranked_summary = top[ft][tt_summary]
            if not ranked_summary:
                continue

            _score, iid = ranked_summary[0]
            im = get_image_for_extractor_image_ix(iid)
            im_sq = im.convert("RGB").resize((args.cell_size, args.cell_size), resample=Image.BICUBIC)

            patch_subset = patch_subset_for_rank_type(tt_summary)
            cache_key = (iid, ft, patch_subset)

            if cache_key in act_heat_cache:
                heat_im = act_heat_cache[cache_key]
            else:
                _shard_path, shard_obj = get_shard_path_and_obj_for_image_ix(iid)
                if shard_obj is None:
                    act_grid = torch.zeros((1, 1), dtype=torch.float32)
                else:
                    act_grid = compute_sae_activation_patch_grid_from_shard(
                        shard_obj=shard_obj,
                        image_ix_target=iid,
                        feature_idx=ft,
                        W=W,
                        b=b,
                        mu=mu,
                        sigma=sigma,
                        num_prefix_tokens_fallback=5,
                        patch_subset=patch_subset,
                    )
                heat_im = _grid_to_color_image(
                    act_grid,
                    out_wh=(args.cell_size, args.cell_size),
                    clip_q=args.clip_q,
                )
                act_heat_cache[cache_key] = heat_im

            overlay_im = _overlay(im_sq, heat_im, alpha=args.overlay_alpha)
            summary_triplets.append((im_sq, overlay_im, heat_im))

        if summary_triplets:
            summary_cells = make_grid_cells(summary_triplets, cell_size=args.cell_size)
            summary_out = out_dir / f"feat_{ft:04d}_one_of_each_type.png"
            save_grid(summary_cells, summary_out, cols=3)

    print(f"[done] saved grids to: {out_dir}")


if __name__ == "__main__":
    main()