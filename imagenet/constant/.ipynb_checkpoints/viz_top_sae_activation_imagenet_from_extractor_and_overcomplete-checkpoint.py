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
"""

from __future__ import annotations

import argparse
import bisect
import io
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
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


def build_extractor_image_range_index(paths: List[Path]) -> List[Tuple[int, int, Path]]:
    ranges: List[Tuple[int, int, Path]] = []
    for p in tqdm(paths, desc="index extractor shards", dynamic_ncols=True):
        obj = torch.load(p, map_location="cpu")
        image_ix = obj["image_ix"].to(torch.int64)
        if image_ix.numel() == 0:
            continue
        ranges.append((int(image_ix.min().item()), int(image_ix.max().item()), p))
    ranges.sort(key=lambda x: x[0])
    return ranges


def find_extractor_shard_for_image_ix(image_ix: int, ranges: List[Tuple[int, int, Path]]) -> Optional[Path]:
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
) -> torch.Tensor:
    """
    Direct SAE activation map for one image and feature from extractor tokens.
    """
    tokens = shard_obj["tokens"].to(torch.float32)       # CPU [T,D]
    image_ix = shard_obj["image_ix"].to(torch.int64)     # CPU [T]
    token_pos = shard_obj["token_pos"].to(torch.int64)   # CPU [T]

    mask_img = (image_ix == int(image_ix_target))
    if mask_img.sum().item() == 0:
        return torch.zeros((1, 1), dtype=torch.float32)

    x_img = tokens[mask_img]
    pos_img = token_pos[mask_img]

    # sort by token position so patch order is spatially correct
    order = torch.argsort(pos_img)
    x_img = x_img[order]
    pos_img = pos_img[order]

    num_prefix = int(shard_obj.get("num_prefix_tokens", num_prefix_tokens_fallback))
    patch_mask = pos_img >= num_prefix
    if patch_mask.sum().item() == 0:
        return torch.zeros((1, 1), dtype=torch.float32)

    x_patch = x_img[patch_mask].to(DEVICE)
    x_patch = (x_patch - mu) / sigma

    wf = W[feature_idx]
    bf = b[feature_idx]
    acts = F.relu(x_patch @ wf + bf)  # [P]

    return _reshape_patch_grid(acts.detach().cpu())


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
    print(f"[info] num_features={num_features}, d_model={d_model}")

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

        for p in tqdm(paths, desc="feature selection pass", dynamic_ncols=True):
            obj = torch.load(p, map_location="cpu")
            X = obj["tokens"].to(torch.float32)

            n = X.shape[0]
            for s in range(0, n, args.batch):
                xb = X[s:s + args.batch].to(DEVICE)
                xb = (xb - mu) / sigma
                a = F.relu(xb @ W.t() + b)  # (B,F)
                feat_sum += a.sum(dim=0)
                feat_cnt += a.shape[0]

        feat_mean = feat_sum / max(1, feat_cnt)
        k = min(args.num_features_to_viz, num_features)
        top_feats = torch.topk(feat_mean, k=k).indices.tolist()
        print(f"[info] selected features (first 10): {top_feats[:10]}")

    sel = torch.tensor(top_feats, device=DEVICE, dtype=torch.long)

    # Collect image-level scores by token type
    if args.agg_mode == "max":
        image_scores: Dict[int, Dict[int, Dict[int, float]]] = {
            ft: {TT_REG: {}, TT_NORMAL: {}, TT_HIGH_NORM: {}} for ft in top_feats
        }
    else:
        image_sum: Dict[int, Dict[int, Dict[int, float]]] = {
            ft: {TT_REG: {}, TT_NORMAL: {}, TT_HIGH_NORM: {}} for ft in top_feats
        }
        image_cnt: Dict[int, Dict[int, Dict[int, int]]] = {
            ft: {TT_REG: {}, TT_NORMAL: {}, TT_HIGH_NORM: {}} for ft in top_feats
        }

    print(f"[info] collecting image-level scores ({args.agg_mode}) ...")
    for p in tqdm(paths, desc="ranking pass", dynamic_ncols=True):
        obj = torch.load(p, map_location="cpu")
        X = obj["tokens"].to(torch.float32)
        imgid = obj["image_ix"].to(torch.int64)
        ttype = obj["token_bucket"].to(torch.int64)

        n = X.shape[0]
        for s in range(0, n, args.batch):
            xb = X[s:s + args.batch].to(DEVICE)
            xb = (xb - mu) / sigma
            a = F.relu(xb @ W[sel].t() + b[sel])  # (B, F_sel)

            img_b = imgid[s:s + args.batch].cpu()
            tt_b = ttype[s:s + args.batch].cpu()
            a_cpu = a.detach().cpu()

            B = a_cpu.shape[0]
            F_sel = a_cpu.shape[1]

            for i in range(B):
                tt = int(tt_b[i].item())
                if tt not in (TT_REG, TT_NORMAL, TT_HIGH_NORM):
                    continue
                pid = int(img_b[i].item())

                row = a_cpu[i]
                for j in range(F_sel):
                    ft = top_feats[j]
                    val = float(row[j].item())

                    if args.agg_mode == "max":
                        prev = image_scores[ft][tt].get(pid)
                        if (prev is None) or (val > prev):
                            image_scores[ft][tt][pid] = val
                    else:
                        image_sum[ft][tt][pid] = image_sum[ft][tt].get(pid, 0.0) + val
                        image_cnt[ft][tt][pid] = image_cnt[ft][tt].get(pid, 0) + 1

    top: Dict[int, Dict[int, List[Tuple[float, int]]]] = {
        ft: {TT_REG: [], TT_NORMAL: [], TT_HIGH_NORM: []} for ft in top_feats
    }

    for ft in top_feats:
        for tt in (TT_REG, TT_NORMAL, TT_HIGH_NORM):
            if args.agg_mode == "max":
                items = [(score, iid) for iid, score in image_scores[ft][tt].items()]
            else:
                items = []
                for iid, ssum in image_sum[ft][tt].items():
                    cnt = image_cnt[ft][tt].get(iid, 0)
                    if cnt > 0:
                        items.append((ssum / cnt, iid))

            items.sort(key=lambda x: (-x[0], x[1]))
            top[ft][tt] = items[: args.top_n]

    extractor_ranges = build_extractor_image_range_index(paths)

    # Caches
    parquet_cache = ParquetIndexCache()
    extractor_shard_cache: Dict[str, Dict[str, Any]] = {}
    extractor_meta_cache: Dict[str, Dict[int, Tuple[str, int]]] = {}
    image_rgb_cache: Dict[int, Image.Image] = {}
    act_heat_cache: Dict[Tuple[int, int], Image.Image] = {}

    def get_shard_path_and_obj_for_image_ix(image_ix: int) -> Tuple[Optional[Path], Optional[Dict[str, Any]]]:
        sp = find_extractor_shard_for_image_ix(image_ix, extractor_ranges)
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
        for tt, name in labels:
            ranked = top[ft][tt]
            if not ranked:
                continue

            triplets: List[Tuple[Image.Image, Image.Image, Image.Image]] = []

            for _score, iid in ranked:
                im = get_image_for_extractor_image_ix(iid)
                im_sq = im.convert("RGB").resize((args.cell_size, args.cell_size), resample=Image.BICUBIC)

                cache_key = (iid, ft)
                if cache_key in act_heat_cache:
                    heat_im = act_heat_cache[cache_key]
                else:
                    shard_path, shard_obj = get_shard_path_and_obj_for_image_ix(iid)
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

    print(f"[done] saved grids to: {out_dir}")


if __name__ == "__main__":
    main()