#!/usr/bin/env python3
"""
top5_images_per_feature_first10.py

Compute top-k images per SAE feature from extracted DINOv2 shards, with multiple
aggregation perspectives and optional activation/attention overlays.

Designed to work with extraction shards from:
  extract_dinov2_tokens_and_attn.py
and SAE checkpoints from:
  train_overcomplete_sae.py

Fixes included:
- No pandas dependency (uses pyarrow for parquet image reads)
- Register-token aggregation bug fixed (no scatter_reduce shape mismatch)
- Heap tie bug fixed (stable tiebreaker)
- Perspective-correct activation overlays
- --save_png now acts as convenience flag (enables original + overlays)
- Consistent aggregate_feature_scores_per_image return signature
- General cleanup / progress prints

Perspectives saved per feature:
1) all_tokens          (register + normal patch + high-norm patch)
2) registers           (register tokens only)
3) high_norm_patches   (token_bucket == 2)
4) normal_patches      (token_bucket == 1)
5) all_patches         (normal + high-norm patches)

Outputs:
<out_dir>/
  feature_0000/
    top5_all_tokens.json
    top5_registers.json
    top5_high_norm_patches.json
    top5_normal_patches.json
    top5_all_patches.json
    rank01_all_tokens_orig.png
    rank01_all_tokens_activation_overlay.png
    rank01_all_tokens_attention_overlay.png      (if attention available + applicable)
    ...
  run_meta.json

Notes:
- "activation overlay" = patch-grid heatmap derived from SAE feature activations
  aggregated over the perspective token subset (spatial patch-based perspectives only).
- "attention overlay" = uses stored CLS->patch or REG->patch summaries from extraction shards.
- For register perspective attention overlay, uses weighted mean(REG->patch) over selected register slots.
- For non-register perspectives, attention overlay defaults to CLS->patch.
"""

from __future__ import annotations

import argparse
import json
import math
import heapq
import itertools
from dataclasses import dataclass
from pathlib import Path
from io import BytesIO
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from PIL import Image

import torch
from tqdm.auto import tqdm

import pyarrow.parquet as pq


# ----------------------------
# Small utilities
# ----------------------------
def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, indent=2))


def load_image_from_bytes(b) -> Image.Image:
    if isinstance(b, memoryview):
        b = b.tobytes()
    return Image.open(BytesIO(b)).convert("RGB")


def colorize_heatmap_np(hm: np.ndarray) -> np.ndarray:
    """
    hm: [H, W] in [0,1]
    returns RGB uint8 [H, W, 3]
    Simple blue->cyan->yellow->red colormap without matplotlib dependency.
    """
    x = np.clip(hm.astype(np.float32), 0.0, 1.0)

    # piecewise ramps
    r = np.clip(2.0 * x - 0.5, 0.0, 1.0)
    g = np.clip(2.0 * x, 0.0, 1.0)
    b = np.clip(1.5 - 2.0 * x, 0.0, 1.0)

    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255.0).astype(np.uint8)


def overlay_heatmap_on_image(
    image: Image.Image,
    heatmap_2d: np.ndarray,
    alpha: float = 0.45,
) -> Image.Image:
    """
    heatmap_2d expected in [0,1], shape [Gh, Gw]
    """
    w, h = image.size
    hm_img = Image.fromarray(colorize_heatmap_np(heatmap_2d), mode="RGB").resize((w, h), Image.BILINEAR)
    return Image.blend(image.convert("RGB"), hm_img, alpha=alpha)


def infer_grid_hw(num_patches: int) -> Tuple[int, int]:
    """
    Try to infer patch grid HxW from number of patches.
    """
    s = int(round(math.sqrt(num_patches)))
    if s * s == num_patches:
        return s, s

    best = (1, num_patches)
    best_diff = num_patches
    for h in range(1, int(math.sqrt(num_patches)) + 1):
        if num_patches % h == 0:
            w = num_patches // h
            diff = abs(w - h)
            if diff < best_diff:
                best = (h, w)
                best_diff = diff
    return best


# heap tie-safe top-k
_HEAP_COUNTER = itertools.count()


def upsert_topk_heap(heap: List[Tuple[float, int, Dict[str, Any]]], score: float, item: Dict[str, Any], k: int) -> None:
    """
    Min-heap of size <= k.
    Stores (score, tiebreaker, item) so ties do not compare dicts.
    """
    tup = (float(score), next(_HEAP_COUNTER), item)
    if len(heap) < k:
        heapq.heappush(heap, tup)
    else:
        if score > heap[0][0]:
            heapq.heapreplace(heap, tup)


def heap_to_sorted_list(heap: List[Tuple[float, int, Dict[str, Any]]]) -> List[Dict[str, Any]]:
    return [x[2] for x in sorted(heap, key=lambda t: t[0], reverse=True)]


# ----------------------------
# SAE loading + forward helpers
# ----------------------------
def load_sae_checkpoint(ckpt_path: Path, device: str):
    """
    Loads TopKSAE from overcomplete package and checkpoint state_dict.
    """
    try:
        from overcomplete.sae import TopKSAE
    except Exception:
        from overcomplete import TopKSAE  # type: ignore

    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "sae_state" not in ckpt:
        raise KeyError(f"Checkpoint missing 'sae_state': {ckpt_path}")

    state = ckpt["sae_state"]

    # Infer dimensions from state dict
    d_model = None
    nb_concepts = None
    for k, v in state.items():
        if isinstance(v, torch.Tensor) and v.ndim == 2:
            a, b = v.shape
            if (a > b and b in (384, 512, 768, 1024, 1536)) or (b > a and a in (384, 512, 768, 1024, 1536)):
                nb_concepts = max(a, b)
                d_model = min(a, b)
                break

    if d_model is None or nb_concepts is None:
        raise RuntimeError("Could not infer SAE dimensions from checkpoint state_dict.")

    # top_k may not be stored; default from your training setup
    top_k = int(ckpt.get("top_k", 20)) if isinstance(ckpt, dict) else 20

    sae = TopKSAE(d_model, nb_concepts, top_k=top_k, device=device).to(device)
    sae.load_state_dict(state, strict=False)
    sae.eval()

    return sae, {
        "d_model": d_model,
        "nb_concepts": nb_concepts,
        "top_k": top_k,
        "step": ckpt.get("step", None),
    }


def unpack_overcomplete_output(out):
    if isinstance(out, (tuple, list)) and len(out) == 3:
        z_pre, z, x_hat = out
        return x_hat, z_pre, z

    def _get_attr(o, names):
        for n in names:
            if hasattr(o, n):
                return getattr(o, n)
        return None

    x_hat = _get_attr(out, ["x_hat", "xhat", "recons", "reconstruction"])
    z_pre = _get_attr(out, ["pre_codes", "z_pre", "precode", "zpre"])
    z = _get_attr(out, ["codes", "z", "code"])

    if x_hat is None or z_pre is None or z is None:
        raise TypeError("Could not unpack overcomplete output.")
    return x_hat, z_pre, z


# ----------------------------
# Parquet image cache / retrieval
# ----------------------------
class ParquetImageReader:
    """
    Lazy pyarrow parquet reader cache keyed by parquet path.
    Reads rows by index and decodes image.bytes.
    """

    def __init__(self):
        self._tables: Dict[str, Any] = {}

    def _get_table(self, parquet_path: str):
        if parquet_path not in self._tables:
            # Try with label column first, then fallback
            try:
                self._tables[parquet_path] = pq.read_table(parquet_path, columns=["image", "label"])
            except Exception:
                self._tables[parquet_path] = pq.read_table(parquet_path, columns=["image"])
        return self._tables[parquet_path]

    def read_image(self, parquet_path: str, row_idx: int) -> Tuple[Image.Image, Optional[int], str]:
        table = self._get_table(parquet_path)
        if row_idx < 0 or row_idx >= table.num_rows:
            raise IndexError(f"Row {row_idx} out of range for {parquet_path}")

        row = table.slice(row_idx, 1)
        image_col = row.column("image")[0].as_py()
        label = None
        if "label" in row.column_names:
            try:
                label_val = row.column("label")[0].as_py()
                if label_val is not None:
                    label = int(label_val)
            except Exception:
                label = None

        b = image_col.get("bytes", None)
        if b is None:
            raise ValueError(f"Missing image.bytes in {parquet_path} row {row_idx}")
        img = load_image_from_bytes(b)
        img_path = str(image_col.get("path", "")) if image_col.get("path", None) is not None else ""
        return img, label, img_path


# ----------------------------
# Aggregation logic
# ----------------------------
@dataclass
class PerspectiveAgg:
    score_per_image: torch.Tensor                    # [B, F]
    patch_heat_per_image: Optional[torch.Tensor]     # [B, F, P] or None
    reg_slot_scores_per_image: Optional[torch.Tensor]  # [B, F, R] or None


def aggregate_feature_scores_per_image(
    acts_feat: torch.Tensor,        # [T, F]
    image_ix: torch.Tensor,         # [T]
    token_pos: torch.Tensor,        # [T]
    token_bucket: torch.Tensor,     # [T] 0=reg,1=normal_patch,2=high_norm_patch
    num_prefix_tokens: int,
    num_register_tokens: int,
    selected_perspective: str,
    agg_mode: str = "mean",
) -> Tuple[PerspectiveAgg, torch.Tensor]:
    """
    Returns:
      PerspectiveAgg
      unique_imgs [B] (global image ids corresponding to rows)
    """
    device = acts_feat.device
    Fdim = acts_feat.shape[1]

    unique_imgs = torch.unique(image_ix, sorted=True)
    B = unique_imgs.numel()
    if B == 0:
        raise RuntimeError("No images in shard after filtering.")

    # map global image id -> local index
    img_min = int(unique_imgs[0].item())
    contiguous = bool(
        torch.equal(
            unique_imgs,
            torch.arange(img_min, img_min + B, device=device, dtype=unique_imgs.dtype),
        )
    )
    if contiguous:
        local_img = image_ix.long() - img_min
    else:
        img_to_local = {int(g.item()): i for i, g in enumerate(unique_imgs)}
        local_img = torch.tensor([img_to_local[int(g.item())] for g in image_ix], device=device, dtype=torch.long)

    is_reg = (token_bucket == 0)
    is_norm_patch = (token_bucket == 1)
    is_high_patch = (token_bucket == 2)
    is_patch = is_norm_patch | is_high_patch

    if selected_perspective == "all_tokens":
        score_mask = torch.ones_like(token_bucket, dtype=torch.bool)
    elif selected_perspective == "registers":
        score_mask = is_reg
    elif selected_perspective == "high_norm_patches":
        score_mask = is_high_patch
    elif selected_perspective == "normal_patches":
        score_mask = is_norm_patch
    elif selected_perspective == "all_patches":
        score_mask = is_patch
    else:
        raise ValueError(f"Unknown perspective: {selected_perspective}")

    # image-level aggregate score
    t_img = local_img[score_mask]         # [Tm]
    t_act = acts_feat[score_mask]         # [Tm, F]
    counts = torch.zeros((B,), device=device, dtype=torch.float32)
    if t_img.numel() > 0:
        counts.index_add_(0, t_img, torch.ones((t_img.shape[0],), device=device, dtype=torch.float32))

    if agg_mode == "mean":
        score_per_image = torch.zeros((B, Fdim), device=device, dtype=torch.float32)
        if t_act.numel() > 0:
            score_per_image.index_add_(0, t_img, t_act.to(torch.float32))
        denom = counts.clamp_min(1.0).unsqueeze(1)
        score_per_image = score_per_image / denom
    elif agg_mode == "max":
        score_per_image = torch.full((B, Fdim), float("-inf"), device=device, dtype=torch.float32)
        if t_act.numel() > 0:
            for fi in range(Fdim):
                vals = t_act[:, fi].to(torch.float32)
                out = torch.full((B,), float("-inf"), device=device, dtype=torch.float32)
                out.scatter_reduce_(0, t_img, vals, reduce="amax", include_self=True)
                score_per_image[:, fi] = out
        score_per_image = torch.where(torch.isfinite(score_per_image), score_per_image, torch.zeros_like(score_per_image))
    else:
        raise ValueError("agg_mode must be 'mean' or 'max'")

    patch_heat_per_image = None
    reg_slot_scores = None

    # Perspective-correct patch heatmaps (for activation overlays)
    # registers perspective has no spatial patch activation heatmap
    if selected_perspective in ("all_tokens", "all_patches"):
        patch_overlay_mask = is_patch
    elif selected_perspective == "high_norm_patches":
        patch_overlay_mask = is_high_patch
    elif selected_perspective == "normal_patches":
        patch_overlay_mask = is_norm_patch
    elif selected_perspective == "registers":
        patch_overlay_mask = torch.zeros_like(is_patch, dtype=torch.bool)
    else:
        patch_overlay_mask = torch.zeros_like(is_patch, dtype=torch.bool)

    if patch_overlay_mask.any():
        patch_pos = token_pos[patch_overlay_mask].long() - int(num_prefix_tokens)   # [Tp], 0..P-1
        patch_img = local_img[patch_overlay_mask]
        patch_act = acts_feat[patch_overlay_mask].to(torch.float32)                  # [Tp, F]

        if patch_pos.numel() > 0:
            pmin = int(patch_pos.min().item())
            pmax = int(patch_pos.max().item())
            if pmin < 0:
                raise RuntimeError(
                    f"Patch positions produced negative slot index. min={pmin}, "
                    f"num_prefix_tokens={num_prefix_tokens}. Check token_pos semantics."
                )
            P = pmax + 1

            patch_sum = torch.zeros((B, Fdim, P), device=device, dtype=torch.float32)
            patch_cnt = torch.zeros((B, P), device=device, dtype=torch.float32)

            patch_cnt.index_put_(
                (patch_img, patch_pos),
                torch.ones_like(patch_pos, dtype=torch.float32),
                accumulate=True,
            )

            for fi in range(Fdim):
                vals = patch_act[:, fi]
                patch_sum[:, fi, :].index_put_((patch_img, patch_pos), vals, accumulate=True)

            patch_den = patch_cnt.clamp_min(1.0).unsqueeze(1)  # [B,1,P]
            patch_heat_per_image = patch_sum / patch_den

    # Register slot scores (fixed)
    if num_register_tokens > 0 and is_reg.any():
        reg_mask = is_reg
        reg_img = local_img[reg_mask]                                # [Tr]
        reg_pos = token_pos[reg_mask].long()                         # [Tr], expected 1..R if CLS removed
        reg_slot = reg_pos - 1                                       # [Tr], expected 0..R-1
        reg_act = acts_feat[reg_mask].to(torch.float32)              # [Tr, F]

        if reg_slot.numel() > 0:
            rmin = int(reg_slot.min().item())
            rmax = int(reg_slot.max().item())
            if rmin < 0 or rmax >= num_register_tokens:
                raise RuntimeError(
                    f"Register slot out of range. slot min/max=({rmin},{rmax}), "
                    f"num_register_tokens={num_register_tokens}. "
                    "This may mean token_bucket==0 includes non-register prefix tokens."
                )

        reg_slot_scores = torch.zeros((B, Fdim, num_register_tokens), device=device, dtype=torch.float32)
        reg_slot_cnt = torch.zeros((B, num_register_tokens), device=device, dtype=torch.float32)

        reg_slot_cnt.index_put_(
            (reg_img, reg_slot),
            torch.ones_like(reg_slot, dtype=torch.float32),
            accumulate=True,
        )

        for fi in range(Fdim):
            vals = reg_act[:, fi]
            reg_slot_scores[:, fi, :].index_put_((reg_img, reg_slot), vals, accumulate=True)

        reg_slot_scores = reg_slot_scores / reg_slot_cnt.clamp_min(1.0).unsqueeze(1)

    return PerspectiveAgg(
        score_per_image=score_per_image,
        patch_heat_per_image=patch_heat_per_image,
        reg_slot_scores_per_image=reg_slot_scores,
    ), unique_imgs


# ----------------------------
# Attention extraction
# ----------------------------
def get_attention_map_for_image(
    obj: Dict[str, Any],
    local_img_idx: int,
    perspective: str,
    reg_slot_scores_for_feature: Optional[torch.Tensor],  # [R] for this image-feature
) -> Optional[torch.Tensor]:
    """
    Returns patch attention vector [P] on CPU float32 if available.
    """
    has_cls = "cls_to_patch_attn" in obj
    has_reg = "reg_to_patch_attn" in obj
    if not has_cls and not has_reg:
        return None

    if perspective == "registers" and has_reg and reg_slot_scores_for_feature is not None:
        reg_attn = obj["reg_to_patch_attn"]  # [B, R, P] CPU
        if local_img_idx < 0 or local_img_idx >= reg_attn.shape[0]:
            return None
        ra = reg_attn[local_img_idx].to(torch.float32)  # [R, P]
        w = reg_slot_scores_for_feature.to(torch.float32).detach().cpu()
        if w.numel() != ra.shape[0]:
            return ra.mean(dim=0)

        w = torch.clamp(w, min=0.0)
        s = float(w.sum().item())
        if s <= 0:
            return ra.mean(dim=0)
        w = w / s
        return (w[:, None] * ra).sum(dim=0)

    if has_cls:
        cls_attn = obj["cls_to_patch_attn"]  # [B, P] CPU
        if local_img_idx < 0 or local_img_idx >= cls_attn.shape[0]:
            return None
        return cls_attn[local_img_idx].to(torch.float32)

    return None


# ----------------------------
# Main analysis
# ----------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--extract_dir", type=str, required=True)
    ap.add_argument("--sae_run_dir", type=str, required=True, help="Directory with norm_stats.pt and checkpoints/")
    ap.add_argument("--output_dir", type=str, required=True)

    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--feature_start", type=int, default=0)
    ap.add_argument("--num_features", type=int, default=10)
    ap.add_argument("--top_k_images", type=int, default=5)

    ap.add_argument("--agg_mode", type=str, choices=["mean", "max"], default="mean")
    ap.add_argument("--token_batch_size", type=int, default=65536)

    ap.add_argument("--ckpt_name", type=str, default="best.pt")

    # save flags
    ap.add_argument("--save_png", action="store_true", help="Convenience flag: save original + overlays")
    ap.add_argument("--save_attention_overlay", action="store_true")
    ap.add_argument("--save_activation_overlay", action="store_true")
    ap.add_argument("--save_original", action="store_true")
    ap.add_argument("--overlay_alpha", type=float, default=0.45)

    args = ap.parse_args()

    if args.save_png:
        args.save_original = True
        args.save_activation_overlay = True
        args.save_attention_overlay = True

    extract_dir = Path(args.extract_dir)
    sae_run_dir = Path(args.sae_run_dir)
    output_dir = Path(args.output_dir)
    safe_mkdir(output_dir)

    shard_paths = sorted(extract_dir.glob("extract_hook*_shard*.pt"))
    if not shard_paths:
        raise FileNotFoundError(f"No extract shards found in {extract_dir}")

    norm_stats_path = sae_run_dir / "norm_stats.pt"
    ckpt_path = sae_run_dir / "checkpoints" / args.ckpt_name
    if not norm_stats_path.exists():
        raise FileNotFoundError(f"Missing norm_stats.pt: {norm_stats_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    # Load norm stats
    stats = torch.load(norm_stats_path, map_location="cpu")
    mean = stats["mean"].to(torch.float32)
    std = stats["std"].to(torch.float32)

    # Load SAE
    sae, sae_meta = load_sae_checkpoint(ckpt_path, device=args.device)

    # Extraction meta from first shard
    first_obj = torch.load(shard_paths[0], map_location="cpu")
    num_prefix_tokens = int(first_obj.get("num_prefix_tokens", 1))
    num_register_tokens = int(first_obj.get("num_register_tokens", 0))
    has_attention = ("cls_to_patch_attn" in first_obj) or ("reg_to_patch_attn" in first_obj)

    feature_ids = list(range(args.feature_start, args.feature_start + args.num_features))
    if len(feature_ids) == 0:
        raise ValueError("num_features must be > 0")

    print(f"[INFO] shards={len(shard_paths)}  features={feature_ids[0]}..{feature_ids[-1]}  topK={args.top_k_images}")
    print(f"[INFO] num_prefix_tokens={num_prefix_tokens} num_register_tokens={num_register_tokens}")
    print(f"[INFO] attention stored in extraction: {has_attention}")

    perspectives = [
        "all_tokens",
        "registers",
        "high_norm_patches",
        "normal_patches",
        "all_patches",
    ]

    # topk heaps: perspective -> feature_id -> heap[(score, tiebreaker, item)]
    topk: Dict[str, Dict[int, List[Tuple[float, int, Dict[str, Any]]]]] = {
        p: {fid: [] for fid in feature_ids} for p in perspectives
    }

    # Pass 1: score candidates
    shard_bar = tqdm(enumerate(shard_paths, start=1), total=len(shard_paths), desc="Scoring shards")
    for shard_i, sp in shard_bar:
        print(f"[Shard {shard_i}/{len(shard_paths)}] {sp.name}")
        obj = torch.load(sp, map_location="cpu")

        tokens = obj["tokens"].to(torch.float32)          # [T, D]
        image_ix = obj["image_ix"].to(torch.long)         # [T]
        token_pos = obj["token_pos"].to(torch.long)       # [T]
        if "token_bucket" not in obj:
            raise KeyError(f"{sp.name} missing token_bucket; rerun extraction with token labels.")
        token_bucket = obj["token_bucket"].to(torch.long)  # [T]

        # Drop CLS token for SAE encoding (match train split logic)
        keep = token_pos != 0
        tokens = tokens[keep]
        image_ix = image_ix[keep]
        token_pos = token_pos[keep]
        token_bucket = token_bucket[keep]

        # Normalize
        x = (tokens - mean) / std

        # Encode in batches, store selected feature activations only
        acts_chunks = []
        for start in range(0, x.shape[0], args.token_batch_size):
            xb = x[start:start + args.token_batch_size].to(args.device, non_blocking=True)
            with torch.no_grad():
                out = sae(xb)
                _x_hat, _z_pre, z = unpack_overcomplete_output(out)
                acts_sel = z[:, feature_ids].detach().to(torch.float32).cpu()
            acts_chunks.append(acts_sel)

        acts_feat = torch.cat(acts_chunks, dim=0)  # [T', Fsel]

        # Aggregate for each perspective
        for perspective in perspectives:
            agg, unique_imgs = aggregate_feature_scores_per_image(
                acts_feat=acts_feat,
                image_ix=image_ix,
                token_pos=token_pos,
                token_bucket=token_bucket,
                num_prefix_tokens=num_prefix_tokens,
                num_register_tokens=num_register_tokens,
                selected_perspective=perspective,
                agg_mode=args.agg_mode,
            )
            scores = agg.score_per_image  # [B, Fsel]

            # metadata arrays from shard payload are per-image in shard order
            src_parquet = obj["src_parquet"]
            src_row = obj["src_row"]
            image_path = obj.get("image_path", [""] * len(src_row))

            min_global = int(unique_imgs[0].item())
            contiguous = bool(
                torch.equal(
                    unique_imgs,
                    torch.arange(min_global, min_global + unique_imgs.numel(), dtype=unique_imgs.dtype),
                )
            )
            img_to_local_meta = None if contiguous else {int(g.item()): i for i, g in enumerate(unique_imgs)}

            for bi in range(scores.shape[0]):
                global_img_id = int(unique_imgs[bi].item())
                local_meta_idx = (global_img_id - min_global) if contiguous else img_to_local_meta[global_img_id]  # type: ignore[index]
                if local_meta_idx < 0 or local_meta_idx >= len(src_row):
                    continue

                for fj, fid in enumerate(feature_ids):
                    score = float(scores[bi, fj].item())
                    if not np.isfinite(score):
                        continue

                    item = {
                        "score": score,
                        "feature_id": fid,
                        "shard_path": str(sp),
                        "shard_name": sp.name,
                        "global_image_id": global_img_id,
                        # local row in this perspective's unique_imgs tensor
                        "local_image_idx_in_shard": int(bi),
                        "src_parquet": str(src_parquet[local_meta_idx]),
                        "src_row": int(src_row[local_meta_idx]),
                        "image_path": str(image_path[local_meta_idx]) if local_meta_idx < len(image_path) else "",
                        "perspective": perspective,
                    }
                    upsert_topk_heap(topk[perspective][fid], score, item, args.top_k_images)

    # Convert heaps to sorted lists
    topk_sorted: Dict[str, Dict[int, List[Dict[str, Any]]]] = {
        p: {fid: heap_to_sorted_list(topk[p][fid]) for fid in feature_ids}
        for p in perspectives
    }

    # Save summary JSONs
    for fid in feature_ids:
        feat_dir = output_dir / f"feature_{fid:04d}"
        safe_mkdir(feat_dir)
        for perspective in perspectives:
            save_json(
                feat_dir / f"top5_{perspective}.json",
                {
                    "feature_id": fid,
                    "perspective": perspective,
                    "agg_mode": args.agg_mode,
                    "top_k": args.top_k_images,
                    "items": topk_sorted[perspective][fid],
                },
            )

    # Pass 2: generate overlays / originals only for selected winners
    do_render = args.save_original or args.save_activation_overlay or args.save_attention_overlay
    if do_render:
        print("[INFO] Generating PNGs for selected top-k items ...")

        # Group requests by shard to avoid reloading many times
        requests_by_shard: Dict[str, List[Tuple[str, int, int, Dict[str, Any]]]] = {}
        # entries: (perspective, feature_id, rank, item)
        for perspective in perspectives:
            for fid in feature_ids:
                for rank, item in enumerate(topk_sorted[perspective][fid], start=1):
                    requests_by_shard.setdefault(item["shard_path"], []).append((perspective, fid, rank, item))

        pq_reader = ParquetImageReader()

        for shard_path_str, reqs in tqdm(requests_by_shard.items(), desc="Render shards"):
            obj = torch.load(shard_path_str, map_location="cpu")

            tokens = obj["tokens"].to(torch.float32)
            image_ix = obj["image_ix"].to(torch.long)
            token_pos = obj["token_pos"].to(torch.long)
            token_bucket = obj["token_bucket"].to(torch.long)

            keep = token_pos != 0
            tokens = tokens[keep]
            image_ix = image_ix[keep]
            token_pos = token_pos[keep]
            token_bucket = token_bucket[keep]

            x = (tokens - mean) / std

            acts_chunks = []
            for start in range(0, x.shape[0], args.token_batch_size):
                xb = x[start:start + args.token_batch_size].to(args.device, non_blocking=True)
                with torch.no_grad():
                    out = sae(xb)
                    _x_hat, _z_pre, z = unpack_overcomplete_output(out)
                    acts_sel = z[:, feature_ids].detach().to(torch.float32).cpu()
                acts_chunks.append(acts_sel)
            acts_feat = torch.cat(acts_chunks, dim=0)  # [T, Fsel]

            # Precompute all perspectives aggregations for this shard once
            precomp: Dict[str, Tuple[PerspectiveAgg, torch.Tensor]] = {}
            for perspective in perspectives:
                precomp[perspective] = aggregate_feature_scores_per_image(
                    acts_feat=acts_feat,
                    image_ix=image_ix,
                    token_pos=token_pos,
                    token_bucket=token_bucket,
                    num_prefix_tokens=num_prefix_tokens,
                    num_register_tokens=num_register_tokens,
                    selected_perspective=perspective,
                    agg_mode=args.agg_mode,
                )

            for perspective, fid, rank, item in reqs:
                feat_dir = output_dir / f"feature_{fid:04d}"
                local_img_idx = int(item["local_image_idx_in_shard"])
                fj = fid - feature_ids[0]

                agg, _unique_imgs = precomp[perspective]
                patch_heat = None
                reg_slot_scores_vec = None

                if agg.patch_heat_per_image is not None:
                    if 0 <= local_img_idx < agg.patch_heat_per_image.shape[0] and 0 <= fj < agg.patch_heat_per_image.shape[1]:
                        patch_heat = agg.patch_heat_per_image[local_img_idx, fj].detach().cpu().numpy()  # [P]

                if agg.reg_slot_scores_per_image is not None:
                    if 0 <= local_img_idx < agg.reg_slot_scores_per_image.shape[0] and 0 <= fj < agg.reg_slot_scores_per_image.shape[1]:
                        reg_slot_scores_vec = agg.reg_slot_scores_per_image[local_img_idx, fj].detach().cpu()  # [R]

                img, label, img_path_actual = pq_reader.read_image(item["src_parquet"], int(item["src_row"]))

                stem = f"rank{rank:02d}_{perspective}"

                if args.save_original:
                    img.save(feat_dir / f"{stem}_orig.png")

                if args.save_activation_overlay and patch_heat is not None:
                    P = patch_heat.shape[0]
                    gh, gw = infer_grid_hw(P)
                    hm = patch_heat.reshape(gh, gw).astype(np.float32)

                    hm_min = float(hm.min()) if hm.size else 0.0
                    hm_max = float(hm.max()) if hm.size else 0.0
                    if hm_max > hm_min:
                        hm_n = (hm - hm_min) / (hm_max - hm_min)
                    else:
                        hm_n = np.zeros_like(hm, dtype=np.float32)

                    over = overlay_heatmap_on_image(img, hm_n, alpha=args.overlay_alpha)
                    over.save(feat_dir / f"{stem}_activation_overlay.png")

                if args.save_attention_overlay and has_attention:
                    attn_vec = get_attention_map_for_image(
                        obj=obj,
                        local_img_idx=local_img_idx,
                        perspective=perspective,
                        reg_slot_scores_for_feature=reg_slot_scores_vec,
                    )
                    if attn_vec is not None:
                        attn_np = attn_vec.detach().cpu().numpy().astype(np.float32)
                        P = attn_np.shape[0]
                        gh, gw = infer_grid_hw(P)
                        hm = attn_np.reshape(gh, gw)

                        hm_min = float(hm.min()) if hm.size else 0.0
                        hm_max = float(hm.max()) if hm.size else 0.0
                        if hm_max > hm_min:
                            hm_n = (hm - hm_min) / (hm_max - hm_min)
                        else:
                            hm_n = np.zeros_like(hm, dtype=np.float32)

                        over = overlay_heatmap_on_image(img, hm_n, alpha=args.overlay_alpha)
                        over.save(feat_dir / f"{stem}_attention_overlay.png")

                # Optional: enrich JSON provenance if needed later
                _ = (label, img_path_actual)

    run_meta = {
        "extract_dir": str(extract_dir),
        "sae_run_dir": str(sae_run_dir),
        "checkpoint": str(ckpt_path),
        "norm_stats": str(norm_stats_path),
        "feature_start": args.feature_start,
        "num_features": args.num_features,
        "top_k_images": args.top_k_images,
        "agg_mode": args.agg_mode,
        "num_prefix_tokens": num_prefix_tokens,
        "num_register_tokens": num_register_tokens,
        "attention_available": has_attention,
        "sae_meta": sae_meta,
        "perspectives": perspectives,
        "save_original": bool(args.save_original),
        "save_activation_overlay": bool(args.save_activation_overlay),
        "save_attention_overlay": bool(args.save_attention_overlay),
    }
    save_json(output_dir / "run_meta.json", run_meta)

    print("[DONE] Wrote outputs to:", output_dir)


if __name__ == "__main__":
    main()