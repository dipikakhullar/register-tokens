#!/usr/bin/env python3
"""
extract_imagenet1k_layer6_tokens.py

Paper-aligned scope (Vision Transformers Need Registers, ICLR 2024):
- "Outlier" patch tokens are HIGH-NORM tokens (upper tail of L2 norms), i.e. artifacts. :contentReference[oaicite:0]{index=0}
- "Non-outlier" patch tokens are all remaining (non-high-norm) patch tokens.
- Registers are saved as their own token type; in register-trained models, high-norm behavior is
  expected to shift from patches into registers. :contentReference[oaicite:1]{index=1}

Changes vs your current script:
1) Outlier selection is threshold/upper-tail based (not symmetric; not mean±std).
2) Saves per-token L2 norm alongside vecs for downstream analysis/visualization.
3) Non-outliers are sampled from the complement set (norm <= cutoff).
4) Robust handling when an image has <K_OUTLIER outliers (save all available, optionally fill with top-k if desired).

Shard format (now):
  vecs, norms, image_id, token_type, token_pos

token_pos: position in the original token sequence at the tapped layer output.
"""

import io
import json
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pyarrow.parquet as pq
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, Dinov2WithRegistersModel

ROOT = Path("/lambda/nfs/neel/Research")

PARQUET_DIR = ROOT / "datasets" / "imagenet1k" / "data"
OUT_BASE = ROOT / "runs" / "dinov2" / "imagenet1k_sae"

MODEL_ID = "facebook/dinov2-with-registers-base"

# ---- extraction config ----
LAYER_1_INDEXED = 6
K_OUTLIER = 4            # max outliers to save PER IMAGE (upper-tail/high-norm only)
K_NONOUTLIER = 4         # non-outliers sampled from (norm <= cutoff)
BATCH_SIZE = 32

MAX_IMAGES: Optional[int] = None   # None = all
SHARD_IMAGES = 2000               # images per shard flush
DTYPE_SAVE = torch.float16

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Token types
TT_REG = 0          # register tokens
TT_HIGH_NORM = 1    # high-norm patch tokens ("outliers" in paper)
TT_NON = 2 

# ---- Paper-aligned outlier definition (choose ONE approach) ----
# Option A (paper-style): fixed cutoff (varies by model; paper used 150 as an example for DINOv2-g/14). :contentReference[oaicite:2]{index=2}
OUTLIER_NORM_CUTOFF: Optional[float] = None  # e.g., 150.0

# Option B: percentile-based cutoff (recommended for portability across layers/models)
# Example: 0.98 means "top 2% highest-norm patch tokens are outliers"
OUTLIER_PERCENTILE: Optional[float] = 0.98

# If both are set, OUTLIER_NORM_CUTOFF takes precedence.
# If neither is set, defaults to percentile 0.98.
# --------------------------------------------


def pack_image_id(file_idx: int, row_idx: int) -> int:
    return (int(file_idx) << 32) | int(row_idx)


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


def decode_pil_from_cell(cell: Any, fallback_size: int = 224) -> Image.Image:
    b = _bytes_from_cell(cell)
    if b is None:
        return Image.new("RGB", (fallback_size, fallback_size))
    try:
        with Image.open(io.BytesIO(b)) as im:
            return im.convert("RGB").copy()
    except Exception:
        return Image.new("RGB", (fallback_size, fallback_size))


def list_parquet_files(parquet_dir: Path) -> List[str]:
    files = sorted(glob(str(parquet_dir / "*.parquet")))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {parquet_dir}")
    return files


def _compute_cutoff_from_percentile(patch_norm: torch.Tensor, pct: float) -> torch.Tensor:
    """
    patch_norm: (B, N) float tensor
    returns cutoff per-image: (B,) where cutoff is the pct-quantile
    """
    pct = float(pct)
    if not (0.0 < pct < 1.0):
        raise ValueError("OUTLIER_PERCENTILE must be between 0 and 1 (exclusive).")
    # torch.quantile supports per-row quantiles on recent versions; fall back gracefully if needed.
    return torch.quantile(patch_norm, pct, dim=1)


def _select_outlier_indices(
    patch_norm: torch.Tensor,
    k_outlier: int,
    cutoff: Optional[float],
    percentile: Optional[float],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      out_idx_padded: (B, K) indices into patch tokens (0..N-1), padded with -1 if fewer than K
      out_counts: (B,) number of valid outliers per image (<=K)

    Paper-aligned: outliers are HIGH-NORM tokens only (upper tail). :contentReference[oaicite:3]{index=3}
    """
    B, N = patch_norm.shape
    K = min(int(k_outlier), N)
    if K <= 0:
        out_idx = torch.full((B, 0), -1, device=patch_norm.device, dtype=torch.long)
        out_counts = torch.zeros((B,), device=patch_norm.device, dtype=torch.long)
        return out_idx, out_counts

    if cutoff is None and percentile is None:
        percentile = 0.98

    if cutoff is not None:
        # constant cutoff for all images
        cut = torch.full((B,), float(cutoff), device=patch_norm.device, dtype=patch_norm.dtype)
        cutoff_kind = "fixed"
    else:
        # per-image percentile cutoff
        cut = _compute_cutoff_from_percentile(patch_norm, float(percentile))  # (B,)
        cutoff_kind = f"percentile_{percentile}"

    # mask outliers: norm > cutoff (strictly greater, matches "norm larger than cutoff" language)
    mask = patch_norm > cut[:, None]  # (B, N)
    out_counts = mask.sum(dim=1).clamp(max=K)  # cap at K for saving

    # For each image: if >K outliers, take the top-K by norm among them.
    # If <K outliers, take all and pad with -1.
    out_idx_padded = torch.full((B, K), -1, device=patch_norm.device, dtype=torch.long)

    # Efficient per-row selection:
    # - set non-outliers to -inf, take topk K -> gives candidate indices, then filter by mask validity
    masked_vals = patch_norm.masked_fill(~mask, float("-inf"))
    topk = torch.topk(masked_vals, K, dim=1)
    cand_idx = topk.indices  # (B,K)
    cand_vals = topk.values  # (B,K)

    # Valid where value != -inf
    valid = cand_vals.isfinite()
    # Place valid indices in order; keep -1 for invalid
    out_idx_padded[valid] = cand_idx[valid]

    return out_idx_padded, out_counts


@torch.inference_mode()
def main():
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    out_dir = OUT_BASE / f"layer{LAYER_1_INDEXED}_high_norm_tokens"

    out_dir.mkdir(parents=True, exist_ok=True)
    shards_dir = out_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    parquet_files = list_parquet_files(PARQUET_DIR)

    proc = AutoImageProcessor.from_pretrained(MODEL_ID, use_fast=True)
    model = Dinov2WithRegistersModel.from_pretrained(MODEL_ID).eval().to(DEVICE)
    R = int(model.config.num_register_tokens)

    layer_idx = LAYER_1_INDEXED - 1
    if layer_idx < 0:
        raise ValueError("LAYER_1_INDEXED must be >= 1")
    if layer_idx >= len(model.encoder.layer):
        raise ValueError("Layer too large for this model")

    # total rows (metadata only)
    total_rows = 0
    for fp in parquet_files:
        total_rows += pq.ParquetFile(fp).metadata.num_rows
    if MAX_IMAGES is not None:
        total_rows = min(total_rows, MAX_IMAGES)

    outlier_def = {
        "paper_aligned_definition": "outlier == high-norm patch token (upper tail only)",
        "paper_note": "cutoff varies by model; 'high-norm' and 'outlier' used interchangeably",  # :contentReference[oaicite:4]{index=4}
        "outlier_norm_cutoff": OUTLIER_NORM_CUTOFF,
        "outlier_percentile": OUTLIER_PERCENTILE,
        "outliers_selected_per_image_max": K_OUTLIER,
        "non_outliers_sampled_per_image": K_NONOUTLIER,
        "thresholding_rule": "norm > cutoff",
    }

    meta = {
        "model_id": MODEL_ID,
        "num_register_tokens": R,
        "layer_1_indexed": LAYER_1_INDEXED,
        "layer_0_indexed": layer_idx,
        "tap": "encoder.layer[layer_idx] output (block output)",
        "dtype_save": str(DTYPE_SAVE),
        "parquet_dir": str(PARQUET_DIR),
        "parquet_glob": str(PARQUET_DIR / "*.parquet"),
        "num_parquet_files": len(parquet_files),
        "max_images": MAX_IMAGES,
        "num_images_planned": int(total_rows),
        "shard_images": SHARD_IMAGES,
        "batch_size": BATCH_SIZE,
        "image_id_encoding": "packed(file_idx<<32 | row_in_file)",
        "supported_image_cols": ["bytes", "image"],
        "note": "image column chosen by probing iter_batches() schema per file",
        "outlier_definition": outlier_def,
        "shard_keys": ["vecs", "norms", "image_id", "token_type", "token_pos"],
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    captured: Dict[str, torch.Tensor] = {}

    def hook_fn(_module, _inp, out):
        captured["h"] = out[0] if isinstance(out, tuple) else out

    hnd = model.encoder.layer[layer_idx].register_forward_hook(hook_fn)

    shard_idx = 0
    buf_vecs: List[torch.Tensor] = []
    buf_norms: List[torch.Tensor] = []
    buf_imgid: List[int] = []
    buf_toktype: List[int] = []
    buf_tokpos: List[int] = []

    def flush():
        nonlocal shard_idx, buf_vecs, buf_norms, buf_imgid, buf_toktype, buf_tokpos
        if not buf_vecs:
            return

        vecs = torch.cat(buf_vecs, dim=0)
        norms = torch.cat(buf_norms, dim=0)  # float32 or float16; we’ll store float16 for space
        imgid = torch.tensor(buf_imgid, dtype=torch.int64)
        toktype = torch.tensor(buf_toktype, dtype=torch.int8)
        tokpos = torch.tensor(buf_tokpos, dtype=torch.int16)

        shard_path = shards_dir / f"shard_{shard_idx:05d}.pt"
        torch.save(
            {"vecs": vecs, "norms": norms, "image_id": imgid, "token_type": toktype, "token_pos": tokpos},
            shard_path,
        )

        shard_idx += 1
        buf_vecs, buf_norms, buf_imgid, buf_toktype, buf_tokpos = [], [], [], [], []

    processed_images_total = 0
    processed_since_flush = 0

    pbar = tqdm(total=total_rows, desc="extract", dynamic_ncols=True)

    try:
        for file_idx, fp in enumerate(parquet_files):
            if MAX_IMAGES is not None and processed_images_total >= MAX_IMAGES:
                break

            pf = pq.ParquetFile(fp)

            # Probe actual batch schema (pf.schema can disagree with iter_batches schema)
            probe = next(pf.iter_batches(batch_size=1), None)
            if probe is None:
                continue
            probe_cols = list(probe.schema.names)

            if "bytes" in probe_cols:
                img_col_name = "bytes"
            elif "image" in probe_cols:
                img_col_name = "image"
            else:
                raise RuntimeError(f"No supported image column in {fp}. Probe columns: {probe_cols}")

            row_offset = 0

            for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=[img_col_name]):
                if MAX_IMAGES is not None and processed_images_total >= MAX_IMAGES:
                    break

                if batch.num_columns == 0:
                    raise RuntimeError(
                        f"Got RecordBatch with 0 columns for {fp} using column '{img_col_name}'. "
                        f"Probe columns were: {probe_cols}"
                    )

                img_col = batch.column(0)
                bsz = batch.num_rows

                pils: List[Image.Image] = []
                packed_ids: List[int] = []

                for j in range(bsz):
                    if MAX_IMAGES is not None and processed_images_total + len(pils) >= MAX_IMAGES:
                        break

                    cell = img_col[j].as_py()
                    pil = decode_pil_from_cell(cell)
                    pils.append(pil)
                    packed_ids.append(pack_image_id(file_idx, row_offset + j))

                if not pils:
                    row_offset += bsz
                    continue

                inputs = proc(images=pils, return_tensors="pt").to(DEVICE)

                captured.clear()
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(DEVICE == "cuda")):
                    _ = model(**inputs)

                h = captured.get("h", None)
                if h is None:
                    raise RuntimeError("Hook did not capture hidden states.")

                B, T, D = h.shape
                patches = h[:, 1 + R :, :]     # (B, N, D)
                regs = h[:, 1 : 1 + R, :]      # (B, R, D)
                N = patches.shape[1]

                patch_norm = patches.norm(dim=-1)  # (B, N)
                reg_norm = regs.norm(dim=-1)       # (B, R)

                # Paper-aligned outliers: high-norm patch tokens (upper tail only). :contentReference[oaicite:5]{index=5}
                out_idx_padded, out_counts = _select_outlier_indices(
                    patch_norm=patch_norm,
                    k_outlier=K_OUTLIER,
                    cutoff=OUTLIER_NORM_CUTOFF,
                    percentile=OUTLIER_PERCENTILE,
                )

                for i in range(B):
                    pid = packed_ids[i]

                    # ---- register tokens ----
                    reg_vec = regs[i].to("cpu", dtype=DTYPE_SAVE)  # (R,D)
                    reg_n = reg_norm[i].to("cpu", dtype=torch.float16)  # (R,)
                    buf_vecs.append(reg_vec)
                    buf_norms.append(reg_n)
                    buf_imgid.extend([pid] * R)
                    buf_toktype.extend([TT_REG] * R)
                    buf_tokpos.extend(list(range(1, 1 + R)))

                    # ---- outlier patch tokens ----
                    oi = out_idx_padded[i]  # (K,) with -1 padding
                    valid_mask = oi >= 0
                    if valid_mask.any():
                        oi_valid = oi[valid_mask]
                        out_vec = patches[i, oi_valid].to("cpu", dtype=DTYPE_SAVE)  # (k_i, D)
                        out_n = patch_norm[i, oi_valid].to("cpu", dtype=torch.float16)  # (k_i,)
                        buf_vecs.append(out_vec)
                        buf_norms.append(out_n)
                        buf_imgid.extend([pid] * int(oi_valid.numel()))
                        buf_toktype.extend([TT_HIGH_NORM] * int(oi_valid.numel()))
                        buf_tokpos.extend(((1 + R) + oi_valid).to("cpu").tolist())

                    # ---- non-outlier patch tokens (complement set) ----
                    # sample from indices that are NOT outliers (norm <= cutoff)
                    # To ensure complement matches the outlier rule, we recompute the same cutoff here.
                    if OUTLIER_NORM_CUTOFF is not None:
                        cut_i = float(OUTLIER_NORM_CUTOFF)
                    else:
                        pct = OUTLIER_PERCENTILE if OUTLIER_PERCENTILE is not None else 0.98
                        cut_i = float(torch.quantile(patch_norm[i], float(pct)).item())

                    non_mask = patch_norm[i] <= cut_i
                    avail = torch.nonzero(non_mask, as_tuple=False).squeeze(1)

                    if avail.numel() > 0 and K_NONOUTLIER > 0:
                        non_k = min(K_NONOUTLIER, int(avail.numel()))

                        g = torch.Generator(device=patches.device)
                        g.manual_seed(int(pid) % (2**31 - 1))
                        perm = avail[torch.randperm(avail.numel(), generator=g, device=patches.device)]
                        ni = perm[:non_k]

                        non_vec = patches[i, ni].to("cpu", dtype=DTYPE_SAVE)  # (non_k, D)
                        non_n = patch_norm[i, ni].to("cpu", dtype=torch.float16)  # (non_k,)
                        buf_vecs.append(non_vec)
                        buf_norms.append(non_n)
                        buf_imgid.extend([pid] * non_k)
                        buf_toktype.extend([TT_NON] * non_k)
                        buf_tokpos.extend(((1 + R) + ni).to("cpu").tolist())

                processed_images_total += B
                processed_since_flush += B
                pbar.update(B)

                if processed_since_flush >= SHARD_IMAGES:
                    flush()
                    processed_since_flush = 0

                row_offset += bsz

        flush()
    finally:
        hnd.remove()
        pbar.close()

    print("done:", out_dir)


if __name__ == "__main__":
    main()
