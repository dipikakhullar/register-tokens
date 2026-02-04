#!/usr/bin/env python3
"""
extract_imagenet1k_layer6_tokens.py

Reads ImageNet-1k parquet shards from:
  /lambda/nfs/neel/Research/datasets/imagenet1k/data/*.parquet

Important:
- Depending on how the parquet was produced, pyarrow may expose either:
    * ['bytes', 'path', 'label']  (common HF-style)
    * ['image', 'label']          (we observed this in row-group reads)
  BUT in practice, pf.schema.names can disagree with the schema returned by
  iter_batches(). This script probes iter_batches() per-file to choose the
  correct image column robustly.

Key design:
- Reads raw image bytes via pyarrow RecordBatch access (no .to_pydict()).
- Extracts DINOv2 hidden states at encoder layer LAYER_1_INDEXED (block output).
- Saves token vectors for:
    * register tokens
    * top K_OUTLIER patch tokens by norm
    * K_NONOUTLIER randomly sampled non-outlier patch tokens

Shard format:
  vecs, image_id, token_type, token_pos

image_id is a packed int64 encoding (parquet_file_index, row_in_file):
  image_id = (file_idx << 32) | row_idx
"""

import io
import json
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional

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
K_OUTLIER = 4
K_NONOUTLIER = 4
BATCH_SIZE = 32

MAX_IMAGES: Optional[int] = None   # None = all
SHARD_IMAGES = 2000               # images per shard flush
DTYPE_SAVE = torch.float16

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TT_REG = 0
TT_OUT = 1
TT_NON = 2


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


@torch.inference_mode()
def main():
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    out_dir = OUT_BASE / f"layer{LAYER_1_INDEXED}_block_output_k{K_OUTLIER}"
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

    meta = {
        "model_id": MODEL_ID,
        "num_register_tokens": R,
        "layer_1_indexed": LAYER_1_INDEXED,
        "layer_0_indexed": layer_idx,
        "tap": "encoder.layer[layer_idx] output (block output)",
        "k_outlier": K_OUTLIER,
        "k_nonoutlier": K_NONOUTLIER,
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
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    captured: Dict[str, torch.Tensor] = {}

    def hook_fn(_module, _inp, out):
        captured["h"] = out[0] if isinstance(out, tuple) else out

    hnd = model.encoder.layer[layer_idx].register_forward_hook(hook_fn)

    shard_idx = 0
    buf_vecs: List[torch.Tensor] = []
    buf_imgid: List[int] = []
    buf_toktype: List[int] = []
    buf_tokpos: List[int] = []

    def flush():
        nonlocal shard_idx, buf_vecs, buf_imgid, buf_toktype, buf_tokpos
        if not buf_vecs:
            return

        vecs = torch.cat(buf_vecs, dim=0)
        imgid = torch.tensor(buf_imgid, dtype=torch.int64)
        toktype = torch.tensor(buf_toktype, dtype=torch.int8)
        tokpos = torch.tensor(buf_tokpos, dtype=torch.int16)

        shard_path = shards_dir / f"shard_{shard_idx:05d}.pt"
        torch.save(
            {"vecs": vecs, "image_id": imgid, "token_type": toktype, "token_pos": tokpos},
            shard_path,
        )

        shard_idx += 1
        buf_vecs, buf_imgid, buf_toktype, buf_tokpos = [], [], [], []

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
                patches = h[:, 1 + R :, :]
                regs = h[:, 1 : 1 + R, :]

                patch_norm = patches.norm(dim=-1)
                k = min(K_OUTLIER, patches.shape[1])
                out_idx = patch_norm.topk(k, dim=1).indices

                N = patches.shape[1]
                non_k = min(K_NONOUTLIER, max(0, N - k))

                for i in range(B):
                    pid = packed_ids[i]

                    reg_vec = regs[i].to("cpu", dtype=DTYPE_SAVE)
                    buf_vecs.append(reg_vec)
                    buf_imgid.extend([pid] * R)
                    buf_toktype.extend([TT_REG] * R)
                    buf_tokpos.extend(list(range(1, 1 + R)))

                    oi = out_idx[i]
                    out_vec = patches[i, oi].to("cpu", dtype=DTYPE_SAVE)
                    buf_vecs.append(out_vec)
                    buf_imgid.extend([pid] * k)
                    buf_toktype.extend([TT_OUT] * k)
                    buf_tokpos.extend(((1 + R) + oi).to("cpu").tolist())

                    if non_k > 0:
                        mask = torch.ones(N, device=patches.device, dtype=torch.bool)
                        mask[oi] = False
                        avail = torch.nonzero(mask, as_tuple=False).squeeze(1)

                        g = torch.Generator(device=patches.device)
                        g.manual_seed(int(pid) % (2**31 - 1))
                        perm = avail[torch.randperm(avail.numel(), generator=g, device=patches.device)]
                        ni = perm[:non_k]

                        non_vec = patches[i, ni].to("cpu", dtype=DTYPE_SAVE)
                        buf_vecs.append(non_vec)
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
