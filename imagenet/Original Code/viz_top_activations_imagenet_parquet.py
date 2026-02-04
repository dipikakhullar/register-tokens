#!/usr/bin/env python3
"""
viz_top_activations_imagenet_parquet.py

COCO-style viz logic, but for ImageNet parquet-backed images.

Shards:
- image_id is packed int64:
    image_id = (file_idx << 32) | row_idx

Parquet:
- Row groups may expose either "image" or "bytes" column.
- Columns may be chunked; we use combine_chunks() before row indexing.

Behavior (matches COCO):
- Pass 1: choose NUM_FEATURES_TO_VIZ by global mean activation
- Pass 2: for each selected feature + token type, keep TOP_N highest-activation tokens
  (duplicates ARE allowed, matching the COCO script)
- Writes grids for reg/outlier/nonoutlier for each selected feature

Scope:
- Scans ALL shard_*.pt in SHARDS_DIR (entire extracted dataset).
"""

import io
import math
import bisect
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

# --- paths ---
RUN_DIR = Path("/lambda/nfs/neel/Research/runs/dinov2/imagenet1k_sae/layer6_block_output_k4")
SHARDS_DIR = RUN_DIR / "shards"

SAE_DIR = RUN_DIR / "sae_1024"
SAE_PATH = SAE_DIR / "sae.pt"

PARQUET_DIR = Path("/lambda/nfs/neel/Research/datasets/imagenet1k/data")
PARQUET_FILES = sorted(glob(str(PARQUET_DIR / "*.parquet")))
if not PARQUET_FILES:
    raise FileNotFoundError(f"No parquet files found in {PARQUET_DIR}")

OUT_DIR = SAE_DIR / "viz"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- viz config ---
TOP_N = 5
NUM_FEATURES_TO_VIZ = 80
BATCH = 5000

TT_REG = 0
TT_OUT = 1
TT_NON = 2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def unpack_image_id(packed: int) -> Tuple[int, int]:
    file_idx = (int(packed) >> 32) & 0xFFFFFFFF
    row_idx = int(packed) & 0xFFFFFFFF
    return file_idx, row_idx


def _bytes_from_cell(cell: Any) -> Optional[bytes]:
    """
    Supports:
      - bytes / bytearray
      - dict-like cells containing raw bytes under keys like 'bytes' or 'data'
    """
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
        self._pf: Dict[int, pq.ParquetFile] = {}
        self._rg_prefix: Dict[int, List[int]] = {}
        self._nrows: Dict[int, int] = {}

    def get_pf(self, file_idx: int) -> pq.ParquetFile:
        if file_idx not in self._pf:
            self._pf[file_idx] = pq.ParquetFile(PARQUET_FILES[file_idx])
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


CACHE = ParquetIndexCache()


def pil_from_parquet(file_idx: int, row_idx: int, fallback_size: int = 224) -> Image.Image:
    if file_idx < 0 or file_idx >= len(PARQUET_FILES):
        return Image.new("RGB", (fallback_size, fallback_size))

    total_rows = CACHE.get_total_rows(file_idx)
    if row_idx < 0 or row_idx >= total_rows:
        return Image.new("RGB", (fallback_size, fallback_size))

    pf = CACHE.get_pf(file_idx)
    prefix = CACHE.get_rowgroup_prefix(file_idx)

    rg = bisect.bisect_right(prefix, row_idx) - 1
    rg = max(0, min(rg, pf.num_row_groups - 1))
    in_rg = row_idx - prefix[rg]

    table = pf.read_row_group(rg)

    # prefer 'image' if present, else 'bytes'
    if "image" in table.column_names:
        img_col = "image"
    elif "bytes" in table.column_names:
        img_col = "bytes"
    else:
        return Image.new("RGB", (fallback_size, fallback_size))

    if in_rg < 0 or in_rg >= table.num_rows:
        return Image.new("RGB", (fallback_size, fallback_size))

    arr = table[img_col].combine_chunks()
    cell = arr[in_rg].as_py()

    b = _bytes_from_cell(cell)
    if b is None:
        return Image.new("RGB", (fallback_size, fallback_size))

    try:
        with Image.open(io.BytesIO(b)) as im:
            return im.convert("RGB").copy()
    except Exception:
        return Image.new("RGB", (fallback_size, fallback_size))


def shard_paths(shards_dir: Path):
    paths = sorted(shards_dir.glob("shard_*.pt"))
    if not paths:
        raise FileNotFoundError(f"No shards found in {shards_dir}")
    return paths


def make_grid_from_pils(pils: List[Image.Image], out_path: Path, cols: int = 5, size: int = 224):
    imgs = []
    for im in pils:
        try:
            imgs.append(im.convert("RGB").resize((size, size)))
        except Exception:
            imgs.append(Image.new("RGB", (size, size)))

    rows = math.ceil(len(imgs) / cols)
    grid = Image.new("RGB", (cols * size, rows * size))
    for i, im in enumerate(imgs):
        r = i // cols
        c = i % cols
        grid.paste(im, (c * size, r * size))
    grid.save(out_path)


def main():
    ckpt = torch.load(SAE_PATH, map_location="cpu")
    sd = ckpt["state_dict"]
    mu = ckpt["mu"]
    sigma = ckpt["sigma"]

    W = sd["enc.weight"].to(torch.float32).to(DEVICE)
    b = sd["enc.bias"].to(torch.float32).to(DEVICE)
    mu = mu.to(torch.float32).to(DEVICE)
    sigma = sigma.to(torch.float32).to(DEVICE)

    num_features = W.shape[0]
    print("loaded SAE, num_features =", num_features)

    paths = shard_paths(SHARDS_DIR)

    # ---- pass 1: pick features by global mean activation (COCO logic) ----
    feat_sum = torch.zeros(num_features, device=DEVICE)
    feat_cnt = 0

    for p in tqdm(paths, desc="score features", dynamic_ncols=True):
        shard = torch.load(p, map_location="cpu")
        X = shard["vecs"].to(torch.float32)
        n = X.shape[0]
        for s in range(0, n, BATCH):
            xb = X[s : s + BATCH].to(DEVICE)
            xb = (xb - mu) / sigma
            a = F.relu(xb @ W.t() + b)  # (B, F)
            feat_sum += a.sum(dim=0)
            feat_cnt += a.size(0)

    feat_mean = feat_sum / max(1, feat_cnt)
    top_feats = torch.topk(feat_mean, k=min(NUM_FEATURES_TO_VIZ, num_features)).indices.tolist()
    print("selected features:", top_feats[:10], "...")

    # ---- pass 2: collect TOP_N by token activation (duplicates allowed, COCO logic) ----
    top = {ft: {TT_REG: [], TT_OUT: [], TT_NON: []} for ft in top_feats}

    def push(ft: int, ttype: int, act_val: float, packed_id: int):
        arr = top[ft][ttype]
        arr.append((act_val, packed_id))
        arr.sort(key=lambda x: x[0], reverse=True)
        if len(arr) > TOP_N:
            arr.pop()

    for p in tqdm(paths, desc="collect tops", dynamic_ncols=True):
        shard = torch.load(p, map_location="cpu")
        X = shard["vecs"].to(torch.float32)
        imgid = shard["image_id"].to(torch.int64)
        ttype = shard["token_type"].to(torch.int64)

        n = X.shape[0]
        for s in range(0, n, BATCH):
            xb = X[s : s + BATCH].to(DEVICE)
            xb = (xb - mu) / sigma
            a = F.relu(xb @ W[top_feats].t() + b[top_feats])  # (B, F_sel)

            img_b = imgid[s : s + BATCH]
            tt_b = ttype[s : s + BATCH]

            for j, ft in enumerate(top_feats):
                acts = a[:, j].detach().cpu()
                for i in range(acts.numel()):
                    tt = int(tt_b[i].item())
                    if tt not in (TT_REG, TT_OUT, TT_NON):
                        continue
                    push(ft, tt, float(acts[i].item()), int(img_b[i].item()))

    # cache decoded images across all grids
    img_cache: Dict[int, Image.Image] = {}

    # ---- write grids ----
    for ft in top_feats:
        for tt, name in [(TT_REG, "reg"), (TT_OUT, "outlier"), (TT_NON, "nonoutlier")]:
            packed_ids = [pid for _, pid in top[ft][tt]]
            pils: List[Image.Image] = []
            for pid in packed_ids:
                if pid in img_cache:
                    pils.append(img_cache[pid])
                    continue
                file_idx, row_idx = unpack_image_id(pid)
                im = pil_from_parquet(file_idx, row_idx)
                img_cache[pid] = im
                pils.append(im)

            out_path = OUT_DIR / f"feat_{ft:04d}_{name}.png"
            make_grid_from_pils(pils, out_path)

    print("saved grids to:", OUT_DIR)


if __name__ == "__main__":
    main()
