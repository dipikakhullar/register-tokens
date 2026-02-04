#!/usr/bin/env python3
import io
import math
import bisect
import os
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

RUN_DIR = Path("/lambda/nfs/neel/Research/runs/dinov2/imagenet1k_sae/layer6_block_output_k4")
SHARDS_DIR = RUN_DIR / "shards"

SAE_DIR = RUN_DIR / "sae_1024"
SAE_PATH = SAE_DIR / "sae.pt"

PARQUET_DIR = Path("/lambda/nfs/neel/Research/datasets/imagenet1k/data")
PARQUET_FILES = sorted(glob(str(PARQUET_DIR / "*.parquet")))
if not PARQUET_FILES:
    raise FileNotFoundError(f"No parquet files found in {PARQUET_DIR}")

OUT_DIR = SAE_DIR / "viz_smoke"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE = int(os.getenv("FEATURE", "0"))
MAX_SHARDS = int(os.getenv("MAX_SHARDS", "3"))
TOP_N = int(os.getenv("TOP_N", "5"))
BATCH = int(os.getenv("BATCH", "5000"))

TT_REG = 0
TT_OUT = 1
TT_NON = 2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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


def pil_from_parquet(file_idx: int, row_idx: int, packed_id: int, fallback_size: int = 224) -> Image.Image:
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


def shard_paths_limited(shards_dir: Path, max_shards: int):
    paths = sorted(shards_dir.glob("shard_*.pt"))
    if not paths:
        raise FileNotFoundError(f"No shards found in {shards_dir}")
    return paths[:max_shards]


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

    W_all = sd["enc.weight"].to(torch.float32).to(DEVICE)
    b_all = sd["enc.bias"].to(torch.float32).to(DEVICE)
    mu = mu.to(torch.float32).to(DEVICE)
    sigma = sigma.to(torch.float32).to(DEVICE)

    num_features = W_all.shape[0]
    if FEATURE < 0 or FEATURE >= num_features:
        raise ValueError(f"FEATURE={FEATURE} out of range (0..{num_features-1})")

    w = W_all[FEATURE]
    b = b_all[FEATURE]

    print("loaded SAE:", SAE_PATH)
    print("feature:", FEATURE, "num_features:", num_features)

    paths = shard_paths_limited(SHARDS_DIR, MAX_SHARDS)
    print("using shards:", len(paths))

    # Uniqueness within each token-type by image_id:
    # top_map[ttype][packed_id] = best_activation_for_that_image
    top_map: Dict[int, Dict[int, float]] = {TT_REG: {}, TT_OUT: {}, TT_NON: {}}

    def push(ttype: int, act_val: float, packed_id: int):
        prev = top_map[ttype].get(packed_id)
        if (prev is None) or (act_val > prev):
            top_map[ttype][packed_id] = act_val

    def finalize(ttype: int, exclude: Set[int]) -> List[int]:
        items = [(pid, act) for pid, act in top_map[ttype].items() if pid not in exclude]
        items.sort(key=lambda x: x[1], reverse=True)
        return [pid for pid, _ in items[:TOP_N]]

    for p in tqdm(paths, desc="collect tops (smoke)", dynamic_ncols=True):
        shard = torch.load(p, map_location="cpu")
        X = shard["vecs"].to(torch.float32)
        imgid = shard["image_id"].to(torch.int64)
        ttype = shard["token_type"].to(torch.int64)

        n = X.shape[0]
        for s in range(0, n, BATCH):
            xb = X[s : s + BATCH].to(DEVICE)
            xb = (xb - mu) / sigma
            acts = F.relu((xb * w).sum(dim=1) + b).detach().cpu()

            img_b = imgid[s : s + BATCH].cpu()
            tt_b = ttype[s : s + BATCH].cpu()

            for i in range(acts.numel()):
                tt = int(tt_b[i].item())
                if tt not in (TT_REG, TT_OUT, TT_NON):
                    continue
                push(tt, float(acts[i].item()), int(img_b[i].item()))

    name_map = {TT_REG: "reg", TT_OUT: "outlier", TT_NON: "nonoutlier"}

    # Disjoint selection across types:
    # - pick reg ids first
    # - outlier ids cannot reuse reg ids
    # - nonoutlier ids cannot reuse reg or outlier ids
    reg_ids = finalize(TT_REG, exclude=set())
    used = set(reg_ids)

    out_ids = finalize(TT_OUT, exclude=used)
    used |= set(out_ids)

    non_ids = finalize(TT_NON, exclude=used)
    used |= set(non_ids)

    print("reg unique ids:", reg_ids)
    print("outlier unique ids (disjoint):", out_ids)
    print("nonoutlier unique ids (disjoint):", non_ids)

    for tt, ids in [(TT_REG, reg_ids), (TT_OUT, out_ids), (TT_NON, non_ids)]:
        pils = []
        for packed in ids:
            file_idx, row_idx = unpack_image_id(packed)
            pils.append(pil_from_parquet(file_idx, row_idx, packed))

        out_path = OUT_DIR / f"smoke_feat_{FEATURE:04d}_{name_map[tt]}.png"
        make_grid_from_pils(pils, out_path)

    print("saved smoke grids to:", OUT_DIR)


if __name__ == "__main__":
    main()

