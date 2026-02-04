import os, json, math, random, re
from pathlib import Path
from typing import List

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, Dinov2WithRegistersModel

ROOT = Path("/lambda/nfs/neel/Research")

# ---- config ----
# CHANGED: use COCO train2017
COCO_DIR = ROOT / "datasets" / "coco" / "train2017"

OUT_BASE = ROOT / "runs" / "dinov2" / "coco_train2017_sae"
MODEL_ID = "facebook/dinov2-with-registers-base"

LAYER_1_INDEXED = 6          # user-chosen
K_OUTLIER = 4                # user-chosen
K_NONOUTLIER = 4             # matched control
BATCH_SIZE = 32

# CHANGED: train2017 has ~118k images; set to None to use all
MAX_IMAGES = None            # None = all, or set an int to cap (e.g., 50_000)

SHARD_IMAGES = 2000          # images per shard
DTYPE = torch.float16

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# COCO filenames: COCO_train2017_000000123456.jpg
COCO_ID_RE = re.compile(r".*_(\d{12})\.(jpg|jpeg|png)$", re.IGNORECASE)

def list_images(img_dir: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png"}
    paths = [p for p in img_dir.iterdir() if p.suffix.lower() in exts]
    paths.sort()
    return paths

def coco_image_id_from_path(p: Path) -> int:
    stem = p.stem
    if stem.isdigit():
        return int(stem)

    m = COCO_ID_RE.match(p.name)
    if m:
        return int(m.group(1))

    return abs(hash(p.name)) % (10**12)

# CHANGED: avoid leaking file handles
def load_pils(paths: List[Path]) -> List[Image.Image]:
    out = []
    for p in paths:
        with Image.open(p) as im:
            out.append(im.convert("RGB").copy())
    return out

@torch.inference_mode()
def main():
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    out_dir = OUT_BASE / f"layer{LAYER_1_INDEXED}_block_output_k{K_OUTLIER}"
    out_dir.mkdir(parents=True, exist_ok=True)
    shards_dir = out_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    proc = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = Dinov2WithRegistersModel.from_pretrained(MODEL_ID).eval().to(DEVICE)
    R = int(model.config.num_register_tokens)

    # layer index for HF encoder: 0-based
    layer_idx = LAYER_1_INDEXED - 1
    assert layer_idx >= 0, "Layer must be >= 1"
    assert layer_idx < len(model.encoder.layer), "Layer too large for this model"

    img_paths = list_images(COCO_DIR)
    if MAX_IMAGES is not None:
        img_paths = img_paths[:MAX_IMAGES]

    meta = {
        "model_id": MODEL_ID,
        "num_register_tokens": R,
        "layer_1_indexed": LAYER_1_INDEXED,
        "layer_0_indexed": layer_idx,
        "tap": "encoder.layer[layer_idx] output (block output)",
        "k_outlier": K_OUTLIER,
        "k_nonoutlier": K_NONOUTLIER,
        "dtype": str(DTYPE),
        "coco_dir": str(COCO_DIR),
        "num_images": len(img_paths),
        "shard_images": SHARD_IMAGES,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    # forward hook to capture layer output
    captured = {}
    def hook_fn(_module, _inp, out):
        if isinstance(out, tuple):
            captured["h"] = out[0]
        else:
            captured["h"] = out

    hnd = model.encoder.layer[layer_idx].register_forward_hook(hook_fn)

    shard_idx = 0
    buf_vecs = []
    buf_imgid = []
    buf_toktype = []
    buf_tokpos = []

    # token_type encoding
    TT_REG = 0
    TT_OUT = 1
    TT_NON = 2

    def flush():
        nonlocal shard_idx, buf_vecs, buf_imgid, buf_toktype, buf_tokpos
        if not buf_vecs:
            return
        vecs = torch.cat(buf_vecs, dim=0)  # (M, D)
        imgid = torch.tensor(buf_imgid, dtype=torch.int64)
        toktype = torch.tensor(buf_toktype, dtype=torch.int8)
        tokpos = torch.tensor(buf_tokpos, dtype=torch.int16)

        shard_path = shards_dir / f"shard_{shard_idx:05d}.pt"
        torch.save(
            {
                "vecs": vecs,          # (M, D) float16
                "image_id": imgid,     # (M,)
                "token_type": toktype, # (M,) 0=reg,1=outlier_patch,2=nonoutlier_patch
                "token_pos": tokpos,   # (M,) position in token sequence (0=CLS, 1..R regs, rest patches)
            },
            shard_path,
        )
        shard_idx += 1
        buf_vecs, buf_imgid, buf_toktype, buf_tokpos = [], [], [], []

    pbar = tqdm(total=len(img_paths), desc="extract", dynamic_ncols=True)
    processed_since_flush = 0

    for s in range(0, len(img_paths), BATCH_SIZE):
        batch_paths = img_paths[s:s+BATCH_SIZE]
        pils = load_pils(batch_paths)
        inputs = proc(images=pils, return_tensors="pt").to(DEVICE)

        captured.clear()
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(DEVICE == "cuda")):
            _ = model(**inputs)

        h = captured.get("h", None)
        if h is None:
            raise RuntimeError("Hook did not capture hidden states. Check HF version/model output.")

        # h: (B, 1+R+N, D)
        B, T, D = h.shape
        assert T > 1 + R, "Unexpected token count"
        patches = h[:, 1+R:, :]          # (B, N, D)
        regs = h[:, 1:1+R, :]            # (B, R, D)

        # outliers by patch token norm at this layer
        patch_norm = patches.norm(dim=-1)                # (B, N)
        k = min(K_OUTLIER, patches.shape[1])
        out_idx = patch_norm.topk(k, dim=1).indices      # (B, k)

        # non-outliers: sample k from remaining patch indices
        N = patches.shape[1]
        non_k = min(K_NONOUTLIER, max(0, N - k))

        img_ids = [coco_image_id_from_path(p) for p in batch_paths]
        for i in range(B):
            # regs: positions 1..R
            reg_vec = regs[i].to("cpu", dtype=DTYPE)  # (R,D)
            buf_vecs.append(reg_vec)
            buf_imgid.extend([img_ids[i]] * R)
            buf_toktype.extend([TT_REG] * R)
            buf_tokpos.extend(list(range(1, 1+R)))

            # outlier patches
            oi = out_idx[i]                                  # (k,)
            out_vec = patches[i, oi].to("cpu", dtype=DTYPE)   # (k,D)
            buf_vecs.append(out_vec)
            buf_imgid.extend([img_ids[i]] * k)
            buf_toktype.extend([TT_OUT] * k)
            buf_tokpos.extend(((1+R) + oi).to("cpu").tolist())

            if non_k > 0:
                mask = torch.ones(N, device=patches.device, dtype=torch.bool)
                mask[oi] = False
                avail = torch.nonzero(mask, as_tuple=False).squeeze(1)  # (N-k,)

                g = torch.Generator(device=patches.device)
                g.manual_seed(int(img_ids[i]) % (2**31 - 1))
                perm = avail[torch.randperm(avail.numel(), generator=g, device=patches.device)]
                ni = perm[:non_k]

                non_vec = patches[i, ni].to("cpu", dtype=DTYPE)  # (non_k,D)
                buf_vecs.append(non_vec)
                buf_imgid.extend([img_ids[i]] * non_k)
                buf_toktype.extend([TT_NON] * non_k)
                buf_tokpos.extend(((1+R) + ni).to("cpu").tolist())

        pbar.update(len(batch_paths))

        processed_since_flush += len(batch_paths)
        if processed_since_flush >= SHARD_IMAGES:
            flush()
            processed_since_flush = 0

    pbar.close()
    flush()
    hnd.remove()
    print("done:", out_dir)

if __name__ == "__main__":
    main()
