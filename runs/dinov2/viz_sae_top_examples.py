import math
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

SHARDS_DIR = Path("/home/ubuntu/neel/Research/runs/dinov2/coco_train2017_sae/layer6_block_output_k4/shards")

# Make sure this matches where you saved your SAE
SAE_PATH = SHARDS_DIR.parent / "sae_4096" / "sae.pt"

# CHANGED: use COCO train2017 images (since shards are from train)
COCO_DIR = Path("/home/ubuntu/neel/Research/datasets/coco/train2017")

OUT_DIR = SHARDS_DIR.parent / "sae_4096" / "viz"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TOP_N = 5
NUM_FEATURES_TO_VIZ = 80   # keep manageable
BATCH = 5000

TT_REG = 0
TT_OUT = 1
TT_NON = 2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def coco_path_from_id(image_id: int) -> Path:
    return COCO_DIR / f"{int(image_id):012d}.jpg"

def shard_paths(shards_dir: Path):
    paths = sorted(shards_dir.glob("shard_*.pt"))
    if not paths:
        raise FileNotFoundError(f"No shards found in {shards_dir}")
    return paths

def make_grid(img_paths, out_path: Path, cols=5, size=224):
    imgs = []
    for p in img_paths:
        try:
            with Image.open(p) as im:
                im = im.convert("RGB").resize((size, size))
                imgs.append(im.copy())  # detach from file handle
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

    # rebuild encoder weights only
    W = sd["enc.weight"].to(torch.float32).to(DEVICE)
    b = sd["enc.bias"].to(torch.float32).to(DEVICE)
    mu = mu.to(torch.float32).to(DEVICE)
    sigma = sigma.to(torch.float32).to(DEVICE)

    num_features = W.shape[0]
    print("loaded SAE, num_features =", num_features)

    paths = shard_paths(SHARDS_DIR)

    # pick features to visualize by global mean activation (one pass)
    feat_sum = torch.zeros(num_features, device=DEVICE)
    feat_cnt = 0

    for p in tqdm(paths, desc="score features", dynamic_ncols=True):
        shard = torch.load(p, map_location="cpu")
        X = shard["vecs"].to(torch.float32)
        n = X.shape[0]
        for s in range(0, n, BATCH):
            xb = X[s:s+BATCH].to(DEVICE)
            xb = (xb - mu) / sigma
            a = F.relu(xb @ W.t() + b)  # (B, F)
            feat_sum += a.sum(dim=0)
            feat_cnt += a.size(0)

    feat_mean = feat_sum / max(1, feat_cnt)
    top_feats = torch.topk(feat_mean, k=min(NUM_FEATURES_TO_VIZ, num_features)).indices.tolist()
    print("selected features:", top_feats[:10], "...")

    # for each feature and token type, keep top-N (activation, image_id)
    top = {ft: {TT_REG: [], TT_OUT: [], TT_NON: []} for ft in top_feats}

    def push(ft, ttype, act_val, img_id):
        arr = top[ft][ttype]
        arr.append((act_val, img_id))
        arr.sort(key=lambda x: x[0], reverse=True)
        if len(arr) > TOP_N:
            arr.pop()

    # second pass: collect top examples for selected features
    for p in tqdm(paths, desc="collect tops", dynamic_ncols=True):
        shard = torch.load(p, map_location="cpu")
        X = shard["vecs"].to(torch.float32)
        imgid = shard["image_id"].to(torch.int64)
        ttype = shard["token_type"].to(torch.int64)

        n = X.shape[0]
        for s in range(0, n, BATCH):
            xb = X[s:s+BATCH].to(DEVICE)
            xb = (xb - mu) / sigma
            a = F.relu(xb @ W[top_feats].t() + b[top_feats])  # (B, F_sel)

            img_b = imgid[s:s+BATCH]
            tt_b = ttype[s:s+BATCH]

            # update per feature per token type
            for j, ft in enumerate(top_feats):
                acts = a[:, j].detach().cpu()
                for i in range(acts.numel()):
                    tt = int(tt_b[i].item())
                    if tt not in (TT_REG, TT_OUT, TT_NON):
                        continue
                    push(ft, tt, float(acts[i].item()), int(img_b[i].item()))

    # write grids
    for ft in top_feats:
        for tt, name in [(TT_REG, "reg"), (TT_OUT, "outlier"), (TT_NON, "nonoutlier")]:
            img_ids = [img for _, img in top[ft][tt]]
            img_paths = [coco_path_from_id(img) for img in img_ids]
            out_path = OUT_DIR / f"feat_{ft:04d}_{name}.png"
            make_grid(img_paths, out_path)

    print("saved grids to:", OUT_DIR)

if __name__ == "__main__":
    main()
