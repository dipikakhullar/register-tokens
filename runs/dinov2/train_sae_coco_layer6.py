import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# --- paths ---
SHARDS_DIR = Path("/home/ubuntu/neel/Research/runs/dinov2/coco_train2017_sae/layer6_block_output_k4/shards")
OUT_DIR = SHARDS_DIR.parent / "sae_1024"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- SAE config ---
NUM_FEATURES = 1024
BATCH_SIZE = 2048
EPOCHS = 10
LR = 1e-3
L1_LAMBDA = 1e-3  # sparsity weight (tune later)
WEIGHT_DECAY = 0.0
SEED = 0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32

torch.manual_seed(SEED)

def load_all_shards(shards_dir: Path):
    shard_paths = sorted(shards_dir.glob("shard_*.pt"))
    if not shard_paths:
        raise FileNotFoundError(f"No shards found in {shards_dir}")

    vecs_all = []
    imgid_all = []
    ttype_all = []
    tpos_all = []

    for p in tqdm(shard_paths, desc="load shards", dynamic_ncols=True):
        d = torch.load(p, map_location="cpu")
        vecs_all.append(d["vecs"].to(torch.float32))  # train in fp32
        imgid_all.append(d["image_id"].to(torch.int64))
        ttype_all.append(d["token_type"].to(torch.int8))
        tpos_all.append(d["token_pos"].to(torch.int16))

    vecs = torch.cat(vecs_all, 0)
    imgid = torch.cat(imgid_all, 0)
    ttype = torch.cat(ttype_all, 0)
    tpos = torch.cat(tpos_all, 0)
    return vecs, imgid, ttype, tpos

class SAE(nn.Module):
    def __init__(self, d_in: int, n_feat: int):
        super().__init__()
        self.enc = nn.Linear(d_in, n_feat, bias=True)
        self.dec = nn.Linear(n_feat, d_in, bias=True)

        # small init
        nn.init.normal_(self.enc.weight, std=0.02)
        nn.init.zeros_(self.enc.bias)
        nn.init.normal_(self.dec.weight, std=0.02)
        nn.init.zeros_(self.dec.bias)

    def forward(self, x):
        a = F.relu(self.enc(x))
        x_hat = self.dec(a)
        return x_hat, a

# ... keep your imports and config as-is ...

def main():
    X, image_id, token_type, token_pos = load_all_shards(SHARDS_DIR)
    n, d = X.shape
    print("loaded vecs:", X.shape, "D=", d)

    # normalize (helps SAE stability)
    mu = X.mean(dim=0, keepdim=True)
    sigma = X.std(dim=0, keepdim=True).clamp_min(1e-6)
    Xn = (X - mu) / sigma

    sae = SAE(d_in=d, n_feat=NUM_FEATURES).to(DEVICE)
    opt = torch.optim.AdamW(sae.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    idx = torch.arange(n)

    best_loss = float("inf")
    metrics_path = OUT_DIR / "train_metrics.jsonl"  # <--- reuse a single path

    for ep in range(EPOCHS):
        perm = idx[torch.randperm(n)]
        tot_recon = 0.0
        tot_l1 = 0.0
        tot = 0.0
        count = 0

        sae.train()
        for s in tqdm(range(0, n, BATCH_SIZE), desc=f"train ep {ep+1}/{EPOCHS}", dynamic_ncols=True):
            b = perm[s:s+BATCH_SIZE]
            xb = Xn[b].to(DEVICE, dtype=DTYPE)

            opt.zero_grad(set_to_none=True)
            x_hat, a = sae(xb)

            recon = F.mse_loss(x_hat, xb)
            l1 = a.abs().mean()
            loss = recon + L1_LAMBDA * l1

            loss.backward()
            opt.step()

            bs = xb.size(0)
            tot_recon += recon.item() * bs
            tot_l1 += l1.item() * bs
            tot += loss.item() * bs
            count += bs

        avg_recon = tot_recon / count
        avg_l1 = tot_l1 / count
        avg_loss = tot / count

        # quick sparsity stat (sample random tokens instead of first chunk)
        sae.eval()
        with torch.inference_mode():
            m = min(8192, n)
            j = torch.randint(0, n, (m,))
            xb = Xn[j].to(DEVICE, dtype=DTYPE)
            _, a = sae(xb)
            active = (a > 1e-6).float().sum(dim=1).mean().item()

        metrics = {
            "epoch": ep + 1,
            "avg_loss": avg_loss,
            "avg_recon_mse": avg_recon,
            "avg_l1": avg_l1,
            "avg_active_features_per_token": active,
            "n_tokens": n,
            "d_model": d,
            "num_features": NUM_FEATURES,
            "l1_lambda": L1_LAMBDA,
            "batch_size": BATCH_SIZE,
            "model_path": str(OUT_DIR / "sae.pt"),
        }

        # FIX: close the file properly
        with metrics_path.open("a") as f:
            f.write(json.dumps(metrics) + "\n")

        print(metrics)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                {
                    "state_dict": sae.state_dict(),
                    "mu": mu,
                    "sigma": sigma,
                    "config": metrics,
                },
                OUT_DIR / "sae.pt",
            )

    print("saved:", OUT_DIR / "sae.pt")

if __name__ == "__main__":
    main()
