#!/usr/bin/env python3
"""
run_all_high_norm_pipeline.py

Single-command runner for the full paper-aligned pipeline:

  1) Extract high-norm patch tokens (paper-defined outliers)
  2) Train SAE
  3) Visualize top SAE activations with heatmaps

Assumes these scripts exist in the same directory:
  - extract_imagenet1k_high_norm_tokens_layer6.py
  - train_sae_from_imagenet_high_norm_tokens_shards.py
  - viz_top_activations_imagenet_high_norm_tokens.py
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

PYTHON = sys.executable

EXTRACT = ROOT / "extract_imagenet1k_high_norm_tokens_layer6.py"
TRAIN   = ROOT / "train_sae_from_imagenet_high_norm_tokens_shards.py"
VIZ     = ROOT / "viz_top_activations_imagenet_high_norm_tokens.py"

RUN_DIR = "/lambda/nfs/neel/Research/runs/dinov2/imagenet1k_sae/layer6_high_norm_tokens"
NUM_FEATURES = "1024"
TOKEN_SET = "all"
TOP_N = "5"
NUM_FEATURES_TO_VIZ = "80"


def run(cmd):
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    # ---- Step 1: Extract tokens ----
    run([PYTHON, str(EXTRACT)])

    # ---- Step 2: Train SAE ----
    run([
        PYTHON, str(TRAIN),
        "--run_dir", RUN_DIR,
        "--token_set", TOKEN_SET,
        "--num_features", NUM_FEATURES,
    ])

    # ---- Step 3: Visualize ----
    run([
        PYTHON, str(VIZ),
        "--run_dir", RUN_DIR,
        "--sae_dir", f"sae_{NUM_FEATURES}_{TOKEN_SET}",
        "--top_n", TOP_N,
        "--num_features_to_viz", NUM_FEATURES_TO_VIZ,
    ])

    print("\n✅ Pipeline complete.")


if __name__ == "__main__":
    main()
