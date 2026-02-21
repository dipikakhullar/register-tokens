#!/usr/bin/env python3
"""
smoketest_top5_sae_feature_probe.py

Lightweight smoke test wrapper for top5_images_per_feature_first10.py.

What it does
- Creates a temporary subset extract_dir with the first N shard files copied in
- Runs the top5 script on only those shards
- Enables overlays (or lets you disable)
- Prints a short summary of expected outputs
- Optionally cleans up the temporary subset folder

Use this to verify the pipeline works end-to-end before running all shards.

Example:
python3 smoketest_top5_sae_feature_probe.py \
  --full_extract_dir /lambda/nfs/neel/Research/runs/dinov2/imagenet1k_base_reg4_hook06_extract \
  --sae_run_dir /lambda/nfs/neel/Research/runs/sae/imagenet1k_base_reg4_hook06_overcomplete_topk4096_k20_run1 \
  --output_dir /lambda/nfs/neel/Research/runs/sae_top5_debug_first10/smoketest_k20_run1 \
  --top5_script /lambda/nfs/neel/Research/imagenet/constant/top5_images_per_feature_first10.py \
  --num_shards 5 \
  --feature_start 0 \
  --num_features 10 \
  --top_k_images 5 \
  --save_png
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def list_shards(extract_dir: Path) -> List[Path]:
    return sorted(extract_dir.glob("extract_hook*_shard*.pt"))


def copy_subset_shards(src_dir: Path, dst_dir: Path, num_shards: int) -> List[Path]:
    shards = list_shards(src_dir)
    if not shards:
        raise FileNotFoundError(f"No shard files found in {src_dir}")
    chosen = shards[:num_shards]
    safe_mkdir(dst_dir)

    print(f"[SMOKETEST] Copying {len(chosen)} shard(s) to subset dir:")
    for sp in chosen:
        dst = dst_dir / sp.name
        print(f"  - {sp.name}")
        shutil.copy2(sp, dst)

    return [dst_dir / sp.name for sp in chosen]


def build_top5_command(
    python_exe: str,
    top5_script: Path,
    subset_extract_dir: Path,
    sae_run_dir: Path,
    output_dir: Path,
    feature_start: int,
    num_features: int,
    top_k_images: int,
    agg_mode: str,
    token_batch_size: int,
    ckpt_name: str,
    save_png: bool,
    save_original: bool,
    save_activation_overlay: bool,
    save_attention_overlay: bool,
    device: str,
) -> List[str]:
    cmd = [
        python_exe,
        str(top5_script),
        "--extract_dir", str(subset_extract_dir),
        "--sae_run_dir", str(sae_run_dir),
        "--output_dir", str(output_dir),
        "--feature_start", str(feature_start),
        "--num_features", str(num_features),
        "--top_k_images", str(top_k_images),
        "--agg_mode", agg_mode,
        "--token_batch_size", str(token_batch_size),
        "--ckpt_name", ckpt_name,
        "--device", device,
    ]

    if save_png:
        cmd.append("--save_png")
    else:
        if save_original:
            cmd.append("--save_original")
        if save_activation_overlay:
            cmd.append("--save_activation_overlay")
        if save_attention_overlay:
            cmd.append("--save_attention_overlay")

    return cmd


def summarize_outputs(output_dir: Path, feature_start: int, num_features: int) -> None:
    print("\n[SMOKETEST] Output summary")
    print(f"  output_dir: {output_dir}")
    run_meta = output_dir / "run_meta.json"
    print(f"  run_meta.json exists: {run_meta.exists()}")

    checked = 0
    for fid in range(feature_start, feature_start + num_features):
        feat_dir = output_dir / f"feature_{fid:04d}"
        if not feat_dir.exists():
            continue
        checked += 1
        jsons = sorted(feat_dir.glob("top5_*.json"))
        pngs = sorted(feat_dir.glob("*.png"))
        print(f"  feature_{fid:04d}: {len(jsons)} json(s), {len(pngs)} png(s)")
        if checked >= 3:
            break

    if checked == 0:
        print("  No feature directories found yet.")


def main() -> None:
    ap = argparse.ArgumentParser()

    # Required paths
    ap.add_argument("--full_extract_dir", type=str, required=True, help="Directory with all extraction shards")
    ap.add_argument("--sae_run_dir", type=str, required=True, help="SAE run dir containing norm_stats.pt and checkpoints/")
    ap.add_argument("--output_dir", type=str, required=True, help="Where smoketest outputs will be written")
    ap.add_argument("--top5_script", type=str, required=True, help="Path to top5_images_per_feature_first10.py")

    # Smoke test controls
    ap.add_argument("--num_shards", type=int, default=5, help="How many shards to copy for subset run")
    ap.add_argument("--subset_dir", type=str, default="", help="Optional explicit subset dir; defaults next to output_dir")
    ap.add_argument("--cleanup_subset", action="store_true", help="Delete subset_dir after run")

    # Forwarded top5 args
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--feature_start", type=int, default=0)
    ap.add_argument("--num_features", type=int, default=10)
    ap.add_argument("--top_k_images", type=int, default=5)
    ap.add_argument("--agg_mode", type=str, choices=["mean", "max"], default="mean")
    ap.add_argument("--token_batch_size", type=int, default=65536)
    ap.add_argument("--ckpt_name", type=str, default="best.pt")

    # Overlay flags
    ap.add_argument("--save_png", action="store_true", help="Convenience: save originals + both overlays")
    ap.add_argument("--save_original", action="store_true")
    ap.add_argument("--save_activation_overlay", action="store_true")
    ap.add_argument("--save_attention_overlay", action="store_true")

    # Execution behavior
    ap.add_argument("--python_exe", type=str, default=sys.executable, help="Python executable to use for child run")
    ap.add_argument("--dry_run", action="store_true", help="Print commands only, do not run")

    args = ap.parse_args()

    full_extract_dir = Path(args.full_extract_dir)
    sae_run_dir = Path(args.sae_run_dir)
    output_dir = Path(args.output_dir)
    top5_script = Path(args.top5_script)

    if not full_extract_dir.exists():
        raise FileNotFoundError(f"--full_extract_dir not found: {full_extract_dir}")
    if not sae_run_dir.exists():
        raise FileNotFoundError(f"--sae_run_dir not found: {sae_run_dir}")
    if not top5_script.exists():
        raise FileNotFoundError(f"--top5_script not found: {top5_script}")

    # Quick validation of SAE run dir structure
    norm_stats = sae_run_dir / "norm_stats.pt"
    ckpt = sae_run_dir / "checkpoints" / args.ckpt_name
    if not norm_stats.exists():
        raise FileNotFoundError(f"Missing norm_stats.pt in SAE run dir: {norm_stats}")
    if not ckpt.exists():
        raise FileNotFoundError(f"Missing checkpoint in SAE run dir: {ckpt}")

    # Build subset dir path
    if args.subset_dir:
        subset_dir = Path(args.subset_dir)
    else:
        subset_dir = output_dir.parent / f"{output_dir.name}__subset_shards_{args.num_shards}"

    print("[SMOKETEST] Configuration")
    print(f"  full_extract_dir: {full_extract_dir}")
    print(f"  sae_run_dir:      {sae_run_dir}")
    print(f"  top5_script:      {top5_script}")
    print(f"  output_dir:       {output_dir}")
    print(f"  subset_dir:       {subset_dir}")
    print(f"  num_shards:       {args.num_shards}")
    print(f"  features:         {args.feature_start}..{args.feature_start + args.num_features - 1}")
    print(f"  top_k_images:     {args.top_k_images}")

    if args.num_shards <= 0:
        raise ValueError("--num_shards must be >= 1")

    if not args.dry_run:
        if subset_dir.exists():
            print(f"[SMOKETEST] subset_dir already exists, removing: {subset_dir}")
            shutil.rmtree(subset_dir)
        copy_subset_shards(full_extract_dir, subset_dir, args.num_shards)

    safe_mkdir(output_dir)

    cmd = build_top5_command(
        python_exe=args.python_exe,
        top5_script=top5_script,
        subset_extract_dir=subset_dir,
        sae_run_dir=sae_run_dir,
        output_dir=output_dir,
        feature_start=args.feature_start,
        num_features=args.num_features,
        top_k_images=args.top_k_images,
        agg_mode=args.agg_mode,
        token_batch_size=args.token_batch_size,
        ckpt_name=args.ckpt_name,
        save_png=args.save_png,
        save_original=args.save_original,
        save_activation_overlay=args.save_activation_overlay,
        save_attention_overlay=args.save_attention_overlay,
        device=args.device,
    )

    print("\n[SMOKETEST] Running top5 script command:")
    print("  " + " ".join(cmd))

    if args.dry_run:
        print("[SMOKETEST] dry_run enabled. Exiting.")
        return

    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        print(f"\n[SMOKETEST] top5 script failed with return code {proc.returncode}")
        if args.cleanup_subset and subset_dir.exists():
            print(f"[SMOKETEST] cleanup_subset enabled, deleting {subset_dir}")
            shutil.rmtree(subset_dir)
        sys.exit(proc.returncode)

    summarize_outputs(output_dir, args.feature_start, args.num_features)

    if args.cleanup_subset and subset_dir.exists():
        print(f"\n[SMOKETEST] cleanup_subset enabled, deleting {subset_dir}")
        shutil.rmtree(subset_dir)

    print("\n[SMOKETEST] Done.")


if __name__ == "__main__":
    main()