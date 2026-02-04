#!/usr/bin/env python3
"""
run_blip2_swap_only_main.py

Swap-only runner:
  - Generates predictions for BLIP-2 on:
      * COCO caption 5k
      * VQA v2 balanced 5k
  - Applies ONLY:
      swap queries = swap Q-Former last_hidden_state (query states) across images in-batch

Writes predictions under:
  runs/blip2/coco_caption_5k/preds_<tag>_swap_queries.jsonl
  runs/blip2/vqa_v2_balanced_5k/preds_<tag>_swap_queries.jsonl

Optional:
  --eval_metrics will compute metrics for THIS run only by calling eval_metrics_from_preds.py.

Robust to metadata image keys:
  image_path, path, image, image_file, file_name, filename

If the value is a filename (relative), it is resolved under:
  --coco_images_dir (default: <root>/datasets/coco/val2017)
  --vqa_images_dir  (default: <root>/datasets/vqav2/images/val2014)
"""

from __future__ import annotations

import argparse
import contextlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch
from PIL import Image
from transformers import AutoProcessor, Blip2ForConditionalGeneration


# -----------------------------
# Small utilities
# -----------------------------
def norm_id(x: Any) -> str:
    try:
        return str(int(x))
    except Exception:
        return str(x).strip()


# -----------------------------
# JSONL helpers
# -----------------------------
def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# -----------------------------
# Image helpers
# -----------------------------
def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def batch_iter(rows: List[Dict[str, Any]], batch_size: int):
    for i in range(0, len(rows), batch_size):
        yield rows[i : i + batch_size]


def get_image_id(row: Dict[str, Any]) -> str:
    v = row.get("image_id", row.get("img_id", row.get("id", "")))
    return norm_id(v)


def get_vqa_example_id(row: Dict[str, Any], fallback: str) -> str:
    v = row.get("question_id", row.get("example_id", row.get("id", fallback)))
    return norm_id(v)


def get_image_path(row: Dict[str, Any], images_dir: Path) -> str:
    for k in ("image_path", "path", "image", "image_file", "file_name", "filename"):
        if k in row and row[k]:
            p = Path(str(row[k]))

            if p.is_absolute():
                return str(p)

            cand = images_dir / p
            if cand.exists():
                return str(cand)

            parts = p.parts
            for t in range(1, min(6, len(parts)) + 1):
                cand2 = images_dir / Path(*parts[-t:])
                if cand2.exists():
                    return str(cand2)

            raise FileNotFoundError(
                f"Could not resolve image path. raw='{p}', tried under images_dir='{images_dir}'."
            )

    if "image_id" in row and row["image_id"] is not None:
        return str(images_dir / f"{int(row['image_id']):012d}.jpg")

    raise KeyError(f"metadata row missing image path. Keys: {list(row.keys())}")


# -----------------------------
# Swap intervention: swap Q-Former query states
# -----------------------------
def _swap_batch(x: torch.Tensor, seed: int, step: int) -> torch.Tensor:
    B = x.shape[0]
    if B <= 1:
        return x
    g = torch.Generator(device=x.device).manual_seed(seed + 1000003 * step)
    perm = torch.randperm(B, generator=g, device=x.device)
    return x[perm]


def _swap_qformer_last_hidden_state(out: Any, seed: int, step: int):
    if hasattr(out, "last_hidden_state") and torch.is_tensor(out.last_hidden_state):
        hs = out.last_hidden_state
        hs2 = _swap_batch(hs, seed=seed, step=step)
        try:
            out.last_hidden_state = hs2
        except Exception:
            object.__setattr__(out, "last_hidden_state", hs2)
        return out

    if isinstance(out, dict) and "last_hidden_state" in out and torch.is_tensor(out["last_hidden_state"]):
        out2 = dict(out)
        out2["last_hidden_state"] = _swap_batch(out2["last_hidden_state"], seed=seed, step=step)
        return out2

    return out


@contextlib.contextmanager
def swap_blip2_queries(model: Blip2ForConditionalGeneration, seed: int):
    if not hasattr(model, "qformer"):
        raise AttributeError("Model has no attribute `qformer`. Are you using a HF BLIP-2 model?")

    step = {"n": 0}

    def hook_fn(mod, inp, out):
        step["n"] += 1
        return _swap_qformer_last_hidden_state(out, seed=seed, step=step["n"])

    handle = model.qformer.register_forward_hook(hook_fn)
    try:
        yield
    finally:
        handle.remove()


# -----------------------------
# Generation helpers
# -----------------------------
def blip2_generate_captions(
    model: Blip2ForConditionalGeneration,
    processor: AutoProcessor,
    rows: List[Dict[str, Any]],
    device: str,
    batch_size: int,
    max_new_tokens: int,
    num_beams: int,
    coco_images_dir: Path,
) -> Dict[str, str]:
    preds: Dict[str, str] = {}
    num_batches = (len(rows) + batch_size - 1) // batch_size

    for bi, batch in enumerate(batch_iter(rows, batch_size), start=1):
        if bi == 1 or bi % 50 == 0 or bi == num_batches:
            print(f"[COCO] batch {bi}/{num_batches}", flush=True)

        images: List[Image.Image] = []
        ids: List[str] = []
        for r in batch:
            image_id = get_image_id(r)
            images.append(load_image(get_image_path(r, coco_images_dir)))
            ids.append(image_id)

        prompts = ["Describe the image."] * len(images)
        inputs = processor(images=images, text=prompts, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=num_beams)

        texts = processor.batch_decode(out_ids, skip_special_tokens=True)
        for image_id, text in zip(ids, texts):
            preds[image_id] = text.strip()

    return preds


def blip2_generate_vqa_answers(
    model: Blip2ForConditionalGeneration,
    processor: AutoProcessor,
    rows: List[Dict[str, Any]],
    device: str,
    batch_size: int,
    max_new_tokens: int,
    num_beams: int,
    vqa_images_dir: Path,
) -> Dict[str, str]:
    preds: Dict[str, str] = {}
    num_batches = (len(rows) + batch_size - 1) // batch_size

    for bi, batch in enumerate(batch_iter(rows, batch_size), start=1):
        if bi == 1 or bi % 50 == 0 or bi == num_batches:
            print(f"[VQA]  batch {bi}/{num_batches}", flush=True)

        images: List[Image.Image] = []
        prompts: List[str] = []
        ex_ids: List[str] = []

        for j, r in enumerate(batch):
            ex_id = get_vqa_example_id(r, fallback=f"{(bi-1)*batch_size + j}")
            q = r.get("question", r.get("q", None))
            if q is None:
                raise KeyError(f"VQA metadata row missing question/q. Keys: {list(r.keys())}")

            images.append(load_image(get_image_path(r, vqa_images_dir)))
            prompts.append(f"Question: {q} Answer:")
            ex_ids.append(ex_id)

        inputs = processor(images=images, text=prompts, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=num_beams)

        texts = processor.batch_decode(out_ids, skip_special_tokens=True)
        for ex_id, gen in zip(ex_ids, texts):
            gen = gen.strip()
            m = re.search(r"answer:\s*(.*)$", gen, flags=re.IGNORECASE)
            if m:
                gen = m.group(1).strip()
            preds[ex_id] = gen

    return preds


# -----------------------------
# Model loading helpers
# -----------------------------
def load_processor(model_id: str) -> AutoProcessor:
    try:
        return AutoProcessor.from_pretrained(model_id, use_fast=True)
    except Exception:
        return AutoProcessor.from_pretrained(model_id, use_fast=False)


def load_model(model_id: str, dtype: torch.dtype) -> Blip2ForConditionalGeneration:
    try:
        return Blip2ForConditionalGeneration.from_pretrained(model_id, dtype=dtype)
    except Exception:
        return Blip2ForConditionalGeneration.from_pretrained(model_id, torch_dtype=dtype)


# -----------------------------
# Metrics invocation
# -----------------------------
def run_metrics(root: Path, name: str, do_coco: bool, do_vqa: bool) -> None:
    metrics_py = root / "eval_metrics_from_preds.py"
    if not metrics_py.exists():
        raise FileNotFoundError(f"Missing metrics script: {metrics_py}")

    cmd = [sys.executable, str(metrics_py), "--root", str(root), "--name", name]
    if do_coco:
        cmd.append("--coco")
    if do_vqa:
        cmd.append("--vqa")

    print("[metrics] running:", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="/lambda/nfs/neel/Research")
    ap.add_argument("--model_id", type=str, default="Salesforce/blip2-flan-t5-xl")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_beams", type=int, default=3)
    ap.add_argument("--caption_max_new_tokens", type=int, default=30)
    ap.add_argument("--vqa_max_new_tokens", type=int, default=10)

    ap.add_argument("--run_coco", action="store_true")
    ap.add_argument("--run_vqa", action="store_true")
    ap.add_argument("--tag", type=str, default="blip2_flan_t5_xl")

    ap.add_argument("--coco_images_dir", type=str, default=None)
    ap.add_argument("--vqa_images_dir", type=str, default=None)

    ap.add_argument("--swap_seed", type=int, default=123)
    ap.add_argument("--eval_metrics", action="store_true")
    args = ap.parse_args()

    if not args.run_coco and not args.run_vqa:
        raise SystemExit("Pass at least one of --run_coco or --run_vqa")
    if args.batch_size <= 1:
        raise SystemExit("swap requires --batch_size > 1 to have any effect.")

    root = Path(args.root)
    subsets_dir = root / "subsets"
    runs_dir = root / "runs" / "blip2"

    coco_meta = subsets_dir / "coco_caption_5k" / "metadata.jsonl"
    vqa_meta = subsets_dir / "vqa_v2_balanced_5k" / "metadata.jsonl"

    if args.run_coco and not coco_meta.exists():
        raise SystemExit(f"Missing: {coco_meta}")
    if args.run_vqa and not vqa_meta.exists():
        raise SystemExit(f"Missing: {vqa_meta}")

    coco_images_dir = Path(args.coco_images_dir) if args.coco_images_dir else (root / "datasets" / "coco" / "val2017")
    vqa_images_dir = Path(args.vqa_images_dir) if args.vqa_images_dir else (root / "datasets" / "vqav2" / "images" / "val2014")

    if args.run_coco and not coco_images_dir.exists():
        raise SystemExit(f"Missing COCO images dir: {coco_images_dir}")
    if args.run_vqa and not vqa_images_dir.exists():
        raise SystemExit(f"Missing VQA images dir: {vqa_images_dir}")

    torch.manual_seed(args.seed)

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]

    print("Loading processor + model...", flush=True)
    processor = load_processor(args.model_id)
    model = load_model(args.model_id, dtype=dtype).to(args.device)
    model.eval()
    print("Loaded.", flush=True)

    coco_rows = read_jsonl(coco_meta) if args.run_coco else []
    vqa_rows = read_jsonl(vqa_meta) if args.run_vqa else []
    print(f"COCO rows: {len(coco_rows)} | VQA rows: {len(vqa_rows)}", flush=True)

    name = f"{args.tag}_swap_queries"
    print(f"=== Running {name} (swap only) ===", flush=True)

    with swap_blip2_queries(model, seed=args.swap_seed):
        if args.run_coco:
            coco_preds = blip2_generate_captions(
                model=model,
                processor=processor,
                rows=coco_rows,
                device=args.device,
                batch_size=args.batch_size,
                max_new_tokens=args.caption_max_new_tokens,
                num_beams=args.num_beams,
                coco_images_dir=coco_images_dir,
            )
            coco_pred_path = runs_dir / "coco_caption_5k" / f"preds_{name}.jsonl"
            write_jsonl(
                coco_pred_path,
                ({"image_id": get_image_id(r), "caption": coco_preds.get(get_image_id(r), "")} for r in coco_rows),
            )
            print(f"[wrote] {coco_pred_path}", flush=True)

        if args.run_vqa:
            vqa_preds = blip2_generate_vqa_answers(
                model=model,
                processor=processor,
                rows=vqa_rows,
                device=args.device,
                batch_size=args.batch_size,
                max_new_tokens=args.vqa_max_new_tokens,
                num_beams=args.num_beams,
                vqa_images_dir=vqa_images_dir,
            )
            vqa_pred_path = runs_dir / "vqa_v2_balanced_5k" / f"preds_{name}.jsonl"
            write_jsonl(
                vqa_pred_path,
                (
                    {
                        "example_id": get_vqa_example_id(r, fallback=""),
                        "answer": vqa_preds.get(get_vqa_example_id(r, fallback=""), ""),
                    }
                    for r in vqa_rows
                ),
            )
            print(f"[wrote] {vqa_pred_path}", flush=True)

    if args.eval_metrics:
        run_metrics(root, name, do_coco=args.run_coco, do_vqa=args.run_vqa)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()

