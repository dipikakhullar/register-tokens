#!/usr/bin/env python3
"""
eval_metrics_from_preds.py

Metrics tester:
  - Reads subset metadata.jsonl + prediction jsonl(s)
  - Computes:
      * COCO caption metrics: BLEU-1..4, METEOR (if available), ROUGE_L, CIDEr
      * VQA soft accuracy (min(count/3, 1) over up to 10 answers)

Writes:
  runs/blip2/coco_caption_5k/eval/metrics_<name>.json
  runs/blip2/vqa_v2_balanced_5k/eval/metrics_<name>.json

Assumes prediction formats (tolerant):
  COCO preds jsonl lines: {"image_id": "...", "caption": "..."} (also accepts "prediction"/"pred")
  VQA preds  jsonl lines: {"example_id": "...", "answer": "..."} (also accepts "question_id"/"id" and "pred"/"prediction")

Changes vs your version:
  - METEOR failure writes METEOR=None (valid JSON), plus METEOR_error
  - Removed dependency/check on local ./coco-caption directory
  - Optional --skip_meteor flag to avoid slow/buggy METEOR
  - Safer METEOR cleanup to reduce noisy destructor warnings
  - JSON writer sanitizes NaN/Inf and enforces allow_nan=False
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple


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


def _sanitize_for_json(x: Any) -> Any:
    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            return None
        return x
    if isinstance(x, dict):
        return {k: _sanitize_for_json(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_sanitize_for_json(v) for v in x]
    if isinstance(x, tuple):
        return [_sanitize_for_json(v) for v in x]
    return x


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_obj = _sanitize_for_json(obj)
    with path.open("w", encoding="utf-8") as f:
        json.dump(safe_obj, f, indent=2, allow_nan=False)


# -----------------------------
# VQA normalization + scoring
# -----------------------------
_ARTICLES = {"a", "an", "the"}


def _strip_punct(s: str) -> str:
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def normalize_answer(ans: str) -> str:
    ans = ans.lower().strip()
    ans = _strip_punct(ans)
    toks = [t for t in ans.split() if t not in _ARTICLES]
    return " ".join(toks)


def vqa_soft_score(pred: str, answers: List[str]) -> float:
    if not answers:
        return 0.0
    p = normalize_answer(pred)
    counts = 0
    for a in answers:
        if normalize_answer(a) == p:
            counts += 1
    return min(counts / 3.0, 1.0)


# -----------------------------
# COCO caption metrics via (Python 3) pycocoevalcap
# -----------------------------
def compute_coco_caption_metrics(
    refs: Dict[str, List[str]],
    hyps: Dict[str, str],
    skip_meteor: bool = False,
) -> Dict[str, Any]:
    # Import from installed Python-3 package (NOT local coco-caption repo)
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.cider.cider import Cider

    gts = refs
    res = {k: [v] for k, v in hyps.items()}

    metrics: Dict[str, Any] = {}

    bleu_scorer = Bleu(4)
    bleu_scores, _ = bleu_scorer.compute_score(gts, res)
    metrics["BLEU_1"] = float(bleu_scores[0])
    metrics["BLEU_2"] = float(bleu_scores[1])
    metrics["BLEU_3"] = float(bleu_scores[2])
    metrics["BLEU_4"] = float(bleu_scores[3])

    # METEOR (optional)
    if not skip_meteor:
        try:
            from pycocoevalcap.meteor.meteor import Meteor

            meteor_scorer = Meteor()
            try:
                meteor_score, _ = meteor_scorer.compute_score(gts, res)
                metrics["METEOR"] = float(meteor_score)
            finally:
                # Some versions have buggy __del__. Close if available.
                if hasattr(meteor_scorer, "close"):
                    try:
                        meteor_scorer.close()
                    except Exception:
                        pass
        except Exception as e:
            metrics["METEOR"] = None
            metrics["METEOR_error"] = str(e)
    else:
        metrics["METEOR"] = None
        metrics["METEOR_error"] = "skipped"

    rouge_scorer = Rouge()
    rouge_score, _ = rouge_scorer.compute_score(gts, res)
    metrics["ROUGE_L"] = float(rouge_score)

    cider_scorer = Cider()
    cider_score, _ = cider_scorer.compute_score(gts, res)
    metrics["CIDEr"] = float(cider_score)

    return metrics


# -----------------------------
# Parsing preds (tolerant)
# -----------------------------
def _get_pred_key(d: Dict[str, Any], keys: Tuple[str, ...]) -> Any:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def load_coco_preds(pred_rows: List[Dict[str, Any]]) -> Dict[str, str]:
    hyps: Dict[str, str] = {}
    for r in pred_rows:
        image_id = _get_pred_key(r, ("image_id", "img_id", "id"))
        cap = _get_pred_key(r, ("caption", "prediction", "pred", "text"))
        if image_id is None:
            raise KeyError(f"COCO pred row missing image_id/img_id/id. Keys: {list(r.keys())}")
        if cap is None:
            cap = ""
        hyps[norm_id(image_id)] = str(cap).strip()
    return hyps


def load_vqa_preds(pred_rows: List[Dict[str, Any]]) -> Dict[str, str]:
    preds: Dict[str, str] = {}
    for r in pred_rows:
        ex_id = _get_pred_key(r, ("example_id", "question_id", "id"))
        ans = _get_pred_key(r, ("answer", "prediction", "pred", "text"))
        if ex_id is None:
            raise KeyError(f"VQA pred row missing example_id/question_id/id. Keys: {list(r.keys())}")
        if ans is None:
            ans = ""
        preds[norm_id(ex_id)] = str(ans).strip()
    return preds


# -----------------------------
# Evaluate COCO subset
# -----------------------------
def eval_coco_from_files(root: Path, name: str, skip_meteor: bool) -> Path:
    subsets_dir = root / "subsets"
    runs_dir = root / "runs" / "blip2"

    meta_path = subsets_dir / "coco_caption_5k" / "metadata.jsonl"
    pred_path = runs_dir / "coco_caption_5k" / f"preds_{name}.jsonl"
    out_path = runs_dir / "coco_caption_5k" / "eval" / f"metrics_{name}.json"

    if not meta_path.exists():
        raise FileNotFoundError(f"Missing: {meta_path}")
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing: {pred_path}")

    meta = read_jsonl(meta_path)
    pred_rows = read_jsonl(pred_path)

    refs: Dict[str, List[str]] = {}
    for r in meta:
        image_id = norm_id(r.get("image_id", r.get("img_id", r.get("id", ""))))
        caps = r.get("captions", r.get("references", r.get("caption", None)))
        if isinstance(caps, list):
            refs[image_id] = [str(c) for c in caps]
        elif isinstance(caps, str):
            refs[image_id] = [caps]
        else:
            raise KeyError(f"COCO metadata row missing captions/references/caption. Keys: {list(r.keys())}")

    hyps_raw = load_coco_preds(pred_rows)

    missing = [k for k in refs.keys() if k not in hyps_raw]
    hyps = {k: (hyps_raw.get(k) if hyps_raw.get(k, "").strip() else ".") for k in refs.keys()}

    metrics = compute_coco_caption_metrics(refs, hyps, skip_meteor=skip_meteor)
    metrics["num_refs"] = len(refs)
    metrics["num_hyps"] = len(hyps_raw)
    metrics["missing_hyps"] = len(missing)

    write_json(out_path, metrics)
    return out_path


# -----------------------------
# Evaluate VQA subset
# -----------------------------
def eval_vqa_from_files(root: Path, name: str) -> Path:
    subsets_dir = root / "subsets"
    runs_dir = root / "runs" / "blip2"

    meta_path = subsets_dir / "vqa_v2_balanced_5k" / "metadata.jsonl"
    pred_path = runs_dir / "vqa_v2_balanced_5k" / f"preds_{name}.jsonl"
    out_path = runs_dir / "vqa_v2_balanced_5k" / "eval" / f"metrics_{name}.json"

    if not meta_path.exists():
        raise FileNotFoundError(f"Missing: {meta_path}")
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing: {pred_path}")

    meta = read_jsonl(meta_path)
    pred_rows = read_jsonl(pred_path)
    preds = load_vqa_preds(pred_rows)

    scores: List[float] = []
    missing = 0

    for r in meta:
        ex_id = norm_id(r.get("question_id", r.get("example_id", r.get("id", ""))))
        if not ex_id:
            raise KeyError(f"VQA metadata row missing question_id/example_id/id. Keys: {list(r.keys())}")

        if isinstance(r.get("answers", None), list):
            gt = [str(a) for a in r["answers"]]
        elif r.get("answer", None) is not None:
            gt = [str(r["answer"])]
        else:
            gt = []

        if ex_id not in preds:
            missing += 1

        pred = preds.get(ex_id, "")
        scores.append(vqa_soft_score(pred, gt))

    soft_acc = float(sum(scores) / max(len(scores), 1))
    metrics = {
        "soft_acc": soft_acc,
        "num_questions": len(meta),
        "num_preds": len(preds),
        "missing_preds": missing,
    }

    write_json(out_path, metrics)
    return out_path


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="/lambda/nfs/neel/Research")
    ap.add_argument("--name", type=str, required=True, help="Pred filename stem, e.g. blip2_flan_t5_xl_zero_queries")
    ap.add_argument("--coco", action="store_true")
    ap.add_argument("--vqa", action="store_true")
    ap.add_argument("--skip_meteor", action="store_true", help="Skip METEOR (fast + avoids Java issues)")
    args = ap.parse_args()

    if not args.coco and not args.vqa:
        raise SystemExit("Pass at least one of --coco or --vqa")

    root = Path(args.root)

    if args.coco:
        out = eval_coco_from_files(root, args.name, skip_meteor=args.skip_meteor)
        print(f"[COCO metrics] wrote {out}")

    if args.vqa:
        out = eval_vqa_from_files(root, args.name)
        print(f"[VQA metrics]  wrote {out}")

    print("Done.")


if __name__ == "__main__":
    main()
