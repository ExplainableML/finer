# ash_eval/benchmarks/pope_repope.py

from pathlib import Path
import json

import pandas as pd
from PIL import Image
from tqdm import tqdm

from prompts import build_messages, parse_yesno


BENCHMARKS = {
    "pope": "pope",
    "repope": "repope",
}

SUBSETS = ["random", "popular", "adversarial"]


def find_image_path(img_root: Path, filename: str) -> Path:
    """
    Find image under:
      1. img_root / filename
      2. img_root / "val2014" / filename
    """
    p = img_root / filename
    if p.exists():
        return p

    p2 = img_root / "val2014" / filename
    if p2.exists():
        return p2

    raise FileNotFoundError(
        f"Image not found: {filename} under {img_root} or {img_root / 'val2014'}"
    )


def pct(n: int, d: int) -> str:
    return f"{(n / d):.3%}  ({n}/{d})" if d else "n/a"


def compute_metrics(records: list[dict]) -> dict:
    n_total = len(records)

    n_pos = sum(r["gt_bin"] == 1 for r in records)
    n_neg = sum(r["gt_bin"] == 0 for r in records)

    acc_total = sum(r["correct"] for r in records)

    tp = sum(r["gt_bin"] == 1 and r["pred_bin"] == 1 for r in records)
    tn = sum(r["gt_bin"] == 0 and r["pred_bin"] == 0 for r in records)

    # Preserve your original logic:
    # unknown is counted as an error.
    fn = n_pos - tp
    fp = n_neg - tn

    n_unknown = sum(r["pred_bin"] == -1 for r in records)
    n_pred_yes = sum(r["pred_bin"] == 1 for r in records)
    n_pred_no = sum(r["pred_bin"] == 0 for r in records)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / n_pos if n_pos else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "n_total": n_total,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "n_pred_yes": n_pred_yes,
        "n_pred_no": n_pred_no,
        "n_unknown": n_unknown,

        "acc_total": acc_total,
        "acc_pos": tp,
        "acc_neg": tn,

        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,

        "accuracy": acc_total / n_total if n_total else 0.0,
        "tp_rate": tp / n_pos if n_pos else 0.0,
        "tn_rate": tn / n_neg if n_neg else 0.0,
        "fp_rate": fp / n_neg if n_neg else 0.0,
        "fn_rate": fn / n_pos if n_pos else 0.0,

        # useful for POPE-style reporting
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "yes_ratio": n_pred_yes / n_total if n_total else 0.0,
        "unknown_ratio": n_unknown / n_total if n_total else 0.0,
    }


def evaluate_jsonl_file(
    model_adapter,
    json_path: Path,
    img_root: Path,
    max_new_tokens: int,
) -> tuple[list[dict], dict]:
    """
    Evaluate one POPE/RePOPE JSONL file.

    Each line is expected to contain:
        {
            "question_id": int,
            "image": str,
            "text": str,
            "label": "yes" | "no"
        }
    """
    records = []

    with json_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in tqdm(lines, desc=f"inference: {json_path.name}"):
        ex = json.loads(line)

        qid = ex.get("question_id")
        img_name = ex["image"]
        question = str(ex["text"])
        gt_ans = str(ex["label"]).strip().lower()

        if gt_ans not in {"yes", "no"}:
            raise ValueError(
                f"Unexpected label {gt_ans!r} in {json_path.name}, qid={qid}"
            )

        gt_bin = 1 if gt_ans == "yes" else 0

        img_path = find_image_path(img_root, img_name)
        pil_img = Image.open(img_path).convert("RGB")

        messages = build_messages(pil_img, question)

        raw_text = model_adapter.generate_text(
            messages=messages,
            max_new_tokens=max_new_tokens,
        )

        pred_bin = parse_yesno(raw_text)
        pred_ans = "yes" if pred_bin == 1 else "no" if pred_bin == 0 else "unknown"

        correct = int(pred_bin == gt_bin)

        records.append({
            "question_id": qid,
            "image": img_name,
            "question": question,
            "answer_gt": gt_ans,
            "answer_pred": pred_ans,
            "raw_text": raw_text,
            "correct": correct,
            "image_path": str(img_path),

            # internal fields useful for metrics, can be dropped before saving
            "gt_bin": gt_bin,
            "pred_bin": pred_bin,
        })

    metrics = compute_metrics(records)
    return records, metrics


def print_subset_report(benchmark: str, subset: str, metrics: dict):
    print("\n------------------ Results ------------------")
    print(f"{benchmark.upper()} / {subset}")
    print(f"Total              : {metrics['n_total']}")
    print(f"Accuracy           : {pct(metrics['acc_total'], metrics['n_total'])}")
    print(f"TP rate (GT=yes)   : {pct(metrics['acc_pos'], metrics['n_pos'])}")
    print(f"TN rate (GT=no)    : {pct(metrics['acc_neg'], metrics['n_neg'])}")
    print(f"FP rate            : {pct(metrics['FP'], metrics['n_neg'])}")
    print(f"FN rate            : {pct(metrics['FN'], metrics['n_pos'])}")
    print(f"Precision          : {metrics['precision']:.3%}")
    print(f"Recall             : {metrics['recall']:.3%}")
    print(f"F1                 : {metrics['f1']:.3%}")
    print(f"Yes ratio          : {metrics['yes_ratio']:.3%}")
    print(f"Unknown ratio      : {metrics['unknown_ratio']:.3%}")
    print(f"TP/TN/FP/FN        : {metrics['TP']}/{metrics['TN']}/{metrics['FP']}/{metrics['FN']}")
    print("---------------------------------------------")


def run_pope_repope(
    model_adapter,
    ann_dir: Path,
    img_root: Path,
    out_dir: Path,
    max_new_tokens: int,
    benchmarks: list[str] | None = None,
    subsets: list[str] | None = None,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    if benchmarks is None:
        benchmarks = list(BENCHMARKS.keys())

    if subsets is None:
        subsets = SUBSETS

    summary_rows = []

    for benchmark in benchmarks:
        if benchmark not in BENCHMARKS:
            raise ValueError(
                f"Unknown benchmark: {benchmark}. Available: {list(BENCHMARKS.keys())}"
            )

        prefix = BENCHMARKS[benchmark]

        for subset in subsets:
            json_path = ann_dir / f"coco_{prefix}_{subset}.json"

            if not json_path.exists():
                print(f"[WARN] Missing file: {json_path} — skipping.")
                continue

            print(f"\n=== Running {benchmark.upper()} / {subset} ===")

            records, metrics = evaluate_jsonl_file(
                model_adapter=model_adapter,
                json_path=json_path,
                img_root=img_root,
                max_new_tokens=max_new_tokens,
            )

            # remove internal metric fields before saving
            save_records = []
            for r in records:
                r = dict(r)
                r.pop("gt_bin", None)
                r.pop("pred_bin", None)
                save_records.append(r)

            csv_path = out_dir / f"{benchmark}_{subset}.csv"
            pd.DataFrame.from_records(save_records).to_csv(csv_path, index=False)

            print_subset_report(benchmark, subset, metrics)
            print(f"CSV → {csv_path}")

            summary_rows.append({
                "benchmark": benchmark.upper(),
                "subset": subset,

                "n_total": metrics["n_total"],
                "n_pos": metrics["n_pos"],
                "n_neg": metrics["n_neg"],
                "n_pred_yes": metrics["n_pred_yes"],
                "n_pred_no": metrics["n_pred_no"],
                "n_unknown": metrics["n_unknown"],

                "accuracy": metrics["accuracy"],
                "tp_rate": metrics["tp_rate"],
                "tn_rate": metrics["tn_rate"],
                "fp_rate": metrics["fp_rate"],
                "fn_rate": metrics["fn_rate"],

                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "yes_ratio": metrics["yes_ratio"],
                "unknown_ratio": metrics["unknown_ratio"],

                "TP": metrics["TP"],
                "TN": metrics["TN"],
                "FP": metrics["FP"],
                "FN": metrics["FN"],
            })

    if summary_rows:
        summary_csv = out_dir / "summary.csv"
        pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
        print(f"\n==== Wrote summary → {summary_csv}")
    else:
        print("\nNo results written. No annotation files found.")