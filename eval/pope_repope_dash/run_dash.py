#!/usr/bin/env python3
# run_dashb.py

from pathlib import Path
import argparse

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

from prompts import build_messages, parse_yesno
from metrics import BinaryYesNoMetrics
from registry import build_model_adapter, DEFAULT_MODELS


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--model_type",
        required=True,
        choices=sorted(DEFAULT_MODELS.keys()),
        help="Model backend to use.",
    )
    ap.add_argument(
        "--model",
        default=None,
        help="HF model name or local checkpoint path. If omitted, uses default for model_type.",
    )
    ap.add_argument(
        "--split",
        default="test",
        help="Dataset split to use.",
    )
    ap.add_argument(
        "--out",
        required=True,
        help="Output CSV path.",
    )
    ap.add_argument(
        "--max_new_tokens",
        type=int,
        default=3,
    )

    # Qwen LoRA options
    ap.add_argument(
        "--lora_dir",
        type=str,
        default=None,
        help="Path to LoRA adapter folder. Currently only supported for Qwen2.5-VL.",
    )
    ap.add_argument(
        "--merge_lora",
        action="store_true",
        help="Merge LoRA weights into base model after loading. Currently only supported for Qwen2.5-VL.",
    )

    args = ap.parse_args()

    # Load model adapter
    model_adapter = build_model_adapter(
        model_type=args.model_type,
        model_name_or_path=args.model,
        lora_dir=args.lora_dir,
        merge_lora=args.merge_lora,
    )

    # Load dataset
    print("Loading DASH-B dataset from Hugging Face...")
    ds = load_dataset("YanNeu/DASH-B")

    if args.split not in ds:
        raise ValueError(
            f"Split '{args.split}' not found. Available splits: {list(ds.keys())}"
        )

    dset = ds[args.split]
    print(f"Loaded {len(dset):,} rows from split '{args.split}'")

    records = []
    metrics = BinaryYesNoMetrics()

    for row in tqdm(dset, desc="inference", total=len(dset)):
        pil_img = row["image"]
        question = str(row["question"])
        gt_ans = str(row["answer"]).strip().lower()

        if gt_ans not in {"yes", "no"}:
            raise ValueError(f"Unexpected ground-truth label: {gt_ans!r}")

        gt_bin = 1 if gt_ans == "yes" else 0

        messages = build_messages(pil_img, question)

        raw_pred = model_adapter.generate_text(
            messages=messages,
            max_new_tokens=args.max_new_tokens,
        )

        pred_bin = parse_yesno(raw_pred)
        correct = metrics.update(pred_bin=pred_bin, gt_bin=gt_bin)

        records.append({
            "question_id": row.get("question_id", None),
            "img_id": row.get("img_id", None),
            "answer_gt": gt_ans,
            "answer_pred": (
                "yes" if pred_bin == 1 else
                "no" if pred_bin == 0 else
                "unknown"
            ),
            "raw_prediction": raw_pred,
            "correct": correct,
            "question": question,
            "object": row.get("object", None),
            "image_path": row.get("image_path", None),
        })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame.from_records(records).to_csv(out_path, index=False)

    metrics.print_report()
    print(f"CSV with predictions → {out_path}")


if __name__ == "__main__":
    main()