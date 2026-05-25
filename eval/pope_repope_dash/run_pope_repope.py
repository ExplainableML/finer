#!/usr/bin/env python3
# run_pope_repope.py

from pathlib import Path
import argparse

from registry import build_model_adapter, DEFAULT_MODELS
from pope_repope_utils import run_pope_repope, BENCHMARKS, SUBSETS


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
        "--ann_dir",
        required=True,
        help="Directory containing coco_pope_*.json and coco_repope_*.json.",
    )
    ap.add_argument(
        "--coco_images",
        required=True,
        help="COCO val2014 image root. Images can be directly inside this folder or under val2014/.",
    )
    ap.add_argument(
        "--out_dir",
        required=True,
        help="Directory to write per-subset CSVs and summary.csv.",
    )
    ap.add_argument(
        "--max_new_tokens",
        type=int,
        default=3,
    )

    ap.add_argument(
        "--benchmarks",
        nargs="+",
        default=list(BENCHMARKS.keys()),
        choices=sorted(BENCHMARKS.keys()),
        help="Benchmarks to evaluate.",
    )
    ap.add_argument(
        "--subsets",
        nargs="+",
        default=SUBSETS,
        choices=SUBSETS,
        help="Subsets to evaluate.",
    )

    # LoRA options, mainly for Qwen2.5-VL
    ap.add_argument(
        "--lora_dir",
        type=str,
        default=None,
        help="Path to LoRA adapter folder. Currently only supported by compatible adapters.",
    )
    ap.add_argument(
        "--merge_lora",
        action="store_true",
        help="Merge LoRA into base model after loading, if supported.",
    )

    args = ap.parse_args()

    model_adapter = build_model_adapter(
        model_type=args.model_type,
        model_name_or_path=args.model,
        lora_dir=args.lora_dir,
        merge_lora=args.merge_lora,
    )

    run_pope_repope(
        model_adapter=model_adapter,
        ann_dir=Path(args.ann_dir),
        img_root=Path(args.coco_images),
        out_dir=Path(args.out_dir),
        max_new_tokens=args.max_new_tokens,
        benchmarks=args.benchmarks,
        subsets=args.subsets,
    )


if __name__ == "__main__":
    main()