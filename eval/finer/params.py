#!/usr/bin/env python3
import argparse


DEFAULT_MODELS = {
    "qwen2_5_vl": "Qwen/Qwen2.5-VL-7B-Instruct",
    "internvl": "OpenGVLab/InternVL3_5-8B-HF",
    "llava_next": "llava-hf/llava-v1.6-vicuna-7b-hf",
}


def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--model_type",
        required=True,
        choices=["qwen2_5_vl", "internvl", "llava_next"],
        help="Which model family to run.",
    )
    ap.add_argument(
        "--model",
        default=None,
        help="HF model name or local model path. If omitted, use the default for --model_type.",
    )
    ap.add_argument(
        "--csv",
        required=True,
        help="MCQ CSV, delimiter auto-detected.",
    )
    ap.add_argument(
        "--images",
        required=True,
        help="Directory of images.",
    )
    ap.add_argument(
        "--out",
        required=True,
        help="Output CSV with predictions.",
    )
    ap.add_argument(
        "--max_new_tokens",
        type=int,
        default=2,
        help="Maximum number of newly generated tokens.",
    )

    args = ap.parse_args()

    if args.model is None:
        args.model = DEFAULT_MODELS[args.model_type]

    return args