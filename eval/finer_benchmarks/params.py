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

    # Local CSV + image-folder mode.
    # This is useful for FINER-DOCCI, where users should download images separately.
    ap.add_argument(
        "--csv",
        default=None,
        help="MCQ CSV, delimiter auto-detected. Required if --hf_dataset is not used.",
    )
    ap.add_argument(
        "--images",
        default=None,
        help="Directory of images. Required if --hf_dataset is not used.",
    )

    # Hugging Face dataset mode.
    # This is useful for FINER-CompreCap, where images can be bundled in the HF dataset.
    ap.add_argument(
        "--hf_dataset",
        default=None,
        help="HF dataset repo id, e.g. xiaorui638/finer-comprecap-mcq. If set, use HF dataset mode.",
    )
    ap.add_argument(
        "--hf_split",
        default="multi_obj",
        help="HF split name, e.g. multi_attr, multi_obj, multi_rel, or wh.",
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

    if args.hf_dataset is None:
        if args.csv is None:
            raise ValueError("--csv is required when --hf_dataset is not used.")
        if args.images is None:
            raise ValueError("--images is required when --hf_dataset is not used.")

    return args