#!/usr/bin/env python3
from pathlib import Path

import torch
import pandas as pd
from tqdm import tqdm

from params import parse_args
from utils import (
    mcq_text,
    build_msgs,
    match_choice,
    compute_pair_accuracy,
)


def load_model_and_processor(model_type: str, model_name_or_path: str):
    # Feel free to add your own model here : )
    print(f"Loading model type: {model_type}")
    print(f"Loading model path: {model_name_or_path}")

    if model_type == "qwen2_5_vl":
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        ).eval()

        processor = AutoProcessor.from_pretrained(
            model_name_or_path,
        )

        return model, processor

    elif model_type == "internvl":
        from transformers import AutoProcessor, AutoModelForImageTextToText

        processor = AutoProcessor.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
        )

        model = AutoModelForImageTextToText.from_pretrained(
            model_name_or_path,
            device_map="auto",
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).eval()

        return model, processor

    elif model_type == "llava_next":
        from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

        processor = LlavaNextProcessor.from_pretrained(
            model_name_or_path,
        )

        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_name_or_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        ).eval()

        return model, processor

    else:
        raise ValueError(f"Unsupported model_type: {model_type}")


@torch.inference_mode()
def predict_letter_qwen2_5_vl(
    messages,
    model,
    processor,
    max_new_tokens: int,
) -> int:
    from qwen_vl_utils import process_vision_info

    text_prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text_prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=processor.tokenizer.eos_token_id,
    )

    gen_text = processor.batch_decode(
        output_ids[:, inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
    )[0]

    return match_choice(gen_text)


@torch.inference_mode()
def predict_letter_chat_template(
    messages,
    model,
    processor,
    max_new_tokens: int,
) -> int:
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device, dtype=torch.bfloat16)

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )

    gen_text = processor.decode(
        output_ids[0, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )

    return match_choice(gen_text)


def predict_letter(
    messages,
    model,
    processor,
    model_type: str,
    max_new_tokens: int,
) -> int:
    if model_type == "qwen2_5_vl":
        return predict_letter_qwen2_5_vl(
            messages=messages,
            model=model,
            processor=processor,
            max_new_tokens=max_new_tokens,
        )

    elif model_type in {"internvl", "llava_next"}:
        return predict_letter_chat_template(
            messages=messages,
            model=model,
            processor=processor,
            max_new_tokens=max_new_tokens,
        )

    else:
        raise ValueError(f"Unsupported model_type: {model_type}")


def run_local_csv_inference(args, model, processor):
    """
    Original local CSV + local image directory mode.
    This is kept for FINER-DOCCI or any dataset where images should be downloaded separately.
    """
    img_root = Path(args.images)

    df = pd.read_csv(args.csv, sep=None, engine="python")
    print(f"Loaded {len(df):,} rows from {args.csv}")

    pred_idx = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="inference"):
        img_path = img_root / str(row.image)

        choices = [
            str(row.get(f"choice{i}", ""))
            for i in range(1, 6)
        ]

        prompt = mcq_text(str(row.question), choices)
        messages = build_msgs(img_path, prompt)

        pred = predict_letter(
            messages=messages,
            model=model,
            processor=processor,
            model_type=args.model_type,
            max_new_tokens=args.max_new_tokens,
        )

        pred_idx.append(pred)

    df["pred_idx"] = pred_idx
    return df


def run_hf_dataset_inference(args, model, processor):
    """
    Hugging Face dataset mode.

    Expected HF dataset columns:
      - image: HF Image() column, loaded as PIL image
      - image_filename: original filename, used for output CSV and pair accuracy
      - qtype, question, choice1..choice5, gt_index
    """
    from datasets import load_dataset

    ds = load_dataset(args.hf_dataset, split=args.hf_split)
    print(f"Loaded {len(ds):,} rows from HF dataset: {args.hf_dataset} / split={args.hf_split}")

    pred_idx = []
    rows_for_output = []

    for row in tqdm(ds, total=len(ds), desc="inference"):
        if "image_filename" not in row:
            raise ValueError(
                "HF dataset mode expects an `image_filename` column. "
                "This is needed so the output CSV still has a string `image` column "
                "for pair accuracy and traceability."
            )

        image = row["image"].convert("RGB")

        choices = [
            str(row.get(f"choice{i}", ""))
            for i in range(1, 6)
        ]

        prompt = mcq_text(str(row["question"]), choices)
        messages = build_msgs(image, prompt)

        pred = predict_letter(
            messages=messages,
            model=model,
            processor=processor,
            model_type=args.model_type,
            max_new_tokens=args.max_new_tokens,
        )

        pred_idx.append(pred)

        # Keep all non-image columns.
        # The HF `image` column is a PIL image and should not be written to CSV.
        out_row = {
            k: v
            for k, v in row.items()
            if k != "image"
        }

        # Keep compatibility with compute_pair_accuracy(),
        # which expects a string column called `image`.
        out_row["image"] = str(row["image_filename"])

        rows_for_output.append(out_row)

    df = pd.DataFrame(rows_for_output)
    df["pred_idx"] = pred_idx

    return df


def main():
    args = parse_args()

    model, processor = load_model_and_processor(
        model_type=args.model_type,
        model_name_or_path=args.model,
    )

    if args.hf_dataset is not None:
        df = run_hf_dataset_inference(args, model, processor)
    else:
        df = run_local_csv_inference(args, model, processor)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"\nCSV with predictions → {out_path}")

    # Only compute paired accuracy.
    compute_pair_accuracy(
        df,
        name=out_path.name,
        print_report=True,
    )


if __name__ == "__main__":
    main()