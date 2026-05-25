# ash_eval/models/internvl35.py

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from qwen_vl_utils import process_vision_info

from .base import VisionLanguageModelAdapter


class InternVL35Adapter(VisionLanguageModelAdapter):
    def __init__(self, model_name_or_path: str):
        print(f"Loading InternVL-3.5 model: {model_name_or_path}")

        self.processor = AutoProcessor.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
        )

        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        ).eval()

    @torch.inference_mode()
    def generate_text(self, messages, max_new_tokens: int) -> str:
        text_prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text_prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        out = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.model.config.eos_token_id,
        )

        gen = self.processor.batch_decode(
            out[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )[0]

        return gen