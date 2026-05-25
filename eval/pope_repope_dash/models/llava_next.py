# ash_eval/models/llava_next.py

import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

from .base import VisionLanguageModelAdapter


class LlavaNextAdapter(VisionLanguageModelAdapter):
    def __init__(self, model_name_or_path: str):
        print(f"Loading LLaVA-Next model: {model_name_or_path}")

        self.processor = LlavaNextProcessor.from_pretrained(model_name_or_path)

        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        ).eval()

    @torch.inference_mode()
    def generate_text(self, messages, max_new_tokens: int) -> str:
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        out = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

        gen = self.processor.decode(
            out[0, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        return gen