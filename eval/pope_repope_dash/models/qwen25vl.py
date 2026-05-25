# ash_eval/models/qwen25vl.py

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

from .base import VisionLanguageModelAdapter


class Qwen25VLAdapter(VisionLanguageModelAdapter):
    def __init__(
        self,
        model_name_or_path: str,
        lora_dir: str | None = None,
        merge_lora: bool = False,
    ):
        print(f"Loading Qwen2.5-VL model: {model_name_or_path}")

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(model_name_or_path)

        if lora_dir:
            print(f"Loading LoRA adapter from: {lora_dir}")
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, lora_dir)

            if merge_lora:
                print("Merging LoRA into base model...")
                self.model = self.model.merge_and_unload()

        self.model.eval()

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
            pad_token_id=self.processor.tokenizer.eos_token_id,
        )

        gen = self.processor.batch_decode(
            out[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )[0]

        return gen