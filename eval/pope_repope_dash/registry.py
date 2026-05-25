# ash_eval/registry.py

from models.qwen25vl import Qwen25VLAdapter
from models.internvl35 import InternVL35Adapter
from models.llava_next import LlavaNextAdapter


DEFAULT_MODELS = {
    "qwen25vl": "Qwen/Qwen2.5-VL-7B-Instruct",
    "internvl35": "OpenGVLab/InternVL3_5-8B-HF",
    "llava_next": "llava-hf/llava-v1.6-vicuna-7b-hf",
}


def build_model_adapter(
    model_type: str,
    model_name_or_path: str | None = None,
    lora_dir: str | None = None,
    merge_lora: bool = False,
):
    if model_name_or_path is None:
        model_name_or_path = DEFAULT_MODELS[model_type]

    if model_type == "qwen25vl":
        return Qwen25VLAdapter(
            model_name_or_path=model_name_or_path,
            lora_dir=lora_dir,
            merge_lora=merge_lora,
        )

    if model_type == "internvl35":
        if lora_dir or merge_lora:
            raise ValueError("LoRA loading is currently only implemented for Qwen2.5-VL.")
        return InternVL35Adapter(model_name_or_path=model_name_or_path)

    if model_type == "llava_next":
        if lora_dir or merge_lora:
            raise ValueError("LoRA loading is currently only implemented for Qwen2.5-VL.")
        return LlavaNextAdapter(model_name_or_path=model_name_or_path)

    raise ValueError(f"Unknown model_type: {model_type}")