#!/usr/bin/env bash
set -euo pipefail

# Optional: set Hugging Face cache directory.
# By default, this uses the standard cache location unless HF_HOME is already set.
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

# Paths
HF_PATH="<PATH_TO_MERGED_HF_MODEL>"
CUSTOM_ID="OpenGVLab/InternVL3_5-8B"
SAVE_PATH="<PATH_TO_SAVE_CONVERTED_MODEL>"

python convert_internvl_hf2custom.py \
  --custom_path "$CUSTOM_ID" \
  --hf_path "$HF_PATH" \
  --save_path "$SAVE_PATH"