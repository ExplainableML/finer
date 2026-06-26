#!/usr/bin/env bash
set -euo pipefail

cd ../.. # navigate back to /finer_benchmarks

INFER_PY="inference.py"

HF_DATASET="xiaorui638/finer-comprecap"
MODEL_TYPE="qwen2_5_vl"
MODEL_PATH="xiaorui638/FINER-Qwen2_5-VL-7B"
MODEL_TAG="FINER-Qwen2_5-VL-7B"

OUT_ROOT="outputs/finer_comprecap/${MODEL_TAG}"
mkdir -p "${OUT_ROOT}"

SUBSETS=(
  "multi_object"
  "multi_attribute"
  "multi_relation"
  "wh"
)

for SPLIT in "${SUBSETS[@]}"; do
  echo "=========================================="
  echo "Model: ${MODEL_PATH}"
  echo "Split: ${SPLIT}"
  echo "Output: ${OUT_ROOT}/${SPLIT}.csv"
  echo "=========================================="

  python "${INFER_PY}" \
    --model_type "${MODEL_TYPE}" \
    --model "${MODEL_PATH}" \
    --hf_dataset "${HF_DATASET}" \
    --hf_split "${SPLIT}" \
    --out "${OUT_ROOT}/${SPLIT}.csv" 
done

echo "Done: ${MODEL_TAG}"