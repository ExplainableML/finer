#!/bin/bash
#SBATCH --job-name=finer-tuning-internvl35-14b
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=384G
#SBATCH --time=09:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

mkdir -p logs

# -----------------------------
# User-configurable variables
# -----------------------------
CONDA_ENV="${CONDA_ENV:-llamafactory}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"

TRAIN_CONFIG="${TRAIN_CONFIG:-examples/train_lora/internvl3_5_14b_lora_finer_tuning.yaml}"

HF_HOME="${HF_HOME:-${REPO_ROOT}/.cache/huggingface}"
WANDB_DIR="${WANDB_DIR:-${REPO_ROOT}/wandb}"

MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-35321}"

WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-finer-tuning}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
# Do not hardcode WANDB_API_KEY in a public script.
# Please run `wandb login` beforehand or export WANDB_API_KEY manually.

export HF_HOME
export WANDB_DIR
export MASTER_ADDR
export MASTER_PORT
export WANDB_MODE
export WANDB_PROJECT

if [[ -n "${WANDB_ENTITY}" ]]; then
  export WANDB_ENTITY
fi

# -----------------------------
# Activate conda environment
# -----------------------------
if [[ -z "${CONDA_EXE:-}" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
fi
conda activate "${CONDA_ENV}"

# -----------------------------
# Launch training
# -----------------------------
python -m torch.distributed.run \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --nnodes=1 \
  --node_rank=0 \
  src/train.py "${TRAIN_CONFIG}"