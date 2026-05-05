#!/bin/bash
#SBATCH --job-name=vlm-eval
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=<YOUR_PARTITION>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --qos=<YOUR_QOS>
#SBATCH --mem=80G
#SBATCH --time=24:00:00

# Create logs directory before submitting the job, e.g.:
# mkdir -p logs

# Load conda
source "${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate vlmeval

# Caches/paths
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

# Run evaluation
python run.py \
  --data AMBER HallusionBench CRPE_RELATION MMStar TextVQA_VAL ChartQA_TEST MMVP NaturalBenchDataset VStarBench \
  --model InternVL3_5-14B-FINER-Custom \
  --work-dir ./work_FINER-InternVL3_5-14B-Custom \
  --reuse