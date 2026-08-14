#!/bin/bash
#SBATCH --job-name=vlm_smoke_qwen
#SBATCH --account=a_ai_collab
#SBATCH --partition=gpu_cuda
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --gres=gpu:h100:2
#SBATCH --time=01:00:00
#SBATCH --output=logs/smoke_qwen_%j.out
#SBATCH --error=logs/smoke_qwen_%j.err

# First contact between the Qwen2-VL extractor and real weights.
#
# WHY 2 GPUs AND NOT 3
#   137 GB of bf16 weights fit in 2x80 GB with ~20 GB of headroom, which is
#   ample for the short prompts here. Two GPUs still cross a device boundary,
#   so the zero-copy workaround is exercised, and a 2-GPU request schedules
#   sooner and does not compete with the 3-GPU Llama replication already queued
#   ahead of it. If it does not fit, accelerate spills to CPU -- slower, but
#   host<->device copies are the ones that work on this cluster, so still
#   correct.
#
# The extractor self-tests on load and raises on degenerate logits, so a silent
# uniform-distribution result cannot reach the report below.

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

source .venv/bin/activate
export HF_HUB_OFFLINE=1
export SAFETENSORS_FAST_GPU=1

MODEL_DIR=/scratch/user/$USER/models/Qwen2-VL-72B-Instruct
if [ ! -f "$MODEL_DIR/model.safetensors.index.json" ]; then
    echo "[smoke-qwen] FATAL: weights not staged at $MODEL_DIR" >&2
    echo "[smoke-qwen] run: sbatch scripts/bunya_stage_qwen.sh" >&2
    exit 1
fi

echo "=============================================="
echo "[smoke-qwen] node=$(hostname)  weights=$MODEL_DIR"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "=============================================="

python -u scripts/smoke_qwen.py --weights-path "$MODEL_DIR" --limit 6

echo "[smoke-qwen] peak VRAM per GPU:"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader || true
echo "[smoke-qwen] DONE"
