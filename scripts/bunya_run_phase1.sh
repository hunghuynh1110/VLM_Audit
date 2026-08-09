#!/bin/bash
#SBATCH --job-name=vlm_phase1_llama90b
#SBATCH --account=a_ai_collab
#SBATCH --partition=gpu_cuda
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=250G
#SBATCH --gres=gpu:h100:3
#SBATCH --time=06:00:00
#SBATCH --output=logs/phase1_llama_%j.out
#SBATCH --error=logs/phase1_llama_%j.err

# Phase 1 production run: Llama-3.2-90B-Vision, bf16, unquantised.
#
# WHY bf16 AND 3 GPUs (decided 2026-08-08):
#   The 11B reference dataset (job 27104037) is bf16. Job 27103565 measured the
#   quantisation error against it: 8-bit+skip shifts p_yes by 0.065, 4-bit by
#   0.107. The effect under study -- the HS-BS gap -- is 0.151. A 0.065 shift is
#   ~40% of the effect, inside a study whose whole measurement is subtle logit
#   differences, so quantisation is not acceptable here.
#   bf16 weights are 166 GB (37 shards, verified on disk) -> 3x H100 = 240 GB.
#
#   8-bit does now work (skip vision_model/multi_modal_projector/lm_head fixes
#   the int8 kernel crash) and fits 2 GPUs at 85 GB, if a 2-GPU fallback is ever
#   needed. `short` QOS caps at 2 GPUs, hence `gpu` QOS here.
#
# Walltime: the 11B ran 330 prompts in 75 s. Scaling by parameters and adding
# the ~12 min staging copy plus load, 6 h is generous. Verified with
# `sbatch --test-only` that walltime does not affect the start estimate.

set -euo pipefail

cd $SLURM_SUBMIT_DIR

mkdir -p logs outputs/phase1

source .venv/bin/activate

export HF_TOKEN=$(cat ~/.cache/huggingface/token)
export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN

echo "=============================================="
echo "[phase1] node=$(hostname)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "=============================================="

# Weights are pre-staged on GPFS scratch by scripts/bunya_stage_90b.sh, so
# there is no 45 min copy in this job any more. That copy was also duplicating
# a redundant original/*.pth directory, which is why it took 44 min for what is
# only 166 GB of safetensors.
MODEL_DIR=/scratch/user/$USER/models/Llama-3.2-90B-Vision-Instruct
if [ ! -f "$MODEL_DIR/model.safetensors.index.json" ]; then
    echo "[phase1] FATAL: weights not staged at $MODEL_DIR" >&2
    echo "[phase1] run: sbatch scripts/bunya_stage_90b.sh" >&2
    exit 1
fi
echo "[phase1] weights=$MODEL_DIR"

export HF_HUB_OFFLINE=1
export SAFETENSORS_FAST_GPU=1

python -u scripts/run_phase1.py --model llama --quantization none \
    --weights-path "$MODEL_DIR"

echo "[phase1] peak VRAM per GPU:"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader || true
echo "[phase1] DONE"
