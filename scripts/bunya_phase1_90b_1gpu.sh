#!/bin/bash
#SBATCH --job-name=vlm_p1_90b_1gpu
#SBATCH --account=a_ai_collab
#SBATCH --partition=gpu_cuda
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/p1_90b_1gpu_%j.out
#SBATCH --error=logs/p1_90b_1gpu_%j.err

# Insurance run: Phase 1 on the 90B using a SINGLE GPU, 4-bit.
#
# Why this exists
# ---------------
# Bunya's PCIe GPU nodes silently zero-fill every cross-GPU copy (see
# progress/findings_summary.md). We have a workaround that routes those copies
# through host memory and is validated bit-identical to single-GPU output, but
# it is a workaround for a platform fault we do not control, and it costs
# performance on every layer boundary.
#
# One GPU means zero cross-device transfers, so the fault cannot apply at all.
# That makes this the most trustworthy 90B configuration available right now,
# independent of RCC fixing anything.
#
# The cost is quantisation: 4-bit is the only variant that fits 90B in 80 GB
# (~45 GB; 8-bit is ~90 GB and does not fit). Measured shift vs bf16 on the 11B
# was 0.107 (job 27103565), against an HS-BS effect of ~0.151 -- so this is a
# cross-check and a fallback, NOT a replacement for the bf16 result.
#
# It also queues far better: 1 GPU on `short` QOS (priority 20) rather than
# 3x H100 on `gpu` (priority 10), which currently means days.

set -euo pipefail

cd $SLURM_SUBMIT_DIR
mkdir -p logs outputs/phase1

source .venv/bin/activate
export HF_TOKEN=$(cat ~/.cache/huggingface/token)
export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN

echo "[p1_90b_1gpu] node=$(hostname)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# Guard: if more than one GPU is visible, device_map="auto" will shard across
# them and walk straight into the zero-copy fault. Fail rather than produce
# plausible-looking garbage.
N=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
if [ "$N" -ne 1 ]; then
    echo "[p1_90b_1gpu] FATAL: $N GPUs visible, expected exactly 1." >&2
    echo "[p1_90b_1gpu] Sharding on this cluster silently zeroes cross-GPU copies." >&2
    exit 1
fi

MODEL_DIR=/scratch/user/$USER/models/Llama-3.2-90B-Vision-Instruct
if [ ! -f "$MODEL_DIR/model.safetensors.index.json" ]; then
    echo "[p1_90b_1gpu] FATAL: weights not staged at $MODEL_DIR" >&2
    echo "[p1_90b_1gpu] run: sbatch scripts/bunya_stage_90b.sh" >&2
    exit 1
fi
echo "[p1_90b_1gpu] weights=$MODEL_DIR"

export HF_HUB_OFFLINE=1
export SAFETENSORS_FAST_GPU=1

python -u scripts/run_phase1.py --model llama --quantization 4bit \
    --weights-path "$MODEL_DIR" \
    --output-dir outputs/phase1_90b_1gpu_4bit

echo "[p1_90b_1gpu] peak VRAM:"
nvidia-smi --query-gpu=memory.used --format=csv,noheader || true
echo "[p1_90b_1gpu] DONE"
