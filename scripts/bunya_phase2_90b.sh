#!/bin/bash
#SBATCH --job-name=vlm_phase2_90b
#SBATCH --account=a_ai_collab
#SBATCH --partition=gpu_cuda
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=250G
#SBATCH --gres=gpu:h100:3
#SBATCH --time=06:00:00
#SBATCH --output=logs/phase2_90b_%j.out
#SBATCH --error=logs/phase2_90b_%j.err

# Phase 2 (historical SIGIR stream) on Llama-3.2-90B, bf16.
#
# Intended to be submitted with a dependency so it only runs if the Phase 1 90B
# job succeeded -- no point spending a 3-GPU slot if the model never loaded:
#
#   sbatch --dependency=afterok:<phase1_jobid> scripts/bunya_phase2_90b.sh
#
# Queueing it early costs nothing and saves a whole queue cycle; H100 waits on
# this cluster are measured in days, not hours.
#
# bf16 for the same reason as Phase 1: measured quantisation error (0.065 at
# 8-bit, 0.107 at 4-bit) is a large fraction of the ~0.15 effect being studied.

set -euo pipefail

cd $SLURM_SUBMIT_DIR
mkdir -p logs outputs/phase2

source .venv/bin/activate
export HF_TOKEN=$(cat ~/.cache/huggingface/token)
export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN

echo "[phase2_90b] node=$(hostname)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

# Fail in seconds if the images are missing, rather than after staging 166 GB.
python -c "
from src.data.sigir_loader import get_image_paths
p = get_image_paths(k=9, verify=True)
print(f'[phase2_90b] SIGIR images OK: {len(p)} queries')
"

SCRATCH=${TMPDIR:-/scratch/user/$USER}
SCRATCH_HF=$SCRATCH/huggingface_cache
SRC_MODEL=/QRISdata/Q9468/huggingface_cache/hub/models--meta-llama--Llama-3.2-90B-Vision-Instruct

echo "[phase2_90b] staging 166 GB of weights ..."
df -h "$SCRATCH"
mkdir -p "$SCRATCH_HF/hub"
time rsync -aL "$SRC_MODEL" "$SCRATCH_HF/hub/"

export HF_HOME=$SCRATCH_HF
export SAFETENSORS_FAST_GPU=1

python -u scripts/run_phase2.py --model llama --quantization none

echo "[phase2_90b] peak VRAM per GPU:"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader || true
echo "[phase2_90b] DONE"
