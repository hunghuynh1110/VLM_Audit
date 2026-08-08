#!/bin/bash
#SBATCH --job-name=vlm_phase2_11b
#SBATCH --account=a_ai_collab
#SBATCH --partition=gpu_cuda
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --gres=gpu:l40:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/phase2_11b_%j.out
#SBATCH --error=logs/phase2_11b_%j.err

# Phase 2 on the 11B, historical SIGIR 2018 stream.
#
# 10 queries x 9 images x 2 conditions x 2 scale orders = 360 inferences.
# Phase 1 ran 330 in 75 s on this hardware, so ~2 min of compute; the hour is
# almost entirely the weight staging.
#
# bf16 to match the Phase 1 reference (job 27104037). Quantisation shifts
# p_yes by 0.065-0.107 against an effect of ~0.15, so it is not used.

set -euo pipefail

cd $SLURM_SUBMIT_DIR
mkdir -p logs outputs/phase2

source .venv/bin/activate
export HF_TOKEN=$(cat ~/.cache/huggingface/token)
export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN

echo "[phase2] node=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# The SIGIR images live in the repo, not the HF cache -- confirm before loading
# a model, so a missing extraction fails in seconds rather than after staging.
python -c "
from src.data.sigir_loader import get_image_paths
p = get_image_paths(k=9, verify=True)
print(f'[phase2] SIGIR images OK: {len(p)} queries x {len(next(iter(p.values())))} images')
"

SCRATCH=${TMPDIR:-/scratch/user/$USER}
SCRATCH_HF=$SCRATCH/huggingface_cache
SRC_MODEL=/QRISdata/Q9468/huggingface_cache/hub/models--meta-llama--Llama-3.2-11B-Vision-Instruct

echo "[phase2] staging weights ..."
mkdir -p "$SCRATCH_HF/hub"
time rsync -aL "$SRC_MODEL" "$SCRATCH_HF/hub/"

export HF_HOME=$SCRATCH_HF
export SAFETENSORS_FAST_GPU=1

python -u scripts/run_phase2.py --model llama_dev --quantization none

echo "[phase2] DONE"
