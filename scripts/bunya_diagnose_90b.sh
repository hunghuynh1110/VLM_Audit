#!/bin/bash
#SBATCH --job-name=vlm_diag90b
#SBATCH --account=a_ai_collab
#SBATCH --partition=gpu_cuda
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=250G
#SBATCH --gres=gpu:h100:3
#SBATCH --time=01:00:00
#SBATCH --output=logs/diag90b_%j.out
#SBATCH --error=logs/diag90b_%j.err

# Root-cause diagnostic for the 90B uniform-logit failure.
#
# Reproduces the exact failing configuration -- bf16, device_map=auto over
# 3x H100 -- and traces where the signal dies, then runs the 11B through the
# same code as a control.
#
# This fits in the 1 h `debug` QOS only because the weights are already staged
# on /scratch by scripts/bunya_stage_90b.sh; the old per-job 166 GB copy into
# $TMPDIR took ~45 min on its own.

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

source .venv/bin/activate
export HF_TOKEN=$(cat ~/.cache/huggingface/token)
export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN

MODELS=/scratch/user/$USER/models
M90=$MODELS/Llama-3.2-90B-Vision-Instruct
M11=$MODELS/Llama-3.2-11B-Vision-Instruct
export HF_HUB_OFFLINE=1

echo "=============================================="
echo "[diag] node=$(hostname)  models=$MODELS"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "--- GPU topology (P2P/NVLink) ---"
nvidia-smi topo -m || true
echo "=============================================="

# Order is deliberate: the two 11B runs are ~2 min each and together isolate
# the one variable the original "the 11B works" control never varied. The 11B
# is 21 GB, so device_map=auto puts all of it on GPU 0 -- it has never once
# been sharded. If forcing it across 3 GPUs reproduces the uniform output,
# the cause is sharding, proven on a 20 GB model, and the 90B run below is
# then only confirmation.

echo; echo "############ 11B, unsharded (reproduces the known-good run) ############"
python -u scripts/diagnose_90b_logits.py --variant llama_dev --model-path "$M11"

echo; echo "############ 11B, FORCED across 3 GPUs (the missing control) ############"
python -u scripts/diagnose_90b_logits.py --variant llama_dev --model-path "$M11" \
    --max-memory-gb 8,8,8

echo; echo "#################### 90B (the failing config) ####################"
python -u scripts/diagnose_90b_logits.py --variant llama --model-path "$M90"

echo "[diag] DONE"
