#!/bin/bash
#SBATCH --job-name=vlm_90bfix
#SBATCH --account=a_ai_collab
#SBATCH --partition=gpu_cuda
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=250G
#SBATCH --gres=gpu:a100:3
#SBATCH --time=01:00:00
#SBATCH --output=logs/val90b_%j.out
#SBATCH --error=logs/val90b_%j.err

# Confirm the workaround on the 90B itself, without waiting for 3x H100.
#
# 3x A100 80 GB = 240 GB, enough for the 166 GB bf16 checkpoint, and the A100
# nodes are far less contested than the H100s (the production run is not
# expected to start for ~11 h). Same code path, same sharding, same fault --
# only the GPU model differs, and the fault was already shown to affect A100
# and H100 alike.
#
# Fits the 1 h debug QOS only because the weights are pre-staged on GPFS.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source .venv/bin/activate
export HF_HUB_OFFLINE=1
M90=/scratch/user/$USER/models/Llama-3.2-90B-Vision-Instruct

echo "[val90b] node=$(hostname)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "=============================================="

echo; echo "############ 90B sharded, WITH workaround ############"
python -u scripts/diagnose_90b_logits.py --variant llama --model-path "$M90" \
    --device-map auto --p2p-workaround

echo "[val90b] DONE"
