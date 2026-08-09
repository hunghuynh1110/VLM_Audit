#!/bin/bash
#SBATCH --job-name=vlm_p2pfix
#SBATCH --account=a_ai_collab
#SBATCH --partition=gpu_cuda
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --gres=gpu:a100:2
#SBATCH --time=00:40:00
#SBATCH --output=logs/p2pfix_%j.out
#SBATCH --error=logs/p2pfix_%j.err

# Does staging cross-GPU copies through host memory actually restore the model?
#
# Three arms on the 11B, which is small enough to hold the whole thing on one
# GPU and therefore gives a trustworthy reference:
#   A  single GPU              -- ground truth
#   B  sharded, no workaround  -- must reproduce the uniform failure
#   C  sharded, with workaround-- must reproduce A
#
# C matching A is the pass condition. B is included so the run also demonstrates
# the failure it claims to fix, on the same node, in the same job.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source .venv/bin/activate
export HF_HUB_OFFLINE=1
M11=/scratch/user/$USER/models/Llama-3.2-11B-Vision-Instruct

echo "[p2pfix] node=$(hostname)"
nvidia-smi --query-gpu=index,name --format=csv,noheader
echo "=============================================="

echo; echo "############ A. single GPU (ground truth) ############"
python -u scripts/diagnose_90b_logits.py --variant llama_dev --model-path "$M11" \
    --device-map cuda:0

echo; echo "############ B. sharded, NO workaround (expect uniform) ############"
python -u scripts/diagnose_90b_logits.py --variant llama_dev --model-path "$M11" \
    --device-map auto

echo; echo "############ C. sharded, WITH workaround (expect == A) ############"
python -u scripts/diagnose_90b_logits.py --variant llama_dev --model-path "$M11" \
    --device-map auto --p2p-workaround

echo "[p2pfix] DONE"
