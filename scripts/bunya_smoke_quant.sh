#!/bin/bash
#SBATCH --job-name=vlm_smokeq
#SBATCH --account=a_ai_collab
#SBATCH --partition=gpu_cuda
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --gres=gpu:l40:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/smokeq_%j.out
#SBATCH --error=logs/smokeq_%j.err

# Smoke test of one quantisation path on the 11B, before committing a scarce
# 90B queue slot to it.
#
# Usage:  sbatch --export=ALL,QUANT=4bit  scripts/bunya_smoke_quant.sh
#
# History: 8bit is BROKEN on this stack (bitsandbytes 0.49.2 + torch 2.11 +
# Mllama). Job 27087751 died at ops.py:145 (.view on non-contiguous tensor);
# job 27086002 died at ops.py:34 (4-D tensor into int8 matmul) after the 90B
# had loaded fine at 85.25 GB. 4bit uses Linear4bit and avoids both.

# Fail loudly. Without this a Python crash still exits 0 and SLURM reports
# COMPLETED -- which is exactly how the 8-bit failures nearly went unnoticed.
set -euo pipefail

QUANT="${QUANT:-4bit}"

cd $SLURM_SUBMIT_DIR
mkdir -p logs outputs

source .venv/bin/activate

export HF_TOKEN=$(cat ~/.cache/huggingface/token)
export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN

echo "=============================================="
echo "[smokeq] quantization=$QUANT  node=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "=============================================="

echo "[smokeq] --- library versions ---"
python -c "
import torch, transformers, accelerate, bitsandbytes as bnb
print('torch       ', torch.__version__)
print('transformers', transformers.__version__)
print('accelerate  ', accelerate.__version__)
print('bitsandbytes', bnb.__version__)
"

# safetensors mmap() is pathological on NFS-mounted QRISdata; stage locally.
SCRATCH=${TMPDIR:-/scratch/user/$USER}
SCRATCH_HF=$SCRATCH/huggingface_cache
SRC_MODEL=/QRISdata/Q9468/huggingface_cache/hub/models--meta-llama--Llama-3.2-11B-Vision-Instruct

echo "[smokeq] staging weights to $SCRATCH_HF"
mkdir -p "$SCRATCH_HF/hub"
time rsync -aL "$SRC_MODEL" "$SCRATCH_HF/hub/"

export HF_HOME=$SCRATCH_HF
export SAFETENSORS_FAST_GPU=1

# Isolated output dir per quantisation: the real outputs/phase1/ checkpoint
# holds all 330 bf16 rows, and the resume key does not include quantization,
# so reusing it would skip every prompt and mix dtypes in one dataset.
OUT="outputs/smoke_$QUANT"
rm -rf "$OUT"

echo "[smokeq] --- $QUANT inference (5 prompts) ---"
python -u scripts/run_phase1.py --model llama_dev --quantization "$QUANT" --limit 5 --output-dir "$OUT"

echo "[smokeq] peak VRAM:"
nvidia-smi --query-gpu=memory.used --format=csv,noheader || true

# Quantisation perturbs logits, and logits are the measurement. Quantify the
# shift against the bf16 reference run (job 24256941) on identical prompts.
echo "[smokeq] --- $QUANT vs bf16 p_yes on identical prompts ---"
python -c "
import pandas as pd
a = pd.read_parquet('$OUT/llama_dev.parquet')
b = pd.read_parquet('outputs/phase1/llama_dev.parquet')
k = ['item_id','structure','condition']
m = a.merge(b, on=k, suffixes=('_q','_bf16'))
m['delta'] = m.p_yes_q - m.p_yes_bf16
print(m[k+['p_yes_q','p_yes_bf16','delta']].to_string(index=False))
print('mean |delta| =', m.delta.abs().mean().round(4))
print('max  |delta| =', m.delta.abs().max().round(4))
"

echo "[smokeq] DONE quantization=$QUANT"
