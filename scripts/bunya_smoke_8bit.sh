#!/bin/bash
#SBATCH --job-name=vlm_smoke8
#SBATCH --account=a_ai_collab
#SBATCH --partition=gpu_cuda
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --gres=gpu:l40:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/smoke8_%j.out
#SBATCH --error=logs/smoke8_%j.err

# Smoke test for the 8-bit quantised code path.
#
# The May smoke test (job 24230930) ran --quantization none, so bitsandbytes
# 8-bit has never actually executed on Bunya. The 90B preflight depends on it.
# This job de-risks that before the 90B slot comes up, using the 11B model.

cd $SLURM_SUBMIT_DIR

mkdir -p logs outputs/phase1

source .venv/bin/activate

export HF_TOKEN=$(cat ~/.cache/huggingface/token)
export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN

echo "=============================================="
echo "[smoke8] node=$(hostname)  gpu=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "=============================================="

# CPU-only checks first, so they still report even if the GPU work fails.
echo "[smoke8] --- library versions ---"
python -c "
import torch, transformers, accelerate, bitsandbytes as bnb
print('torch       ', torch.__version__)
print('transformers', transformers.__version__)
print('accelerate  ', accelerate.__version__)
print('bitsandbytes', bnb.__version__)
print('cuda avail  ', torch.cuda.is_available())
"

# Phase 2 needs single-token logits for the 1-7 rating scale. If any of these
# encode to multiple sub-tokens, the Phase 2 extraction is invalid.
echo "[smoke8] --- Phase 2 tokenizer check (1-7 single-token?) ---"
python -c "
from transformers import AutoProcessor
from src.config import CFG
p = AutoProcessor.from_pretrained(CFG['models']['llama_dev'])
tok = p.tokenizer
for t in ['yes','no','1','2','3','4','5','6','7']:
    ids_plain = tok.encode(t, add_special_tokens=False)
    ids_space = tok.encode(' '+t, add_special_tokens=False)
    print(f'{t!r:5} plain={ids_plain} space={ids_space} single={len(ids_plain)==1}')
"

# safetensors uses mmap() which causes extreme slowness on NFS-mounted QRISdata.
# Pre-copy the 11B model weights to node-local scratch so loading is fast.
SCRATCH=${TMPDIR:-/scratch/user/$USER}
SCRATCH_HF=$SCRATCH/huggingface_cache
SRC_MODEL=/QRISdata/Q9468/huggingface_cache/hub/models--meta-llama--Llama-3.2-11B-Vision-Instruct

echo "[smoke8] scratch=$SCRATCH_HF"
df -h "$SCRATCH"
echo "[smoke8] copying llama-11b weights (~20 GB) ..."
mkdir -p "$SCRATCH_HF/hub"
time rsync -aL --info=progress2 "$SRC_MODEL" "$SCRATCH_HF/hub/"
echo "[smoke8] copy done"

export HF_HOME=$SCRATCH_HF
export SAFETENSORS_FAST_GPU=1

# Isolated output dir: the real outputs/phase1/ checkpoint already holds all 330
# bf16 rows, and the resume key does not include quantization -- reusing it would
# make the runner skip every prompt and would mix dtypes in one dataset.
OUT=outputs/smoke8
rm -rf "$OUT"

echo "[smoke8] --- 8-bit inference (5 prompts) ---"
python -u scripts/run_phase1.py --model llama_dev --quantization 8bit --limit 5 --output-dir "$OUT"

echo "[smoke8] peak VRAM:"
nvidia-smi --query-gpu=memory.used --format=csv,noheader

# Compare 8-bit p_yes against the bf16 values from job 24256941 on the same
# prompts. A large shift would mean quantization perturbs the measurement.
echo "[smoke8] --- 8bit vs bf16 p_yes on identical prompts ---"
python -c "
import pandas as pd
a = pd.read_parquet('$OUT/llama_dev.parquet')
b = pd.read_parquet('outputs/phase1/llama_dev.parquet')
k = ['item_id','structure','condition']
m = a.merge(b, on=k, suffixes=('_8bit','_bf16'))
m['delta'] = m.p_yes_8bit - m.p_yes_bf16
print(m[k+['p_yes_8bit','p_yes_bf16','delta']].to_string(index=False))
print('max |delta| =', m.delta.abs().max().round(4))
" || echo "(comparison skipped)"

echo "[smoke8] DONE"
