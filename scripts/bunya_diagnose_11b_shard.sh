#!/bin/bash
#SBATCH --job-name=vlm_diag11b
#SBATCH --account=a_ai_collab
#SBATCH --partition=gpu_cuda
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:3
#SBATCH --time=00:50:00
#SBATCH --output=logs/diag11b_%j.out
#SBATCH --error=logs/diag11b_%j.err

# The control the original investigation never ran.
#
# "The same code works on the 11B" was taken as evidence that the code and the
# loading path are sound, leaving the 90B's size or its checkpoint as the
# suspect. But the 11B is 21 GB: device_map="auto" puts all of it on GPU 0, so
# the 11B has never once been sharded. The comparison held model size and
# multi-GPU sharding confounded.
#
# This forces the 11B across 3 GPUs with max_memory, changing ONLY sharding.
#   - if it stays sane      -> sharding is exonerated, the fault is 90B-specific
#   - if it goes degenerate -> the bug is sharding, reproduced on a 21 GB model
#
# Runs on A100s deliberately: the 11B needs ~8 GB per GPU, the H100s are
# saturated for the next day, and bun003 is idle now.

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

source .venv/bin/activate
export HF_HUB_OFFLINE=1
M11=/scratch/user/$USER/models/Llama-3.2-11B-Vision-Instruct

echo "=============================================="
echo "[diag11b] node=$(hostname)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "--- GPU topology ---"
nvidia-smi topo -m || true
echo "=============================================="

# device_map="auto" BALANCES across every visible GPU rather than filling GPU 0
# first, so it shards the 11B too. The single-GPU baseline therefore has to be
# pinned explicitly -- otherwise both arms would be sharded and the comparison
# would be empty.
echo; echo "############ 11B, pinned to ONE GPU (true unsharded baseline) ############"
python -u scripts/diagnose_90b_logits.py --variant llama_dev --model-path "$M11" \
    --device-map cuda:0

echo; echo "############ 11B, sharded across 3 GPUs ############"
python -u scripts/diagnose_90b_logits.py --variant llama_dev --model-path "$M11" \
    --device-map auto --max-memory-gb 8,8,8

echo "[diag11b] DONE"
