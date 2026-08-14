#!/bin/bash
#SBATCH --job-name=vlm_qwen_run
#SBATCH --account=a_ai_collab
#SBATCH --partition=gpu_cuda
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --gres=gpu:h100:2
#SBATCH --time=08:00:00
#SBATCH --output=logs/qwen_run_%j.out
#SBATCH --error=logs/qwen_run_%j.err

# Qwen2-VL-72B production run: Phase 1 then Phase 2, bf16, unquantised.
#
# WHY `short` QOS AND 2 GPUs
#   137 GB of bf16 weights fit in 2x80 GB, which is exactly the `short` QOS GPU
#   cap -- and `short` has priority 20 against `gpu`'s 10, so this schedules in
#   hours rather than days. The 90B needed 3 GPUs and was forced onto `gpu`;
#   Qwen is smaller and does not have to pay that penalty.
#
# Validated by scripts/bunya_smoke_qwen.sh (job 27239544) before this was run:
#   - visual token budget constant across image conditions (1225 both)
#   - Phase 1 captured_mass 0.40-0.998
#   - Phase 2 captured_mass 0.9936-0.9997, so the trailing-space tokenisation
#     inherited from the Llama prompt survives Qwen's BPE
#
# The extractor self-tests on load and raises on degenerate logits, so this
# cannot quietly produce a uniform distribution the way the 90B did.

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs outputs/phase1 outputs/phase2

source .venv/bin/activate
export HF_HUB_OFFLINE=1
export SAFETENSORS_FAST_GPU=1

echo "=============================================="
echo "[qwen] node=$(hostname)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "=============================================="

MODEL_DIR=/scratch/user/$USER/models/Qwen2-VL-72B-Instruct
if [ ! -f "$MODEL_DIR/model.safetensors.index.json" ]; then
    echo "[qwen] FATAL: weights not staged at $MODEL_DIR" >&2
    echo "[qwen] run: sbatch scripts/bunya_stage_qwen.sh" >&2
    exit 1
fi
echo "[qwen] weights=$MODEL_DIR"

# Fail in seconds if the Phase 2 images are missing, rather than after Phase 1.
python -c "
from src.data.sigir_loader import get_image_paths
p = get_image_paths(k=9, verify=True)
print(f'[qwen] SIGIR images OK: {len(p)} queries')
"

echo; echo "################ Phase 1 ################"
python -u scripts/run_phase1.py --model qwen --quantization none \
    --weights-path "$MODEL_DIR"

echo; echo "################ Phase 2 ################"
python -u scripts/run_phase2.py --model qwen --quantization none \
    --weights-path "$MODEL_DIR"

echo "[qwen] peak VRAM per GPU:"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader || true
echo "[qwen] DONE"
