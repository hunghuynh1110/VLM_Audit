#!/bin/bash
#SBATCH --job-name=vlm_phase1_llama90b
#SBATCH --account=a_ai_collab
#SBATCH --partition=gpu_cuda
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --gres=gpu:h100:2
#SBATCH --time=04:00:00
#SBATCH --output=logs/phase1_llama_%j.out
#SBATCH --error=logs/phase1_llama_%j.err

# --time placeholder above is a conservative default (~44 s/inference).
# After Gate 2 preflight, replace it with:
#   ceil(330 * mean_latency_seconds * 1.4 / 1800) * 30 minutes
# rounded up to the nearest half hour.


cd $SLURM_SUBMIT_DIR

mkdir -p logs outputs/phase1

source .venv/bin/activate

export HF_TOKEN=$(cat ~/.cache/huggingface/token)
export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN

# Pre-copy 90B weights to scratch to avoid mmap-over-NFS hang (~90 GB, ~7 min at 200 MB/s)
SCRATCH=${TMPDIR:-/scratch/user/$USER}
SCRATCH_HF=$SCRATCH/huggingface_cache
SRC_MODEL=/QRISdata/Q9468/huggingface_cache/hub/models--meta-llama--Llama-3.2-90B-Vision-Instruct

echo "[phase1] scratch=$SCRATCH_HF"
echo "[phase1] copying llama-90b weights ..."
mkdir -p "$SCRATCH_HF/hub"
rsync -aL --info=progress2 "$SRC_MODEL" "$SCRATCH_HF/hub/"
echo "[phase1] copy done"

export HF_HOME=$SCRATCH_HF
export SAFETENSORS_FAST_GPU=1

python -u scripts/run_phase1.py --model llama --quantization 8bit
