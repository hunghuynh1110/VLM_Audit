#!/bin/bash
#SBATCH --job-name=vlm_phase1_llama90b
#SBATCH --account=a_ai_collab
#SBATCH --partition=gpu_cuda
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --gres=gpu:a100:2
#SBATCH --time=04:00:00
#SBATCH --output=logs/phase1_llama_%j.out
#SBATCH --error=logs/phase1_llama_%j.err

# --time placeholder above is a conservative default (~44 s/inference).
# After Gate 2 preflight, replace it with:
#   ceil(330 * mean_latency_seconds * 1.4 / 1800) * 30 minutes
# rounded up to the nearest half hour.

module load cuda/13.0.0
module load python/3.11.3-gcccore-12.3.0

cd $SLURM_SUBMIT_DIR

mkdir -p logs outputs/phase1

source .venv/bin/activate

export HF_HOME=/QRISdata/Q9468/huggingface_cache
export HF_TOKEN=$(cat ~/.cache/huggingface/token)
export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN

python scripts/run_phase1.py --model llama --quantization 8bit
