#!/bin/bash
#SBATCH --job-name=vlm_preflight_90b
#SBATCH --account=a_ai_collab
#SBATCH --partition=gpu_cuda
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --gres=gpu:a100:2
#SBATCH --time=00:30:00
#SBATCH --output=logs/preflight_90b_%j.out
#SBATCH --error=logs/preflight_90b_%j.err

module load cuda/13.0.0
module load python/3.11.3-gcccore-12.3.0

cd $SLURM_SUBMIT_DIR

mkdir -p logs

source .venv/bin/activate

export HF_HOME=/QRISdata/Q9468/huggingface_cache
export HF_TOKEN=$(cat ~/.cache/huggingface/token)
export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN

python scripts/preflight_90b.py --quantization 8bit
