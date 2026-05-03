#!/bin/bash
#SBATCH --job-name=vlm_silhouette
#SBATCH --account=a_ai_collab
#SBATCH --partition=gpu_cuda
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/silhouette_%j.out
#SBATCH --error=logs/silhouette_%j.err

module load cuda/13.0.0
module load python/3.11.3-gcccore-12.3.0

cd $SLURM_SUBMIT_DIR

mkdir -p logs

source .venv/bin/activate

export HF_HOME=/QRISdata/Q9468/huggingface_cache
export HF_TOKEN=$(cat ~/.cache/huggingface/token)
export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN

python scripts/validate_silhouette.py --quantization 4bit
