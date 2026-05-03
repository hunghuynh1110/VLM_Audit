#!/bin/bash
#SBATCH --job-name=dl_qwen72b
#SBATCH --account=a_ai_collab
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs/dl_qwen72b_%j.out
#SBATCH --error=logs/dl_qwen72b_%j.err

module load python/3.11.3-gcccore-12.3.0

cd $SLURM_SUBMIT_DIR
mkdir -p logs

source .venv/bin/activate

export HF_HOME=/QRISdata/Q9468/huggingface_cache
export HF_TOKEN=$(cat ~/.cache/huggingface/token)
export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN

python scripts/download_qwen72b.py
