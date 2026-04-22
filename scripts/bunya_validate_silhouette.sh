#!/bin/bash
#SBATCH --job-name=vlm_silhouette
#SBATCH --account=a_ai_collab
#SBATCH --partition=gpu_cuda
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/silhouette_%j.out
#SBATCH --error=logs/silhouette_%j.err

module load cuda

cd $SLURM_SUBMIT_DIR

mkdir -p logs

source .venv/bin/activate

python scripts/validate_silhouette.py
