#!/bin/bash
#SBATCH --job-name=vlm_p2p
#SBATCH --account=a_ai_collab
#SBATCH --partition=gpu_cuda
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:3
#SBATCH --time=00:20:00
#SBATCH --output=logs/p2p_%j.out
#SBATCH --error=logs/p2p_%j.err

# Is a cross-GPU copy on this node lossless? Run with peer access on (default)
# and then with NCCL_P2P_DISABLE=1, so the two arms are directly comparable.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source .venv/bin/activate

echo "[p2p] node=$(hostname)"
nvidia-smi --query-gpu=index,name --format=csv,noheader
nvidia-smi topo -m || true

echo; echo "############ default (peer access as configured) ############"
python -u scripts/diagnose_p2p.py || echo "[p2p] EXIT=$? (mismatches found)"

echo; echo "############ NCCL_P2P_DISABLE=1 ############"
NCCL_P2P_DISABLE=1 python -u scripts/diagnose_p2p.py || echo "[p2p] EXIT=$? (mismatches found)"

echo "[p2p] DONE"
