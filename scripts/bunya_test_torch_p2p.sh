#!/bin/bash
#SBATCH --job-name=vlm_torchp2p
#SBATCH --account=a_ai_collab
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/torchp2p_%j.out
#SBATCH --error=logs/torchp2p_%j.err

# Is the zero-filled cross-GPU copy a torch 2.11.0+cu130 defect or the machine?
#
# Builds throwaway venvs on /scratch with older, widely-deployed torch builds.
# The project venv on /QRISdata is NOT touched -- it is the working environment
# and its transformers/accelerate/bitsandbytes pins were hard-won.
#
# Installs only; the GPU comparison runs in bunya_test_torch_p2p_gpu.sh.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

export PIP_CACHE_DIR=/scratch/user/$USER/pip_cache
ROOT=/scratch/user/$USER/venvs

build () {
    local name=$1 spec=$2 index=$3
    local v="$ROOT/$name"
    echo "=============================================="
    echo "[build] $name  spec=$spec  index=$index"
    rm -rf "$v"
    python3 -m venv "$v"
    "$v/bin/pip" install -q --upgrade pip
    if [ -n "$index" ]; then
        "$v/bin/pip" install -q "$spec" --index-url "$index"
    else
        "$v/bin/pip" install -q "$spec"
    fi
    "$v/bin/python" -c "import torch; print('[build]', '$name', 'torch', torch.__version__)"
}

mkdir -p "$ROOT"
build torch27 "torch==2.7.1" "https://download.pytorch.org/whl/cu126"
build torch28 "torch==2.8.0" "https://download.pytorch.org/whl/cu128"

echo "[build] DONE"
