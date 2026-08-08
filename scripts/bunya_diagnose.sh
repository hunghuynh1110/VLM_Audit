#!/bin/bash
#SBATCH --job-name=vlm_diag
#SBATCH --account=a_ai_collab
#SBATCH --partition=gpu_cuda
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --gres=gpu:l40:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/diag_%j.out
#SBATCH --error=logs/diag_%j.err

# Measurement-validity sweep on the 11B. Runs every quantisation variant we
# might use for the 90B, so the scarce 90B slot is spent on a configuration
# already known to be both working and faithful.
#
# bf16 is the reference: job 24256941 produced the 330-row dataset with it.

set -euo pipefail

cd $SLURM_SUBMIT_DIR
mkdir -p logs outputs/diagnostics

source .venv/bin/activate
export HF_TOKEN=$(cat ~/.cache/huggingface/token)
export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN

SCRATCH=${TMPDIR:-/scratch/user/$USER}
SCRATCH_HF=$SCRATCH/huggingface_cache
SRC_MODEL=/QRISdata/Q9468/huggingface_cache/hub/models--meta-llama--Llama-3.2-11B-Vision-Instruct

echo "[diag] staging weights ..."
mkdir -p "$SCRATCH_HF/hub"
time rsync -aL "$SRC_MODEL" "$SCRATCH_HF/hub/"
export HF_HOME=$SCRATCH_HF
export SAFETENSORS_FAST_GPU=1

N=8

# Each variant is allowed to fail without killing the sweep -- the point is to
# learn which ones survive. Failures are recorded explicitly.
run () {
  echo ""
  echo "################################################################"
  echo "### $*"
  echo "################################################################"
  if python -u scripts/diagnose_measurement.py --limit $N "$@"; then
    echo "### RESULT: OK  ($*)"
  else
    echo "### RESULT: FAILED  ($*)"
  fi
}

run --quantization none
run --quantization 4bit
run --quantization 4bit --no-skip
run --quantization 8bit
run --quantization 8bit --no-skip

echo ""
echo "[diag] ALL VARIANTS ATTEMPTED"
