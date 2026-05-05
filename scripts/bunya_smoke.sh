#!/bin/bash
#SBATCH --job-name=vlm_smoke
#SBATCH --account=a_ai_collab
#SBATCH --partition=gpu_cuda
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --gres=gpu:l40:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/smoke_%j.out
#SBATCH --error=logs/smoke_%j.err


cd $SLURM_SUBMIT_DIR

mkdir -p logs outputs/phase1

source .venv/bin/activate

export HF_TOKEN=$(cat ~/.cache/huggingface/token)
export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN

# safetensors uses mmap() which causes extreme slowness on NFS-mounted QRISdata.
# Pre-copy the 11B model weights to node-local scratch so loading is fast.
SCRATCH=${TMPDIR:-/scratch/user/$USER}
SCRATCH_HF=$SCRATCH/huggingface_cache
SRC_MODEL=/QRISdata/Q9468/huggingface_cache/hub/models--meta-llama--Llama-3.2-11B-Vision-Instruct

echo "[smoke] scratch=$SCRATCH_HF"
echo "[smoke] copying llama-11b weights (~23 GB) ..."
mkdir -p "$SCRATCH_HF/hub"
rsync -aL --info=progress2 "$SRC_MODEL" "$SCRATCH_HF/hub/"
echo "[smoke] copy done"

export HF_HOME=$SCRATCH_HF
export SAFETENSORS_FAST_GPU=1

python -u scripts/run_phase1.py --model llama_dev --quantization none --limit 5
