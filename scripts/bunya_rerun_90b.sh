#!/bin/bash
#SBATCH --job-name=vlm_rerun90b
#SBATCH --account=a_ai_collab
#SBATCH --partition=gpu_cuda
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=250G
#SBATCH --gres=gpu:a100:3
#SBATCH --time=04:00:00
#SBATCH --output=logs/rerun90b_%j.out
#SBATCH --error=logs/rerun90b_%j.err

# Independent replication of the 90B result, on DIFFERENT hardware.
#
# WHY A100 AND NOT ANOTHER H100 RUN
#   model.forward() is deterministic, so repeating the original 3xH100 job
#   would reproduce the same numbers whether they are right or wrong -- it
#   tests determinism, not correctness. The open question is whether the
#   host-staged cross-GPU workaround distorts anything, and that is a property
#   of the execution path, not of the RNG. Running on 3xA100 changes the node,
#   the GPU model and the device map while holding the code and weights fixed.
#   Both GPU types carry the same zero-copy fault (jobs 27113633 / 27105190),
#   so the workaround is exercised here too.
#
#   Agreement across the two would mean the result does not depend on the
#   machine it was produced on. Disagreement would mean the workaround, or the
#   sharding, still leaks into the measurement.
#
# Writes to *_rerun_$RERUN_TAG output dirs so the original results are left
# untouched -- and because both runners resume from a checkpoint keyed on
# model_id, so reusing the original directory would skip all 330/360 rows and
# silently do nothing.
#
# Parameterised so the same script covers both the slow full replication and a
# fast partial one, since the A100 queue is days long and debug+H100 is hours:
#   RERUN_TAG    output suffix                       (default a100)
#   RERUN_PHASES which phases to run, e.g. "1"       (default "1 2")
# Override the hardware at submit time, e.g.
#   sbatch --qos=debug --gres=gpu:h100:3 --time=01:00:00 \
#          --export=ALL,RERUN_TAG=h100,RERUN_PHASES=1 scripts/bunya_rerun_90b.sh

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs outputs

source .venv/bin/activate
export HF_TOKEN=$(cat ~/.cache/huggingface/token)
export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN
export HF_HUB_OFFLINE=1
export SAFETENSORS_FAST_GPU=1

echo "=============================================="
echo "[rerun90b] node=$(hostname)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "=============================================="

MODEL_DIR=/scratch/user/$USER/models/Llama-3.2-90B-Vision-Instruct
if [ ! -f "$MODEL_DIR/model.safetensors.index.json" ]; then
    echo "[rerun90b] FATAL: weights not staged at $MODEL_DIR" >&2
    echo "[rerun90b] run: sbatch scripts/bunya_stage_90b.sh" >&2
    exit 1
fi
echo "[rerun90b] weights=$MODEL_DIR"

TAG=${RERUN_TAG:-a100}
PHASES=${RERUN_PHASES:-"1 2"}
echo "[rerun90b] tag=$TAG phases=$PHASES"

for ph in $PHASES; do
    echo; echo "################ Phase $ph ################"
    python -u "scripts/run_phase${ph}.py" --model llama --quantization none \
        --weights-path "$MODEL_DIR" \
        --output-dir "outputs/phase${ph}_rerun_${TAG}"
done

echo "[rerun90b] peak VRAM per GPU:"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader || true
echo "[rerun90b] DONE"
