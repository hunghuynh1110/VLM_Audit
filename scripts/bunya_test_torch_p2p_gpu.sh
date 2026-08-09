#!/bin/bash
#SBATCH --job-name=vlm_torchp2pg
#SBATCH --account=a_ai_collab
#SBATCH --partition=gpu_cuda
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:2
#SBATCH --time=00:25:00
#SBATCH --output=logs/torchp2pg_%j.out
#SBATCH --error=logs/torchp2pg_%j.err

# Compare the cross-GPU copy under the project torch (2.11.0+cu130) against
# older builds. If the older builds copy correctly, the zero-fill is a torch
# regression and the fix is a version pin. If every version fails, the fault is
# the node/driver and no library change can help.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

echo "[torchp2p] node=$(hostname)"
nvidia-smi
echo "=============================================="

run_one () {
    local label=$1 py=$2
    echo
    echo "################ $label ################"
    if [ ! -x "$py" ]; then echo "  (missing $py -- skipped)"; return; fi
    "$py" -u scripts/diagnose_p2p_values.py 2>&1 | \
        grep -E "^torch=|^gpu[0-9] ->|   src \(|   dst \(|   equal|   src\.to|dst matches src|n_zero in dst|two reads"
}

run_one "project venv (torch 2.11.0+cu130)" "/QRISdata/Q9468/VLM_Audit/.venv/bin/python"
run_one "torch 2.7.1+cu126"                 "/scratch/user/$USER/venvs/torch27/bin/python"
run_one "torch 2.8.0+cu128"                 "/scratch/user/$USER/venvs/torch28/bin/python"

echo "[torchp2p] DONE"
