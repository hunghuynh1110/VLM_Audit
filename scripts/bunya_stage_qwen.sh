#!/bin/bash
#SBATCH --job-name=vlm_stage_qwen
#SBATCH --account=a_ai_collab
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/stage_qwen_%j.out
#SBATCH --error=logs/stage_qwen_%j.err

# Stage Qwen2-VL-72B (136 GB, 38 shards) to /scratch as a flat directory.
#
# Same rationale as scripts/bunya_stage_90b.sh: safetensors mmap() over the
# NFSv4 /QRISdata mount is pathological, so GPU jobs must read from GPFS.
#
# EXPECT THIS TO BE SLOW. Unlike the Llama weights, the Qwen shards are
# migrated to tape-backed storage -- `du` reports a 512-byte stub against a
# 3.6 GB apparent size, and every read triggers a recall. The Llama copy ran at
# ~250 MB/s off warm disk; this one is bounded by the recall, not the network,
# hence the 12 h walltime. It only has to happen once.

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

HUB=/QRISdata/Q9468/huggingface_cache/hub
REPO=models--Qwen--Qwen2-VL-72B-Instruct
DST=/scratch/user/$USER/models/Qwen2-VL-72B-Instruct
NEED_GB=140

echo "[stage-qwen] node=$(hostname)  start=$(date -Is)"

# Quota preflight. The scratch soft quota is 300 G and the staged Llama models
# already account for ~186 G, so this copy can push the fileset over. Exceeding
# the soft limit starts a grace period rather than failing writes outright, but
# a half-copied 136 GB model that dies at 3am is worse than a clear message now.
echo "[stage-qwen] quota:"
mmlsquota --block-size G scratch 2>/dev/null | tail -3 || echo "  (mmlsquota unavailable)"
USED_GB=$(du -sBG /scratch/user/$USER 2>/dev/null | cut -dG -f1)
echo "[stage-qwen] currently using ${USED_GB}G under /scratch/user/$USER"
if [ "$((USED_GB + NEED_GB))" -gt 300 ]; then
    echo "[stage-qwen] WARNING: ${USED_GB}G + ${NEED_GB}G exceeds the 300G soft quota." >&2
    echo "[stage-qwen] Proceeding (hard limit is 5T), but consider removing" >&2
    echo "[stage-qwen]   /scratch/user/$USER/models/Llama-3.2-90B-Vision-Instruct" >&2
    echo "[stage-qwen] once the Llama runs are finished -- re-staging it costs 13 min." >&2
fi

SNAP=$(ls -d "$HUB/$REPO"/snapshots/*/ | head -1)
echo "[stage-qwen] src=$SNAP"
echo "[stage-qwen] dst=$DST"
mkdir -p "$DST"

# -L dereferences the blob symlinks into real files. No original/ directory in
# this repo, but the exclude is harmless and keeps the two staging scripts alike.
time rsync -aL --exclude 'original' --exclude 'original/**' "$SNAP." "$DST/"
echo "[stage-qwen] rsync done=$(date -Is)"

echo "[stage-qwen] verifying ..."
source .venv/bin/activate
python -u - "$SNAP" "$DST" <<'PY'
import json, os, sys
from safetensors import safe_open

sdir, ddir = sys.argv[1], sys.argv[2]
wm = json.load(open(os.path.join(ddir, "model.safetensors.index.json")))["weight_map"]
shards = sorted(set(wm.values()))
print(f"[verify] {len(wm)} tensors across {len(shards)} shards")

bad = []
for sh in shards:
    ssz = os.path.getsize(os.path.realpath(os.path.join(sdir, sh)))
    dsz = os.path.getsize(os.path.join(ddir, sh))
    if ssz != dsz:
        bad.append(f"SIZE MISMATCH {sh}: src={ssz} dst={dsz}")
print(f"[verify] shard sizes: {'OK' if not bad else 'FAILED'}")
for b in bad:
    print("  ", b)

by_shard = {}
for k, v in wm.items():
    by_shard.setdefault(v, []).append(k)
suspect = []
for sh, keys in by_shard.items():
    with safe_open(os.path.join(ddir, sh), framework="pt") as fh:
        present = set(fh.keys())
        for k in keys:
            if k not in present:
                suspect.append(f"ABSENT {k}")
                continue
            sl = fh.get_slice(k)
            shape = sl.get_shape()
            t = (sl[:1] if len(shape) > 1 else sl[:]).float()
            if t.numel() and t.abs().sum().item() == 0:
                suspect.append(f"ZERO {k}")
print(f"[verify] all {len(wm)} tensor headers readable")
print(f"[verify] suspect tensors: {len(suspect)}")
for k in suspect[:20]:
    print("  ", k)
sys.exit(1 if bad else 0)
PY

du -sh "$DST"
echo "[stage-qwen] DONE=$(date -Is)"
