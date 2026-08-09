#!/bin/bash
#SBATCH --job-name=vlm_stage90b
#SBATCH --account=a_ai_collab
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/stage90b_%j.out
#SBATCH --error=logs/stage90b_%j.err

# One-off: stage model weights to /scratch/user PERMANENTLY, as flat directories.
#
# WHY THIS EXISTS
#   safetensors mmap() over the NFSv4 /QRISdata mount is pathological, so every
#   GPU job re-copied the weights into $TMPDIR first -- ~45 min of a GPU
#   allocation spent on I/O, and enough to make the 1 h `debug` QOS unusable.
#   /scratch is GPFS and shared across compute nodes, so staging once here lets
#   every later job start loading immediately.
#
# WHY A FLAT DIR RATHER THAN AN HF CACHE
#   The HF snapshot is 331 GB, not 166 GB: it ships the safetensors AND a
#   redundant original/consolidated.*.pth that transformers never reads. On top
#   of that, `rsync -aL` on a cache tree dereferences snapshots/ into real files
#   while still copying blobs/, duplicating everything again. Copying a single
#   flat directory of just the safetensors + configs is 166 GB once, and
#   from_pretrained() takes a local path just as happily as a hub id.
#
# Runs CPU-only (no --gres), so it costs no GPU allocation and starts promptly.

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

HUB=/QRISdata/Q9468/huggingface_cache/hub
DSTROOT=/scratch/user/$USER/models

stage_one () {
    local repo=$1 name=$2
    local snap
    snap=$(ls -d "$HUB/models--$repo/snapshots/"*/ | head -1)
    local dst="$DSTROOT/$name"

    echo "[stage] $name"
    echo "[stage]   src=$snap"
    echo "[stage]   dst=$dst"
    mkdir -p "$dst"
    # -L dereferences the blob symlinks into real files; original/ excluded.
    time rsync -aL --exclude 'original' --exclude 'original/**' "$snap." "$dst/"
    du -sh "$dst"
}

echo "[stage] node=$(hostname)  start=$(date -Is)"

stage_one meta-llama--Llama-3.2-90B-Vision-Instruct Llama-3.2-90B-Vision-Instruct
stage_one meta-llama--Llama-3.2-11B-Vision-Instruct Llama-3.2-11B-Vision-Instruct

echo "[stage] rsync done=$(date -Is)"

# Integrity check. Exact per-shard size match against the source catches the
# silent truncation that checksum-less rsync could otherwise leave behind
# (hypothesis 4 in the investigation brief), and reading a slice of every
# tensor validates the safetensors header offsets.
echo "[stage] verifying ..."
source .venv/bin/activate
python -u - "$HUB" "$DSTROOT" <<'PY'
import json, os, sys
from safetensors import safe_open

hub, dstroot = sys.argv[1], sys.argv[2]
pairs = [("meta-llama--Llama-3.2-90B-Vision-Instruct", "Llama-3.2-90B-Vision-Instruct"),
         ("meta-llama--Llama-3.2-11B-Vision-Instruct", "Llama-3.2-11B-Vision-Instruct")]

rc = 0
for repo, name in pairs:
    snaps = os.path.join(hub, f"models--{repo}", "snapshots")
    sdir = os.path.join(snaps, os.listdir(snaps)[0])
    ddir = os.path.join(dstroot, name)
    print(f"\n[verify] {name}")

    idx = os.path.join(ddir, "model.safetensors.index.json")
    wm = json.load(open(idx))["weight_map"]
    shards = sorted(set(wm.values()))
    print(f"[verify]   {len(wm)} tensors across {len(shards)} shards")

    bad = []
    for sh in shards:
        ssz = os.path.getsize(os.path.realpath(os.path.join(sdir, sh)))
        dsz = os.path.getsize(os.path.join(ddir, sh))
        if ssz != dsz:
            bad.append(f"SIZE MISMATCH {sh}: src={ssz} dst={dsz}")
    print(f"[verify]   shard sizes: {'OK' if not bad else 'FAILED'}")
    for b in bad:
        print("     ", b)

    by_shard = {}
    for k, v in wm.items():
        by_shard.setdefault(v, []).append(k)
    dead = []
    for sh, keys in by_shard.items():
        with safe_open(os.path.join(ddir, sh), framework="pt") as fh:
            present = set(fh.keys())
            for k in keys:
                if k not in present:
                    dead.append(f"ABSENT {k}")
                    continue
                sl = fh.get_slice(k)
                shape = sl.get_shape()
                t = (sl[:1] if len(shape) > 1 else sl[:]).float()
                if t.numel() and t.abs().sum().item() == 0:
                    dead.append(f"ZERO {k}")
    print(f"[verify]   all {len(wm)} tensor headers readable")
    print(f"[verify]   suspect tensors: {len(dead)}")
    for k in dead[:20]:
        print("     ", k)
    if bad:
        rc = 1
sys.exit(rc)
PY

du -sh "$DSTROOT"/*
echo "[stage] DONE=$(date -Is)"
