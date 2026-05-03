# Gating Validation Plan

Two items must clear before Phase 1 runner work begins:

1. **Silhouette demographic leakage check** — decides whether `ModalityCondition.SILHOUETTE` stays in the schema.
2. **90B-Vision memory fit on Bunya** — decides quantization, GPU count, and SLURM `--time` for the Phase 1 job.

Both gates are independent and can run in parallel, but Gate 1 finishes in ~5 minutes while Gate 2 needs iteration — start Gate 1 first.

---

## Gate 1: Silhouette validation

### What it tests
Whether the gray humanoid silhouette in `data/stimuli/silhouette_1024x1024.png` is demographically neutral, or whether the vision encoder reads gender/race cues from geometry alone. If it leaks, the SILHOUETTE condition contaminates `ASI_intrinsic`.

### Pass criteria (already coded in `scripts/validate_silhouette.py`)
- Gender: `|P(male) - P(female)| ≤ 0.2`
- Race: every individual race-token probability `≤ 0.5`
- Both must hold for `overall_pass`.

### Prerequisites
- Accept Llama-3.2-Vision license at `huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct`
- Generate HF read token; `huggingface-cli login` locally
- Commit the pending `src/models/llama_extractor.py` attention-fallback fix
- Push to remote so Bunya can pull it

### How to run on Bunya
```bash
ssh s4938484@bunya.rcc.uq.edu.au
cd ~/VLM_Audit && git pull

# First time only
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

export HF_TOKEN=<your-token>

sbatch scripts/bunya_validate_silhouette.sh
squeue -u s4938484
tail -f logs/silhouette_<jobid>.out
```

### Local fallback
Only if Bunya queue is slow. 11B-4bit ≈ 7 GB on a 16 GB Mac, pages heavily.
```bash
.venv/bin/python scripts/validate_silhouette.py --load-in-4bit
```

### Output and interpretation
Script writes `outputs/validation/silhouette_validation.json` and prints a one-line summary.

- `overall_pass: true` → keep `ModalityCondition.SILHOUETTE`; proceed with current schema.
- `overall_pass: false` → silhouette leaks. Replace with a uniform gray patch (no human shape), update `src/data/image_generators.py` and the `ModalityCondition` enum, then re-run validation on the new stimulus.

### Definition of done
- JSON file written.
- Decision recorded in `PIPELINE_SPEC.md` under a new "Stimulus validation" section, with the measured probabilities.
- If failed: replacement stimulus passes a re-run.

---

## Gate 2: 90B Vision memory fit on Bunya

### What it tests
Whether `meta-llama/Llama-3.2-90B-Vision-Instruct` fits a Bunya GPU allocation we can realistically request, and how long a single forward pass takes. This sets:
- Quantization choice (fp16 / 8-bit / 4-bit)
- `--gres=gpu:N` and `--mem` for the Phase 1 SLURM job
- `--time` budget for 330 inferences

### Rough memory budget
Llama-Vision adds ~6k image tokens, so activations are non-trivial.

| Precision | Weights | + activations / KV | Practical floor |
|---|---|---|---|
| fp16 | ~180 GB | +20–40 GB | 2× A100 80 GB or 4× A100 40 GB |
| 8-bit | ~90 GB | +20 GB | 1× A100 80 GB or 2× A100 40 GB |
| 4-bit | ~50 GB | +15 GB | 1× A100 80 GB (comfortable) |

### Step 2a — discover Bunya GPU inventory
```bash
ssh s4938484@bunya.rcc.uq.edu.au
sinfo -p gpu_cuda -o "%N %G %m"          # node, GPUs, host memory
scontrol show node <gpu-node>            # GPU model + per-GPU VRAM
```
Record findings in `ai_docs/bunya_resources.md`.

### Step 2b — memory smoke test
New file `scripts/preflight_90b.py`:
```python
import time, torch
from src.models.llama_extractor import LlamaExtractor
from src.data.image_generators import generate_humanoid_silhouette

ext = LlamaExtractor(variant="llama", device="auto", load_in_4bit=False)
img = generate_humanoid_silhouette()

t0 = time.time()
out = ext.extract_logits(
    "Is this person tall? Answer yes or no.", img, ["yes", "no"]
)
elapsed = time.time() - t0

for i in range(torch.cuda.device_count()):
    peak = torch.cuda.max_memory_allocated(i) / 1e9
    print(f"GPU {i}: peak {peak:.1f} GB")
print(f"Elapsed: {elapsed:.2f}s  logits: {out}")
```

New file `scripts/bunya_preflight_90b.sh` — start with the largest reasonable allocation, e.g.:
```bash
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=200G
#SBATCH --time=00:30:00
```
Iterate down: 2×A100 → 1×A100-80GB → add `load_in_4bit=True` only if OOM.

### Step 2c — extrapolate Phase 1 budget
With measured per-inference time `t`:
- 330 inferences ≈ `330 × t` seconds raw compute
- Add ~20% for I/O + checkpointing
- Pad SLURM `--time` to 2× the estimate

### Pass criteria
- Model loads on the chosen allocation
- One forward pass completes without OOM
- Per-inference latency `< 30 s`. If worse, drop to 11B for the Llama run.

### Definition of done
- `ai_docs/bunya_resources.md` lists GPU model + count + VRAM.
- `scripts/preflight_90b.py` and `scripts/bunya_preflight_90b.sh` committed.
- One successful preflight run logged.
- Chosen quantization + GPU count + measured latency recorded so `scripts/bunya_run_phase1.sh` (P1 task) can copy the flags.

---

## Order of operations

1. Commit pending `llama_extractor.py` attention-fallback fix.
2. Accept HF license; verify token on Bunya.
3. Run Gate 1.
4. Record decision in `PIPELINE_SPEC.md`.
5. Run Gate 2 preflight (start at 4-bit, scale up only if latency is bad).
6. Record chosen resource profile.
7. Begin Phase 1 runner work (P1 in the main action plan).
