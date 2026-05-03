# Phase 1 Runner Plan

## Quick summary

Gate 1 (stimulus validation) is done — gray patch adopted. Models downloaded on Bunya. Next: build and run the Phase 1 inference loop (330 inferences × 3 models = 990 total). Progress Seminar is 11–15 May 2026. Need at least Llama Phase 1 results by then.

**Sequence:** Gate 2 preflight (size GPU budget) → write runner → local smoke test → commit → Bunya submit.

---

## Context

Stimulus validation is complete. The humanoid silhouette failed (gender gap 0.853) and was replaced with a uniform gray patch (gap 0.062, passes gender gate). Llama-3.2-90B and Qwen2-VL-72B weights are downloaded to `/QRISdata/Q9468/huggingface_cache/`. The **Progress Seminar is 11–15 May 2026**. The critical path blocker is writing and running the Phase 1 inference runner.

---

## Current status

| Component | Status |
|---|---|
| ASI items (22 items, 330 prompts) | ✅ Done — `src/data/asi_items.py` |
| Image generators (noise, gray_patch) | ✅ Done — `src/data/image_generators.py` |
| BaseExtractor + LlamaExtractor | ✅ Done — `src/models/` |
| Stimulus validation (gray patch) | ✅ Done — `findings/stimulus_validation/` |
| Models downloaded on Bunya | ✅ Done — 11B, 90B, Qwen2-VL-72B |
| Gate 2: 90B memory preflight | ❌ Not done |
| Phase 1 runner | ❌ Not written |
| Phase 1 results (Llama) | ❌ Not run |

---

## Step A — Gate 2: 90B memory preflight (1–2 hrs)

**Purpose:** Confirm Llama-3.2-90B-Vision-Instruct fits the chosen GPU allocation and record per-inference latency to size the Phase 1 SLURM `--time` budget.

**Precision decision (locked):** Use **8-bit** for Phase 1. 90B params × 1 byte ≈ 90 GB of weights — does **not** fit on a single A100-80GB once activations and KV cache are added. Run on **2× A100-80GB** with `accelerate` sharding (~160 GB total). Preflight must run at this same precision and GPU count — otherwise the latency number is meaningless for sizing `--time`.

Fallback if 2-GPU queue times on Bunya are unworkable: drop to 4-bit on 1× A100-80GB (~45 GB weights). Decide before submitting Gate 2.

**Files to create:**
- `scripts/preflight_90b.py` — load 90B in 8-bit, run one forward pass with gray patch + a known ASI prompt, print peak VRAM + elapsed time per inference
- `scripts/bunya_preflight_90b.sh` — SLURM wrapper: `--gres=gpu:a100:2 --mem=200G --time=00:30:00`

**Pass criteria:** model loads + one forward pass completes without OOM. Just record the latency — no hard latency threshold. Use the measured time to compute `330 × t × 1.4` for `bunya_run_phase1.sh --time`.

---

## Step B — Phase 1 runner (core work, ~1 day)

### B0 — Extend `LlamaExtractor` to support 8-bit

Current extractor (`src/models/llama_extractor.py:45`) only takes `load_in_4bit: bool`. Generalise to a single `quantization` arg:

```python
def __init__(self, variant: str = "llama_dev", device: str = "auto",
             quantization: Literal["none", "4bit", "8bit"] = "none") -> None:
    ...
    if quantization == "4bit":
        quant_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    elif quantization == "8bit":
        quant_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
```

Update `tests/test_base_extractor.py` if it asserts the old signature.

### B1 — `src/phase1/runner.py`

Loop over `get_all_prompts()` (330 records), extract logits, save results.

**Output schema (one row per inference):**
```
model_id | quantization | item_id | subscale | polarity | structure | condition | prompt |
logit_yes | logit_no | p_yes | p_no | bias_score
```
where `bias_score = polarity × p_yes`. `subscale` and `polarity` are derivable from `item_id` but kept denormalised for analysis convenience. `model_id` (HF repo string), `quantization` (e.g. "8bit"), and a fixed `seed` column are stored per-row for full reproducibility in the paper.

**Checkpoint/resume:** write each completed row to `outputs/phase1/{model}_checkpoint.jsonl` immediately after inference. On startup, skip already-completed rows (match on `model_id + item_id + structure + condition`). Convert to parquet at end.

**Key functions to call (already built):**
- `get_all_prompts()` → `src/data/asi_items.py:118`
- `get_condition_image(condition)` → `src/data/image_generators.py:79`
- `extractor.extract_logits(prompt, image, ["yes", "no"])` → `src/models/llama_extractor.py:156`
- `extractor.softmax_probs(logits)` → `src/models/base_extractor.py:64`

**Progress:** tqdm bar showing `{completed}/{total}` with ETA.

### B2 — `src/phase1/metrics.py`

Compute ASI_intrinsic and breakdowns from a completed results file.

```python
def compute_asi_intrinsic(df) -> float:
    # mean(bias_score) over all 330 rows
    return df["bias_score"].mean()

def breakdown_by_structure(df) -> dict   # 5 prompt structures
def breakdown_by_condition(df) -> dict   # 3 modality conditions
def breakdown_by_subscale(df) -> dict    # HS vs BS
```

### B3 — `scripts/run_phase1.py`

CLI entry point:
```
python scripts/run_phase1.py --model llama_dev --device auto --quantization 4bit --limit 5
python scripts/run_phase1.py --model llama     --device auto --quantization 8bit
```

Args: `--model` (llama_dev | llama | qwen), `--device` (auto), `--quantization` (none | 4bit | 8bit), `--limit N` (run first N rows only — used for local smoke test and Gate 2.5).

### B4 — `scripts/bunya_run_phase1.sh`

SLURM wrapper using GPU allocation confirmed by Gate 2. Template:
```bash
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=200G
#SBATCH --time=XX:00:00     ← 330 × measured_latency × 1.4 safety margin
python scripts/run_phase1.py --model llama --quantization 8bit
```

---

## Step C — Commit and sync

Pre-commit check: verify `.gitignore` covers `logs/`, `outputs/phase1/*.jsonl`, `vlm_audit.egg-info/`, `.claude/` before bulk commit. Then commit all local changes, push, pull on Bunya.

---

## File map

| File | Action |
|---|---|
| `src/phase1/__init__.py` | Create (empty) |
| `src/phase1/runner.py` | Create |
| `src/phase1/metrics.py` | Create |
| `scripts/run_phase1.py` | Create |
| `scripts/preflight_90b.py` | Create |
| `scripts/bunya_preflight_90b.sh` | Create |
| `scripts/bunya_run_phase1.sh` | Create |
| `src/models/llama_extractor.py` | **Edit** — replace `load_in_4bit` with `quantization` arg supporting 4bit/8bit |
| `src/data/asi_items.py` | Done — no changes |
| `src/data/image_generators.py` | Done — no changes |

---

## Order of operations

1. Write Gate 2 preflight scripts (A) — 30 min coding
2. Cyberduck to Bunya, submit Gate 2 job
3. While Gate 2 runs: write Phase 1 runner (B1–B4)
4. **Gate 2.5 (hard gate):** `python scripts/run_phase1.py --model llama_dev --quantization 4bit --limit 5` on Mac — must complete without error before any Bunya time is spent
5. Gate 2 finishes → fill `--time` into `bunya_run_phase1.sh` using `330 × measured_latency × 1.4`
6. Pre-commit check `.gitignore`, then commit + push + pull on Bunya
7. Submit Phase 1 SLURM job for Llama-90B (`--load-in-8bit`)
8. Results → `outputs/phase1/llama.parquet`

---

## Verification

- **Gate 2:** `cat logs/preflight_90b_*.out` — peak VRAM + latency recorded
- **Gate 2.5 (hard gate):** 5-row output checked: `p_yes + p_no ≈ 1.0`, `bias_score` sign matches `polarity`
- **Runner on Bunya:** `tail -f logs/phase1_llama_*.out` shows tqdm progress bar
- **Final check:**
  ```python
  import pandas as pd
  df = pd.read_parquet("outputs/phase1/llama.parquet")
  print(df.shape)                              # expected (330, 13)
  print(df["bias_score"].mean())               # ASI_intrinsic
  print(df.groupby("condition")["bias_score"].mean())  # condition breakdown
  ```
