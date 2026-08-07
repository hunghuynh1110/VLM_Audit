# Context

SLURM job logs are completely silent during model loading and inference because:
1. `tqdm` auto-disables when stdout/stderr is not a TTY (always the case in SLURM)
2. `transformers.from_pretrained` uses tqdm internally — same issue
3. `import sys` is missing in `llama_extractor.py`, which will crash the new logging code

Goal: make `tail -f logs/smoke_*.out` show real progress at every stage.

---

# Changes

## 1 — `src/models/llama_extractor.py`

**Problem:** `import sys` is missing — `logging.StreamHandler(sys.stdout)` will raise `NameError`.

**Fix:** Add `import sys` to the imports at the top of the file.

**Already in place (keep):**
```python
transformers.logging.set_verbosity_info()
transformers.logging.add_handler(logging.StreamHandler(sys.stdout))
```
This redirects transformers INFO messages (weight file names, dtype, config) to stdout as plain newlines — readable in log files.

## 2 — `src/phase1/runner.py` line 111

**Problem:** `tqdm(pending, desc=..., unit="prompt")` — no `file` or `disable` args.
In a non-TTY environment tqdm auto-disables, producing zero output.

**Fix:**
```python
tqdm(pending, desc=f"phase1[{cfg.model_name}]", unit="prompt",
     file=sys.stdout, disable=False, mininterval=60, miniters=1)
```
- `file=sys.stdout` → goes to `.out` log
- `disable=False` → never auto-disable
- `mininterval=60` → one update line per minute (not every row)

Add `import sys` at top of `runner.py`.

---

# Expected output in `tail -f logs/smoke_*.out`

```
[run_phase1] torch=2.11.0+cu130 cuda_available=True gpu_count=1
[run_phase1] gpu[0] NVIDIA L40 44 GB
[run_phase1] loading model ...
[LlamaExtractor] loading processor for meta-llama/...
[LlamaExtractor] processor loaded
[LlamaExtractor] loading weights (4bit) ...
INFO:transformers.modeling_utils:loading weights file .../model-00001-of-00005.safetensors
INFO:transformers.modeling_utils:loading weights file .../model-00002-of-00005.safetensors
...
[LlamaExtractor] weights loaded
[run_phase1] model loaded
phase1[llama_dev]:  20%|██      | 1/5 [00:05<00:20,  5s/prompt]
phase1[llama_dev]:  40%|████    | 2/5 [01:05<01:35, ...]
```

---

# Files to change

| File | Line | Change |
|---|---|---|
| `src/models/llama_extractor.py` | top | Add `import sys` |
| `src/phase1/runner.py` | 111 + top | Add tqdm args; add `import sys` |

---

# Verification

Resubmit smoke test, then `tail -f logs/smoke_<JOBID>.out` — should show transformers weight file lines during loading, then tqdm rows during inference.
