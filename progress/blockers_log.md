# Blockers log

Each entry: symptom · root cause · resolution · time lost. Append as
encountered. Move resolved-during-this-session items to "Resolved" only after
the fix is confirmed.

---

## Open

### Bunya queue wait (~5 h per job submission)

- **Symptom:** any GPU job sits in `(Priority)` for hours before starting.
- **Root cause:** fairshare queue under load, not a fixable engineering issue.
- **Mitigation strategy:** pack as much work into each submission as possible. Avoid the temptation to "just rerun a 5-minute test" — costs 5 h.

---

## Resolved

### 2026-05-05 (eve) — Acquiescence bias confounds Phase 1 metric

- **Symptom:** Llama-3.2-11B mean p_yes is 0.57–0.79 across **every** prompt structure, including reverse-coded items where "yes" endorses an anti-sexist statement.
- **Impact:** naive `asi_intrinsic = +0.125` was almost entirely yes-bias × polarity arithmetic. Per-structure variation (especially the −0.37 inversion outlier) was an artifact, not a signal.
- **Resolution:** built `scripts/analyze_phase1.py` with three correction variants (per-structure mean subtraction, per-(structure×condition) mean subtraction, log-odds version). All three give the same qualitative result: corrected overall ASI ≈ −0.04, structural variation flattens to ±0.07, noise condition validated at zero.
- **Surviving signal:** HS subscale +0.044 vs BS −0.125 (gap ≈ 0.17). Robust across correction variants. Likely RLHF asymmetry — safety training catches benevolent-sexism tropes but is weaker on hostile items.
- **Forward plan:** apply same script to Qwen2-VL and Llama-90B parquets when available; report cross-model corrected scores plus per-model acquiescence baselines side-by-side.

### 2026-05-05 — Polarity sign on inversion items

- **Symptom:** runner used `bias_score = polarity * p_yes` regardless of structure. `asi_items.py` sets one fixed polarity per item; inversion structure ("would it be incorrect to say...") was not sign-flipped anywhere.
- **Impact:** every inversion row's `bias_score` was inverted. Smoke test `asi_intrinsic = 0.6392` is wrong; inversion+noise `bias_score = +0.96` should have been `-0.96`.
- **Resolution:** runner now applies `structure_sign = -1 if structure == INVERSION else 1` before multiplying by `polarity`. Other four structures (direct, attribution, hypothetical, descriptive) all read "yes" as agreement and need no flip.
- **Time lost:** ~15 min audit, no compute wasted (only smoke output affected; will be regenerated).

### 2026-05-05 — 2-hour SLURM hang at weight loading

- **Symptom:** smoke test job hits 2-hour walltime, stuck at `loading weights file model.safetensors`. No further log output.
- **Root cause:** triple. (1) `transformers` silently upgraded to 5.7.0 with new lazy-mmap loader. (2) safetensors `mmap()` over GPFS = network round-trip per 4 KB page = ~100 min for cold-cache 23 GB load. (3) Forced fp16 conversion on bf16-native model added work.
- **Resolution:** pin transformers <5, switch to bf16, pre-copy weights to `$TMPDIR` before load. Commits `7f8754c`, `f89c796`, `df4d3ae`.
- **Time lost:** ~3 days of failed jobs and queue waits.

### 2026-05-05 — bitsandbytes import error

- **Symptom:** `ModuleNotFoundError: No module named 'triton.ops'` on `import bitsandbytes`.
- **Root cause:** `bitsandbytes 0.44.x` references `triton.ops`, removed in `triton 3.x` (bundled with torch 2.11).
- **Resolution:** pin `bitsandbytes>=0.45.0`. Commit `b8312e7`.
- **Time lost:** ~30 min.

### 2026-05-05 — pip OSError disk quota exceeded

- **Symptom:** `pip install torch` fails with `[Errno 122] Disk quota exceeded` after downloading 71 MB.
- **Root cause:** Bunya `/tmp` has 200 MB quota; pip uses `/tmp` as download/build dir; torch is 530 MB.
- **Resolution:** `export TMPDIR=/QRISdata/Q9468/tmp; export PIP_CACHE_DIR=/QRISdata/Q9468/pip_cache`.
- **Time lost:** ~20 min.

### 2026-05-04 — `module load cuda/13.0.0` does not exist

- **Symptom:** SLURM jobs fail at module-load step.
- **Root cause:** Bunya migrated Rocky8 → Rocky9; standalone CUDA module retired. PyTorch wheels bundle their own CUDA (cu13).
- **Resolution:** removed `module load` lines from all GPU SLURM scripts. Commit `792b847`.
- **Time lost:** ~1 h.

### 2026-05-04 — A100 partition queue wait ~6 days

- **Symptom:** `--gres=gpu:a100:N` jobs estimated to start 2026-05-10.
- **Root cause:** A100 nodes oversubscribed.
- **Resolution:** switched to H100 in 90B scripts (`bunya_preflight_90b.sh`, `bunya_run_phase1.sh`).
- **Time lost:** waited a day before switching, so ~1 day.
