# Timeline

Chronological event log. Append to the bottom. Don't rewrite history.

## 2026-04 (early-to-mid)

- Project proposal finalized (`REIT4841_ProjectProposal_GiaHungHuynh.pdf`).
- Pipeline spec drafted (`PIPELINE_SPEC.md`).
- ASI items + prompt structures + image-condition generators implemented in `src/data/`.
- Stimulus validation (Gate 0) passed: gray-patch silhouette is gender-neutral on Llama-3.2-11B-Vision-Instruct (4-bit). Output: `findings/stimulus_validation/`.

## 2026-05-01

- Downloaded Llama-3.2-90B-Vision-Instruct and Qwen2-VL-72B weights to `/QRISdata/Q9468/huggingface_cache`.
- Commit `65269fd`.

## 2026-05-02 — 05-04

- Built Phase 1 inference runner + metrics module (`src/phase1/`). Commit `2a8c6fd`.
- Added Phase 1 CLI entry point + first Bunya SLURM wrapper. Commit `9d6d9c3`.
- Built Gate 2 preflight script for Llama-90B VRAM/latency measurement. Commit `8aa6c36`.

## 2026-05-04

- Bunya `gpu_cuda` queue wait estimated ~6 days for A100; switched all GPU scripts to H100.
- Discovered `module load cuda/13.0.0` no longer exists after Bunya Rocky8 → Rocky9 migration. Removed module-load lines.

## 2026-05-05

- **Morning:** smoke test on 11B hit a 2-hour SLURM time-limit, stuck at `loading weights file model.safetensors`. Both 4-bit and fp16 paths hung identically. Disk reads via `dd` were fast (216 MB/s), so disk I/O was not the bottleneck.
- **Diagnosis:** triple cause:
  1. `transformers` had been silently upgraded to 5.7.0 by a prior `pip install`. The 5.x loader uses `mmap()` parallelism that punishes GPFS.
  2. Code requested `torch_dtype=float16` on a model whose native dtype is `bfloat16`, forcing extra conversion.
  3. `safetensors` mmap over GPFS triggers per-page network round-trips → ~6M page faults for a 23 GB model.
- **Fix (3 commits):** pin `transformers<5.0`, switch loader to `torch.bfloat16`, and pre-copy model weights to `$TMPDIR` before loading. Commits `7f8754c`, `f89c796`, `df4d3ae`.
- **Side issue:** `bitsandbytes 0.44.x` imports `triton.ops`, which was removed in `triton 3.x` (bundled with torch 2.11). Bumped pin to `bitsandbytes>=0.45`. Commit `b8312e7`.
- **Result:** smoke test (job 24230930) passed end-to-end. 5 inferences ran successfully.
- **Queue wait:** ~4 h 46 min for the smoke job. Logged in `findings/bunya_queue_wait_log.md`.
- **First flagged anomaly:** noise condition shows unexpectedly high `p_yes` (0.89, 0.96) on item 1. Open question.

## 2026-05-05 (evening)

- Audited polarity formula in `src/phase1/runner.py`. Confirmed runner used `polarity * p_yes` regardless of structure — wrong for inversion ("would it be incorrect..."). Added `structure_sign = -1 if INVERSION else 1`.
- Submitted full 330-prompt Phase 1 on Llama-3.2-11B (`scripts/bunya_phase1_11b.sh`, job 24256941). Queue wait dropped to ~10 min (off-peak); compute ~30 min.
- **Three findings from full run:**
  1. Polarity fix validated: `asi_intrinsic` 0.6392 (smoke, buggy) → +0.1251 (full, correct).
  2. Noise anomaly was item-1-specific. Per-condition gap collapses from 0.4 to 0.03 over the full dataset. Multi-seed re-run dropped from plan.
  3. **Acquiescence bias:** mean `p_yes` is 0.57–0.79 across all five structures, including reverse-coded items where "yes" endorses anti-sexist content. The model says yes regardless. Phase 1 metric is confounded.
- Decision pending: acquiescence-controlled metric (cheap), forced-choice redesign (mid), or treat acquiescence as the finding (free, methodological).
- **Late evening:** wrote `scripts/analyze_phase1.py` (per-structure mean subtraction + per-(structure×condition) + log-odds variants). Ran locally on 11B parquet.
  - Overall ASI raw +0.125 → corrected −0.040. Inversion outlier (−0.371) collapses to −0.012. Noise condition lands at zero after correction (stimulus design validated).
  - **Surviving signal:** HS subscale +0.044 vs BS subscale −0.125 (gap ≈ 0.17). Robust to all three correction variants. Likely RLHF: safety training catches obvious BS tropes but is weaker on hostile items.
  - Decision: proceed with **(a) acquiescence correction + (c) cross-model comparison**. Forced-choice redesign deferred.
