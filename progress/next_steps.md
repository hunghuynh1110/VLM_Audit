# Next steps

Prioritized work queue. Top of list = do next. Dependencies noted in italics.

---

## Immediate (today / tomorrow)

### 1. Full 330-prompt Phase 1 run on Llama-3.2-11B
- **What:** submit `bunya_smoke.sh` variant without `--limit` (or new `bunya_phase1_11b.sh`).
- **Why:** definitive answer to whether noise anomaly generalizes; complete dev-model dataset; tests checkpoint/resume code at full scale.
- **Cost:** ~10 min queue (recent observation) + ~30–60 min compute.
- **Pre-flight:** polarity sign fix in runner (see blockers_log "Resolved" 2026-05-05); logging visibility patch already applied.

---

## Near-term (this week)

### 2. Diagnose noise-condition anomaly
- **What:** if #2 confirms anomaly across items, regenerate noise with seeds 43–46 and re-run a small subset (item 1, all 3 conditions). Compare p_yes distribution across seeds.
- **Why:** isolates whether the issue is noise-stimulus contamination or a property of *any* high-entropy image.
- **Cost:** ~5 h queue + ~5 min compute.
- **Depends on:** #1.

### 3. Llama-90B preflight on H100
- **What:** submit `bunya_preflight_90b.sh`. Measures VRAM peak + per-inference latency.
- **Why:** Gate 2. Determines whether 8-bit fits in 2× H100 (160 GB total) and gives the latency number needed to budget Phase 1 walltime.
- **Cost:** ~5 h queue + ~30 min compute (~100 GB scratch copy + load + measurement).
- **Depends on:** smoke test on the *exact* code path (bf16 loader, scratch pre-copy) — already validated in #2's prerequisite.

### 4. Update `bunya_run_phase1.sh` walltime from preflight numbers
- **What:** `ceil(330 * mean_latency_seconds * 1.4 / 1800) * 30 min`, rounded up.
- **Cost:** 5 min.
- **Depends on:** #3.

### 5. Phase 1 production run on Llama-3.2-90B
- **What:** submit `bunya_run_phase1.sh` (8-bit).
- **Why:** the actual Phase 1 deliverable.
- **Cost:** ~5 h queue + variable compute (per #4).
- **Depends on:** #3, #4.

---

## Later (after Phase 1)

### 6. Implement Qwen2-VL-72B extractor
- **What:** new file `src/models/qwen_extractor.py` mirroring `LlamaExtractor` interface. Wire into `scripts/run_phase1.py:_build_extractor`.
- **Why:** TODO.md Step 6. Comparison model for the audit.
- **Cost:** ~half day code + similar SLURM setup as 90B.

### 7. Phase 2 design + implementation
- **What:** Phase 2 = behavioral / generative bias measurement (per `PIPELINE_SPEC.md`).
- **Status:** spec exists, code does not.
- **Depends on:** Phase 1 done.

---

## Maintenance / hygiene

- Audit other open `>=` version pins in `requirements.txt` (transformers lesson).
- Add `pyarrow` to `requirements.txt` so local parquet inspection works in fresh venvs.
- Decide whether to commit `ai_docs/progress_logging_plan.md` and `ai_docs/proposal_summary.md` (currently untracked).
