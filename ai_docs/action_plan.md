# VLM Audit — Action Plan

Prioritised roadmap from current state to a complete Phase 1 + Phase 2 pipeline. P0/P1/P2 cover the active critical path; P3/P4 are long-lead and parallelisable.

---

## Current state (snapshot)

**Done & committed**
- Project scaffold: `config.yaml`, `src/config.py`, `pyproject.toml`, `requirements.txt`
- ASI inventory: 22 items, reverse-codes `{6, 7, 9, 15, 20, 22}`, 5 structures × 3 conditions = 330 prompts (`src/data/asi_items.py`, 13 tests pass)
- Image stimuli: Gaussian-noise + humanoid silhouette with disk caching (`src/data/image_generators.py`)
- SIGIR 2018 loader: low-ASI baselines + per-query objectivity ratings (`src/data/sigir_loader.py`)
- `BaseExtractor` + `LlamaExtractor` (forward-pass-only, token-agnostic, attention exposed)
- Silhouette-validation script + Bunya SLURM job
- Prompt-export script writing 330 rows to `data/prompts/phase1_prompts.csv`

**Uncommitted**
- Small fix in `src/models/llama_extractor.py` falling back from `outputs.cross_attentions` → `outputs.attentions` — should be committed.

**Pending decisions (gates — see `ai_docs/gating_validation_plan.md`)**
- Llama HF license acceptance.
- Silhouette validation run on Bunya (decides keep-silhouette vs gray-patch).
- 90B-Vision memory fit on Bunya (decides quantization + GPU count).

**Not started**
- Phase 1 runner, metrics, HPC entry script, QwenExtractor, analysis layer, all of Phase 2.

---

## P0 — unblock & commit (≤ 1 day)

1. Commit the `llama_extractor.py` attention-fallback fix.
2. Accept Llama-3.2-Vision license on HF; verify HF token works on Bunya.
3. Run silhouette validation on Bunya. Record gender / race probabilities. Decide silhouette vs gray-patch and document in `PIPELINE_SPEC.md`.

---

## P1 — Phase 1 runnable on Llama (Steps 4–5 in `TODO.md`)

4. `src/phase1/runner.py` — iterate `get_all_prompts()`, call `extract_logits(["yes", "no"])`, write rows `(model, item_id, polarity, structure, condition, z_yes, z_no, p_yes, b_i)` to parquet, with tqdm + JSONL checkpoint/resume.
5. `src/phase1/metrics.py` — `compute_asi_intrinsic`, plus sensitivity breakdowns by structure / condition / subscale.
6. `scripts/run_phase1.py` — CLI `--model {llama_dev, llama, qwen} --device --resume`.
7. `scripts/bunya_run_phase1.sh` — SLURM job for the 90B run, using the resource profile chosen in Gate 2.
8. Tests: runner produces 330 rows, schema correct, resume idempotent (use a small dummy extractor — no model load).
9. Run on Bunya, save to `outputs/phase1/llama.parquet`.

---

## P2 — second model & analysis

10. `src/models/qwen_extractor.py` — same `BaseExtractor` interface; handle Qwen2-VL processor differences.
11. Re-run Phase 1 → `outputs/phase1/qwen.parquet`.
12. `src/analysis/phase1_analysis.py` — load both parquets, ASI_intrinsic ranks, permutation test for directional skew, bar chart + b_i heatmap → `outputs/figures/`.

---

## P3 — Phase 2 prep (long-lead; start in parallel with P1/P2)

13. Apply for Google / Bing image-search API key (procurement lag).
14. Email Otterbacher et al. for any additional context on the 2018 image grids referenced in `images.zip`.
15. Extract `data/sigir2018/images.zip` and audit per-query coverage (`k ≥ 9`).

---

## P4 — Phase 2 + CALM (after P2 lands)

16. Image-crawl script for the contemporary stream.
17. Phase 2 runner using the same `extract_logits(["1", …, "7"])`.
18. CALM positional-shuffle wrapper (4 runs / pair) + RR / CR / ΔRR metrics.
19. GPT-4o extractor (API).

---

## Code-hygiene side-quests (low priority, parallel to P1)

- Add `ruff` / `mypy` config (or skip).
- Verify 90B memory fits the requested Bunya GPU before submitting the 330-inference job (covered by Gate 2).
- Audit prompt grammar across all 5 structures × 22 traits — some traits may read awkwardly under HYPOTHETICAL / INVERSION. Quick eyeball on the exported CSV.

---

## Risks

- **Silhouette leakage:** if Gate 1 fails, `ModalityCondition.SILHOUETTE` becomes a gray patch. Prompts and runner don't change, but `image_generators.py` gets a `gray_patch` branch and the enum value swaps. Better to land that before writing the runner output schema.
- **Phase 2 external dependencies:** API keys + image crawl have the longest lag. Kicking off P3 early matters even though it's not on the critical path.
- **90B feasibility:** if preflight latency is bad, Llama runs use the 11B variant. Update `PIPELINE_SPEC.md` if so.
