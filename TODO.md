# VLM Audit — Phase 1 To-Do

## Legend

- [ ] Not started
- [~] In progress
- [x] Done

---

## Step 1 — ASI Items Data Module ✅

**File:** `src/data/asi_items.py` + `tests/test_asi_items.py`

- [x] Hard-code 22 ASI items with polarity p_i ∈ {+1, -1}
- [x] Generate 5 prompt variants per item (Direct, Inversion, Attribution, Hypothetical, Descriptive)
- [x] Text-only variant: "this person" → "a person"
- [x] Append prompt suffix from config ("Answer yes or no.")
- [x] Test: 22 items, all polarities, prompts end correctly (13/13 passed)

## Step 2 — Image Generators ✅

**File:** `src/data/image_generators.py` + `tests/test_image_generators.py`

- [x] generate_gaussian_noise(size, seed) → PIL.Image
- [x] generate_humanoid_silhouette(size) → PIL.Image
- [x] get_condition_image(condition) → PIL.Image | None
- [x] Cache to data/stimuli/ (seed included in filename)
- [x] Test: shape, mode, pixel range, reproducibility, caching (14/14 passed)

## Step 3 — Logit Extractor (Llama)

**File:** `src/models/base_extractor.py` + `src/models/llama_extractor.py`

- [ ] BaseExtractor: `extract_logits(prompt, image, target_tokens: list[str]) -> dict[str, float]`
- [ ] LlamaExtractor: load 11B locally (dev), swap model ID to 90B for HPC
- [ ] Use model.forward() — NOT model.generate()
- [ ] Resolve token IDs for arbitrary target tokens (not just yes/no)
- [ ] Extract cross-attention weights for heatmaps
- [ ] Test: output is dict, probabilities sum to 1, values in (0,1)

**Prerequisite (before running locally):**
- [ ] Accept Llama license at huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct
- [ ] Create HF access token (read) → `! .venv/bin/huggingface-cli login`
- [ ] Confirm ≥16GB RAM (11B in 4-bit ≈ 7GB); else use HPC

## Step 3b — Silhouette Validation

**File:** `scripts/validate_silhouette.py`

- [ ] Probe: "What gender does this person appear to be? Answer: male, female, or uncertain."
- [ ] Probe: "What race does this person appear to be? Answer: white, black, asian, or uncertain."
- [ ] Flag if |P(male) - P(female)| > 0.2 (gender) or any single race > 0.5
- [ ] Save results to outputs/validation/silhouette_validation.json
- [ ] Decision: keep SILHOUETTE or switch to GRAY_PATCH based on results

## Step 4 — Phase 1 Runner

**File:** `src/phase1/runner.py` + `src/phase1/metrics.py`

- [ ] Nested loop: item × prompt_structure × condition (330 rows/model)
- [ ] Output columns: model, item_id, polarity, prompt_structure, condition, z_yes, z_no, p_yes, b_i
- [ ] Save to outputs/phase1/{model_name}.parquet
- [ ] compute_asi_intrinsic(df) + sensitivity breakdown
- [ ] tqdm progress bar + checkpoint/resume

## Step 5 — Run Phase 1 on Llama (HPC)

**File:** `scripts/run_phase1.py`

- [ ] CLI: --model, --device flags
- [ ] Run 330 inferences, save parquet + summary JSON
- [ ] Print ASI_intrinsic + sensitivity table

## Step 6 — Scale to Qwen2-VL

**File:** `src/models/qwen_extractor.py`

- [ ] Same BaseExtractor interface
- [ ] Handle Qwen2-VL AutoProcessor differences
- [ ] Re-run scripts/run_phase1.py --model qwen

## Step 7 — Validate & Analyse Phase 1

**File:** `src/analysis/phase1_analysis.py`

- [ ] Load both model parquets
- [ ] ASI_intrinsic per model + rank
- [ ] Permutation test (directional skew vs chance)
- [ ] Bar chart + b_i heatmap → outputs/figures/

---

## Phase 2 (future — not planned in detail yet)

- Will reuse BaseExtractor with a different prompt (1–7 rating extraction)
- sigir_loader.py baselines already ready
- Need: GPT-4o extractor, image crawl script, Google/Bing API key
- Apply for API key before Phase 2 starts
