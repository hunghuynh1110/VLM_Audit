# Seminar Outline — VLM Gender Bias Audit: Progress Seminar
**Talk:** REIT4841 Progress Seminar · 14 May 2026
**Student:** Gia Hung Huynh · 49384848
**Supervisor:** Prof Gianluca Demartini
**Audience:** COMP/CSSE/DECO/ENGG/METR final-year peers + Prof Gianluca Demartini
**Format:** 17 slides (~1 min each; title ~30s, acknowledgement ~30s)

---

## PART 0 — OPENING (2 slides, ~1 min)

### Slide 1 — Title
- Project title, student name (Gia Hung Huynh), student number (49384848), supervisor name
- No date on slide
- 🖼️ `Presentation/uqlogo.png` — corner branding
→ `ai_docs/proposal_summary.md` (Cover)

---

### Slide 2 — Acknowledgement of Country
- Full-bleed image slide. No text overlay. No modifications whatsoever.
- 🖼️ `Presentation/Acknowledgement.jpg` — full bleed, completely unchanged
→ UQ standard protocol. MUST remain unedited.

---

## PART 1 — FRAMING (4 slides, ~4 min)

### Slide 3 — Why This Project
**Speaker opens with (spoken aloud — this text does NOT go on the slide):**
> "A few years ago I was working on a project that involved using a VLM to auto-caption a batch of professional headshots. Women kept getting described as 'a woman sitting at a desk.' Men got 'a manager reviewing documents.' I asked the model directly whether it was treating them differently — it said no. That gap between what the model does and what it claims — is exactly what this project is built to measure."

**Bullets (2 — trimmed from previous 3):**
- Safety-aligned models sanitise their *outputs* — but nobody has directly measured the underlying distribution before that filter fires. Existing audits measure the mask, not the face.
- Prof Demartini co-authored SIGIR 2018 [16] — a study measuring human gender bias in image search. This project asks whether the VLMs now used in those same pipelines carry the same bias.

**Speaker note:** Let the story land before moving to bullets. Don't rush past "it said no" — that's the hook.

→ `ai_docs/proposal_summary.md` (§1.2, §2.3), Reference [16]
→ No image on this slide — motivation is carried verbally

---

### Slide 4 — What Is Ambivalent Sexism? (HS vs BS)

**Hostile Sexism (HS) — overt negative views:**
- Apple Card / Goldman Sachs (2019): algorithm gave male applicants significantly higher credit limits than female spouses — even when her credit score was higher. Apple customer rep could not explain: "IT'S JUST THE ALGORITHM."
  - 🖼️ `Presentation/applecard.jpg` — DHH tweet screenshot (Nov 9, 2019). Key visible line: *"HER CREDIT SCORE WAS HIGHER THAN MINE!!!"* Save this file yourself before the build. Consider cropping to the bottom tweet only to avoid the profanity in the paragraph above.
- Amazon's 2018 hiring tool: automatically downranked CVs containing the word "women's." Learned from 10 years of male-dominated hiring data.
  - 🖼️ `Presentation/amazon2.jpeg` — Reuters headline screenshot: *"Amazon scraps secret AI recruiting tool that showed bias against women"* (Jeffrey Dastin, Oct 11 2018). Real news article, credible source.

**Benevolent Sexism (BS) — superficially positive, still constraining:**
- Google Translate: Turkish "o bir doktor" is gender-neutral. Old default output: "he is a doctor." The source text had no gender — the model inserted it.
  - 🖼️ `Presentation/gender-flip.webp` — Before/After screenshot. "Before" panel shows "he is a doctor" as the default. Show full image; audience can read both panels.
- Image captioning (Lu et al. 2018): woman in lab coat → "a woman looking at a microscope." Man in lab coat → "a scientist conducting research." (Verbal — no image needed)

**Punchline:** BS hides behind positive framing but predicts the same real-world harm as HS. The ASI (Glick & Fiske 1996) was designed to capture both.

**Dropped from previous version:** Word2Vec `man:doctor::woman:nurse` analogy (too technical), Stable Diffusion / DALL-E (mention verbally if asked), Siri/Cortana (dated), CEO.jpg (Google has since balanced results — no longer makes the point), google.jpeg (was paired with CEO example only)

→ `src/data/asi_items.py`, `ai_docs/proposal_summary.md` (§2.1.2, [15])

---

### Slide 5 — Why This Is Hard to Measure
- **Algorithmic Defensiveness**: safety-aligned models detect evaluative framing and enter defensive mode → sanitised outputs that look neutral even when the latent distribution is biased.
- **LLM-as-a-Judge** introduces its own biases (positional, egocentric, bandwagon [12,13,14]) — you cannot use the model to audit itself.
- This motivates logit extraction as the novel methodological contribution.

→ `ai_docs/proposal_summary.md` (§1.2, §2.2.3, §2.3 "Auditing Deadlock")
→ No image

---

### Slide 6 — Research Questions
**Simplified language — was jargon-heavy in previous version:**
- **RQ1:** What does the model actually believe — extracted directly from its internal probability distribution, before safety training has a chance to intervene?
- **RQ2:** Does that hidden attitude show up in how the model ranks real-world search results?
- Three target models: Llama-3.2-Vision (90B), Qwen2-VL (72B), GPT-4o

**Speaker note:** "The key word in RQ1 is 'before' — we're not going around the safety layer, we're measuring upstream of it."

→ `ai_docs/proposal_summary.md` (§1.3)
→ No image

---

## PART 2 — METHODOLOGY (3 slides, ~3 min)

### Slide 7 — The Core Method: Logit Extraction
- Call `model.forward()`, not `model.generate()` — captures P(yes) and P(no) before autoregressive sampling begins. Safety overlay never fires.
- Eq. 3.1: `P(yes | x, v) = exp(z_yes) / (exp(z_yes) + exp(z_no))`
- Eq. 3.2: `b_i = p_i × P(yes | x_i, v)` where `p_i ∈ {+1, −1}`
- Eq. 3.3: `ASI_intrinsic = (1 / |ASI|·T·C) × ΣΣΣ b_{i,t,c}`

→ `PIPELINE_SPEC.md` (Core Equations), `src/models/llama_extractor.py`
→ No image

---

### Slide 8 — Prompt Structures & Modality Conditions
- 5 structures × 3 conditions = 15 probe variants per ASI item
- Structures: Direct, Inversion, Attribution, Hypothetical, Descriptive — one worked example shown on slide
- Conditions: Text-Only (v=∅) / Gaussian Noise (1024×1024) / Gray Patch
- Per model: 22 × 5 × 3 = **330 inferences**. Across 3 models: **990 Phase 1 total**
- 🖼️ Gray patch PNG — actual experimental stimulus from `outputs/phase1/` (programmatically generated real data asset, not AI art)
- 🖼️ Gaussian noise PNG — actual experimental stimulus from `outputs/phase1/` (programmatically generated real data asset, not AI art)
- Text-only condition represented as a blank/label

→ `PIPELINE_SPEC.md`, `data/prompts/phase1_prompts.csv`

---

### Slide 9 — Phase 2 & CALM (brief)
- Phase 2 (not yet started): 10 SIGIR 2018 queries, two streams (2018 archive + 2026 re-crawl), benchmarked vs high-ASI and low-ASI human raters from [16]
- CALM positional shuffling isolates vision encoder contribution: `ΔRR = RR_vision − RR_text-only`. Negative ΔRR = visual input adds instability beyond language baseline.
- Total across both phases: **3,150 inferences**

→ `PIPELINE_SPEC.md` (Phase 2, CALM), `data/sigir2018/final_anonymised.csv`
→ No image

---

## PART 3 — WHAT WAS BUILT (2 slides, ~2 min)

### Slide 10 — Pipeline: What Shipped
- Full source pipeline: runner, metrics, extractor, ASI items, image generators. 13 unit tests passing. 330 prompts generated. SIGIR 2018 data loaded. 4 SLURM scripts ready. 90B + 72B weights downloaded to Bunya.
- **Phase status:** Llama-3.2-**11B** development validation run completed 5 May 2026 (SLURM job 24256941). Full Phase 1 on 90B + 72B is **pending** — pipeline is ready, still 3 weeks ahead of the proposed 25 May start date in the proposal.

→ `TODO.md`, `progress/timeline.md` (job 24256941), `ai_docs/proposal_summary.md` (Table 3.2)
→ No image

---

### Slide 11 — Engineering Discoveries (3 bullets, no inline explanation)
1. **GPFS + safetensors mmap** — unusable on Bunya's distributed filesystem. Fix: pre-copy to `$TMPDIR`. Result: 7 min vs ~100 min cold load.
2. **transformers ≥5.0** — breaks the model loader silently. Pinned `<5.0`. Cost a full day to diagnose.
3. **bitsandbytes 0.44.x + triton 3.x** — `triton.ops` namespace removed in triton 3. Pinned `bitsandbytes ≥0.45`.

**Speaker note:** Each of these was a silent failure — they would have corrupted results without crashing. Mention verbally; the slide just carries the headlines.

→ `progress/findings_summary.md` (Findings 1, 2, 6)
→ No image

---

## PART 4 — EMPIRICAL RESULTS (4 slides, ~4 min)

> ⚠️ All results in slides 12–15 are from the **Llama-3.2-11B development validation run** (SLURM job 24256941, 5 May 2026). Full Phase 1 on 90B and 72B is pending. Label all chart headers and slide titles accordingly.

---

### Slide 12 — Gate 1: Stimulus Validation
- Humanoid silhouette **REJECTED**: gender gap = 0.853 (threshold ≤ 0.20)
- Gray patch **ACCEPTED**: gender gap = 0.062 ✓
- Unexpected finding: P(white) higher on featureless gray patch (0.710) than on silhouette (0.547). Proves "white" preference is a linguistic prior in the model, not shape-driven. Constant across all conditions — does not differentially contaminate any condition.
- 🖼️ **Chart**: `outputs/phase1/charts/chart_stimulus_validation.png` — matplotlib PNG, 200 DPI, transparent background. Horizontal bar chart. Generated from `findings/stimulus_validation/gray_patch_result.json` and `findings/stimulus_validation/silhouette_result.json`. Embedded directly — do not regenerate.

→ `findings/stimulus_validation/summary.md`
→ `findings/stimulus_validation/gray_patch_result.json`
→ `findings/stimulus_validation/silhouette_result.json`

---

### Slide 13 — Phase 1 Raw Results *(Llama-3.2-11B · dev run · job 24256941 · 5 May 2026)*
- 330 inferences completed
- ASI_intrinsic (raw) = **+0.125**
- By subscale: HS = +0.209 / BS = +0.041
- By structure: Descriptive +0.328, Attribution +0.258, Hypothetical +0.208, Direct +0.202, Inversion **−0.371** ← flag
- 🖼️ **Chart**: `outputs/phase1/charts/chart_raw_scores.png` — matplotlib PNG, 200 DPI, transparent background. Two-panel: left = HS/BS subscale breakdown, right = per-structure breakdown (inversion bar in red). Generated from `outputs/phase1/llama_dev_summary.json`. Embedded directly — do not regenerate.

→ `outputs/phase1/llama_dev_summary.json`

---

### Slide 14 — Acquiescence Bias: The Confound *(trimmed — chart carries the weight)*
- Yes-rate: 57–79% across all structures — model agrees with whatever proposition it sees, including reverse-coded items
- Sign-flip bug on inversion items: raw ASI +0.639 (buggy) → +0.125 (fixed). Silent failure — caught only by structure-level inspection.
- Even after fix: P(yes | inversion) = 0.788. Cannot distinguish genuine belief from acquiescence.
- 🖼️ **Chart**: `outputs/phase1/charts/chart_acquiescence.png` — matplotlib PNG, 200 DPI, transparent background. Horizontal bars sorted descending, 0.50 baseline marked. Generated from `outputs/phase1/llama_dev_analysis.json`. Embedded directly — do not regenerate.

→ `progress/findings_summary.md` (Findings 4b, 4c)
→ `outputs/phase1/llama_dev_analysis.json`

---

### Slide 15 — Corrected Metric: The Surviving Signal *(trimmed — chart carries the weight)*
- Three correction variants — all give the same qualitative pattern
- **Surviving signal:** HS−BS gap holds. Corrected HS = **+0.044**, BS = **−0.125**, gap ≈ 0.17
- Interpretation: RLHF suppresses obvious BS tropes (items 12, 14 near zero) but is weaker on subtler HS items
- 🖼️ **Chart**: `outputs/phase1/charts/chart_corrected.png` — matplotlib PNG, 200 DPI, transparent background. HS vs BS corrected bars with amber gap arrow. Generated from `outputs/phase1/llama_dev_analysis.json` (field: `by_subscale.bias_score_adj_struct`). Embedded directly — do not regenerate.

→ `outputs/phase1/llama_dev_analysis.json`
→ `progress/findings_summary.md` (Finding 4d)

---

## PART 5 — WHAT'S NEXT + CLOSE (2 slides, ~1.5 min)

### Slide 16 — Next Steps
- **Immediate:** Re-run Phase 1 on Llama-3.2-Vision **90B** and Qwen2-VL **72B** on Bunya. Apply acquiescence-corrected metric across all three models for cross-model comparison.
- **Then Phase 2:** SIGIR 2018 extrinsic queries, δ_m, CALM/ΔRR, Spearman ρ between intrinsic and extrinsic rank.

→ `ai_docs/action_plan.md`
→ `progress/findings_summary.md` (Finding 4c decision)
→ No image

---

### Slide 17 — Timeline vs Proposal + Acknowledgements
- Milestone comparison: proposal had Phase 1 Llama beginning **25 May**. 11B development validation run completed **5 May** — 3 weeks ahead. Full 90B run pending but infrastructure is complete and ready.
- Acknowledge Prof Gianluca Demartini — supervisor and co-author of SIGIR 2018 [16], the human baseline this project benchmarks against.
- 🖼️ `Presentation/gianluca.jpeg` — Prof Demartini's headshot (UQ ITEE faculty page, real photo)
- 🖼️ `Presentation/uqlogo.png` — UQ logo (UQ brand assets)

→ `ai_docs/proposal_summary.md` (Table 3.2)
→ `progress/timeline.md`

---

## Image Map — All Files, Sources, Assignments

| File | Type | Source | Slide | Purpose |
|------|------|--------|-------|---------|
| `Presentation/Acknowledgement.jpg` | Real photo | UQ standard template | 2 | Full-bleed Acknowledgement of Country — unchanged |
| `Presentation/applecard.jpg` | Real screenshot | DHH tweet, Nov 9 2019 | 4 | HS example — Apple Card gender credit limit discrimination |
| `Presentation/amazon2.jpeg` | Real screenshot | Reuters, Oct 11 2018 | 4 | HS example — Amazon hiring tool bias headline |
| `Presentation/gender-flip.webp` | Real screenshot | Google Translate before/after | 4 | BS example — default gendered translation output |
| `Presentation/amazon.jpeg` | Logo | — | **Dropped** | Superseded by amazon2.jpeg |
| `Presentation/CEO.jpg` | Screenshot | — | **Dropped** | Google has since balanced results; no longer makes the point |
| `Presentation/google.jpeg` | Logo | — | **Dropped** | Was paired with CEO.jpg only |
| `Presentation/saas_amazon.webp` | Data chart | — | **Reserve** | Tech headcount by gender — use verbally or if slide space allows |
| `Presentation/gianluca.jpeg` | Real photo | UQ ITEE faculty page | 17 | Prof Demartini headshot |
| `Presentation/uqlogo.png` | Logo | UQ brand assets | 1, 17 | University branding |
| `outputs/phase1/charts/chart_stimulus_validation.png` | matplotlib chart | Generated from `findings/stimulus_validation/*.json` | 12 | Gate 1 bar chart |
| `outputs/phase1/charts/chart_raw_scores.png` | matplotlib chart | Generated from `outputs/phase1/llama_dev_summary.json` | 13 | Raw HS/BS/structure results (two-panel) |
| `outputs/phase1/charts/chart_acquiescence.png` | matplotlib chart | Generated from `outputs/phase1/llama_dev_analysis.json` | 14 | Yes-rate per structure, 0.50 baseline |
| `outputs/phase1/charts/chart_corrected.png` | matplotlib chart | Generated from `outputs/phase1/llama_dev_analysis.json` | 15 | Corrected HS vs BS with gap arrow |
| Gray patch PNG | Real data asset | `outputs/phase1/` programmatically generated | 8 | Actual experimental stimulus — not AI art |
| Gaussian noise PNG | Real data asset | `outputs/phase1/` programmatically generated | 8 | Actual experimental stimulus — not AI art |

**AI-generated images:** Backgrounds and decorative gradients only. Zero AI-generated content images anywhere in the deck.

---

## Timing Summary

| Part | Slides | Est. time |
|------|--------|-----------|
| Opening | 2 | ~1 min |
| Framing | 4 | ~4 min |
| Methodology | 3 | ~3 min |
| What was built | 2 | ~2 min |
| Results | 4 | ~4 min |
| Next steps + close | 2 | ~1.5 min |
| **Total** | **17** | **~15.5 min** |
