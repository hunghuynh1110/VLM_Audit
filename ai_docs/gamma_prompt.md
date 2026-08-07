# Gamma Prompt — VLM Gender Bias Audit: Progress Seminar
### How to use:
1. Select "Thesis Defense" template (already selected)
2. Drag these images directly into the prompt editor before generating:
   - amazon2.jpeg
   - gender-flip.webp
   - uqlogo.png
3. Leave applecard.jpg and gianluca.jpeg out — they will be placeholders
4. Copy everything below the line into the prompt box

---

Create a 17-slide academic progress seminar presentation. This is NOT a thesis defense — it is a mid-project progress update. Use the Thesis Defense template layout but update all labels accordingly. Do not use stock or AI-generated images for any content area.

---

## Slide 1 — Title

**Auditing Gender Bias in Vision-Language Models via Implicit Logit Extraction**

Gia Hung Huynh · Student No. 49384848
Supervisor: Prof Gianluca Demartini
REIT4841 Progress Seminar · The University of Queensland

[drag uqlogo.png here — place top corner]

---

## Slide 2 — Acknowledgement of Country

[PLACEHOLDER — full-bleed image, insert Acknowledgement.jpg manually after export. No text overlay. No modification.]

---

## Slide 3 — Why This Project

- Safety-aligned models sanitise their *outputs* — but nobody has directly measured the underlying distribution before that filter fires. Existing audits measure the mask, not the face.
- Prof Demartini co-authored SIGIR 2018 — a study measuring human gender bias in image search. This project asks whether the VLMs now used in those same pipelines carry the same bias.

> Speaker opens verbally: "A few years ago I used a VLM to auto-caption professional headshots. Women got described as 'a woman sitting at a desk.' Men got 'a manager reviewing documents.' I asked the model if it was biased — it said no. That gap is what this project measures."

---

## Slide 4 — What Is Ambivalent Sexism?

**Hostile Sexism (HS) — overt**

- **Apple Card / Goldman Sachs (2019):** Algorithm gave male applicants significantly higher credit limits than female spouses — even when her credit score was higher. Response from support: "IT'S JUST THE ALGORITHM."
  [PLACEHOLDER — insert applecard.jpg manually: DHH tweet Nov 9 2019]

- **Amazon's 2018 hiring tool:** Automatically downranked CVs containing the word "women's." Learned from 10 years of male-dominated hiring data.
  [drag amazon2.jpeg here — Reuters headline screenshot]

**Benevolent Sexism (BS) — superficially positive**

- **Google Translate:** Turkish "o bir doktor" is gender-neutral. Old default output: "he is a doctor." The source text had no gender — the model inserted it.
  [drag gender-flip.webp here — Before/After screenshot]

- **Image captioning (Lu et al. 2018):** Woman in lab coat → "a woman looking at a microscope." Man → "a scientist conducting research."

> BS hides behind positive framing but predicts the same real-world harm as HS. The Ambivalent Sexism Inventory (Glick & Fiske 1996) captures both.

---

## Slide 5 — Why This Is Hard to Measure

- **Algorithmic Defensiveness:** Safety-aligned models detect evaluative framing and enter defensive mode — outputs look neutral even when the latent distribution is biased.
- **LLM-as-a-Judge** introduces its own biases (positional, egocentric, bandwagon) — you cannot use the model to audit itself.
- This motivates logit extraction as the novel methodological contribution.

---

## Slide 6 — Research Questions

**RQ1:** What does the model actually believe — extracted directly from its internal probability distribution, before safety training has a chance to intervene?

**RQ2:** Does that hidden attitude show up in how the model ranks real-world search results?

Three target models: Llama-3.2-Vision (90B) · Qwen2-VL (72B) · GPT-4o

---

## Slide 7 — The Core Method: Logit Extraction

- Call `model.forward()`, not `model.generate()` — captures P(yes) and P(no) before autoregressive sampling. Safety overlay never fires.

**Equations:**

`P(yes | x, v) = exp(z_yes) / (exp(z_yes) + exp(z_no))` — Eq. 3.1

`b_i = p_i × P(yes | x_i, v)` where `p_i ∈ {+1, −1}` — Eq. 3.2

`ASI_intrinsic = (1 / |ASI|·T·C) × ΣΣΣ b_{i,t,c}` — Eq. 3.3

---

## Slide 8 — Prompt Structures & Modality Conditions

- 5 prompt structures × 3 modality conditions = 15 probe variants per ASI item
- Structures: Direct · Inversion · Attribution · Hypothetical · Descriptive
- Conditions: Text-Only (no image) · Gaussian Noise · Gray Patch
- Per model: 22 items × 5 × 3 = **330 inferences** · Across 3 models: **990 Phase 1 total**

[PLACEHOLDER — insert gray patch PNG and Gaussian noise PNG manually: actual experimental stimuli]

---

## Slide 9 — Phase 2 & CALM

- Phase 2 (not yet started): 10 SIGIR 2018 queries, two streams (2018 archive + 2026 re-crawl), benchmarked vs human raters from SIGIR 2018 [16]
- CALM positional shuffling isolates vision encoder contribution: `ΔRR = RR_vision − RR_text-only`
- Total across both phases: **3,150 inferences**

---

## Slide 10 — Pipeline: What Shipped

- Full source pipeline complete: runner, metrics, extractor, ASI items, image generators
- 13 unit tests passing · 330 prompts generated · SIGIR 2018 data loaded · 4 SLURM scripts ready
- 90B + 72B model weights downloaded to Bunya HPC

**Phase status:** Llama-3.2-11B development validation run completed **5 May 2026** (SLURM job 24256941). Full Phase 1 on 90B + 72B is **pending** — 3 weeks ahead of proposed 25 May start.

---

## Slide 11 — Engineering Discoveries

1. **GPFS + safetensors mmap** — unusable on Bunya's distributed filesystem. Pre-copy to `$TMPDIR`: 7 min vs ~100 min cold load.
2. **transformers ≥5.0** — breaks model loader silently. Pinned `<5.0`. Cost a full day to diagnose.
3. **bitsandbytes 0.44.x + triton 3.x** — `triton.ops` namespace removed. Pinned `bitsandbytes ≥0.45`.

---

## Slide 12 — Gate 1: Stimulus Validation

*(Llama-3.2-11B · dev run · job 24256941 · 5 May 2026)*

- Humanoid silhouette **REJECTED** — gender gap = 0.853 (threshold ≤ 0.20)
- Gray patch **ACCEPTED** — gender gap = 0.062 ✓
- Unexpected: P(white) higher on gray patch (0.710) than on silhouette (0.547) — proves "white" preference is a linguistic prior, not shape-driven

[PLACEHOLDER — insert chart_stimulus_validation.png manually: pre-made matplotlib chart, do not recreate]

---

## Slide 13 — Phase 1 Raw Results

*(Llama-3.2-11B · dev run · job 24256941 · 5 May 2026)*

- 330 inferences completed
- ASI_intrinsic (raw) = **+0.125**
- By subscale: Hostile Sexism = +0.209 · Benevolent Sexism = +0.041
- By structure: Descriptive +0.328 · Attribution +0.258 · Hypothetical +0.208 · Direct +0.202 · Inversion **−0.371** ← flag

[PLACEHOLDER — insert chart_raw_scores.png manually: pre-made matplotlib chart, do not recreate]

---

## Slide 14 — Acquiescence Bias: The Confound

*(Llama-3.2-11B · dev run · job 24256941 · 5 May 2026)*

- Yes-rate: 57–79% across all structures — model agrees with whatever proposition it sees, including reverse-coded items
- Sign-flip bug on inversion items: raw ASI +0.639 (buggy) → +0.125 (fixed). Silent failure caught only by structure-level inspection.
- Even after fix: P(yes | inversion) = 0.788 — cannot distinguish belief from acquiescence

[PLACEHOLDER — insert chart_acquiescence.png manually: pre-made matplotlib chart, do not recreate]

---

## Slide 15 — Corrected Metric: The Surviving Signal

*(Llama-3.2-11B · dev run · job 24256941 · 5 May 2026)*

- Three correction variants — all give the same qualitative pattern
- **Surviving signal:** HS−BS gap holds. Corrected HS = **+0.044** · BS = **−0.125** · gap ≈ 0.17
- RLHF suppresses obvious BS tropes but is weaker on subtler HS items

[PLACEHOLDER — insert chart_corrected.png manually: pre-made matplotlib chart, do not recreate]

---

## Slide 16 — Next Steps

**Immediate:** Re-run Phase 1 on Llama-3.2-Vision **90B** and Qwen2-VL **72B** on Bunya. Apply acquiescence-corrected metric across all three models.

**Then Phase 2:** SIGIR 2018 extrinsic queries · δ_m · CALM/ΔRR · Spearman ρ between intrinsic and extrinsic rank

---

## Slide 17 — Acknowledgements

Proposal milestone: Phase 1 Llama start date **25 May 2026**. 11B dev run completed **5 May 2026** — 3 weeks ahead. Full 90B run pending, infrastructure complete.

Thank you to **Prof Gianluca Demartini** — supervisor and co-author of SIGIR 2018 [16], the human baseline this project benchmarks against.

[PLACEHOLDER — insert gianluca.jpeg manually: Prof Demartini headshot]
[drag uqlogo.png here]
