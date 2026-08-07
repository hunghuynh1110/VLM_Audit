# Gamma Prompt — Part 2 (Slides 10–17, add one by one)
### How to use:
After Part 1 generates, click "+" in Gamma to add each card.
Paste one block at a time — do not paste all 8 at once.

---

## ADD SLIDE 10 — Pipeline: What Shipped

Add a slide titled "Pipeline: What Shipped" with these bullets:
- Full source pipeline complete: runner, metrics, extractor, ASI items, image generators
- 13 unit tests passing · 330 prompts generated · SIGIR 2018 data loaded · 4 SLURM scripts ready · 90B + 72B weights on Bunya HPC
- Phase status: Llama-3.2-11B development validation run completed 5 May 2026 (SLURM job 24256941). Full Phase 1 on 90B + 72B is pending — 3 weeks ahead of proposed 25 May start date.

---

## ADD SLIDE 11 — Engineering Discoveries

Add a slide titled "Engineering Discoveries" with exactly 3 bullets, no sub-text:
1. GPFS + safetensors mmap — unusable on Bunya's distributed filesystem. Pre-copy to $TMPDIR: 7 min vs ~100 min cold load.
2. transformers ≥5.0 — breaks model loader silently. Pinned <5.0. Cost a full day to diagnose.
3. bitsandbytes 0.44.x + triton 3.x — triton.ops namespace removed. Pinned bitsandbytes ≥0.45.

---

## ADD SLIDE 12 — Gate 1: Stimulus Validation

Add a slide titled "Gate 1: Stimulus Validation" with subtitle "Llama-3.2-11B · dev run · job 24256941 · 5 May 2026" and these bullets:
- Humanoid silhouette REJECTED — gender gap = 0.853 (threshold ≤ 0.20)
- Gray patch ACCEPTED — gender gap = 0.062 ✓
- Unexpected: P(white) higher on gray patch (0.710) than silhouette (0.547) — proves "white" preference is a linguistic prior, not shape-driven

Leave a large empty image area on the right labelled: [INSERT: chart_stimulus_validation.png]

---

## ADD SLIDE 13 — Phase 1 Raw Results

Add a slide titled "Phase 1 Raw Results" with subtitle "Llama-3.2-11B · dev run · job 24256941 · 5 May 2026" and these bullets:
- 330 inferences completed
- ASI_intrinsic (raw) = +0.125
- By subscale: Hostile Sexism = +0.209 · Benevolent Sexism = +0.041
- By structure: Descriptive +0.328 · Attribution +0.258 · Hypothetical +0.208 · Direct +0.202 · Inversion −0.371 ← flag

Leave a large empty image area labelled: [INSERT: chart_raw_scores.png]

---

## ADD SLIDE 14 — Acquiescence Bias: The Confound

Add a slide titled "Acquiescence Bias: The Confound" with subtitle "Llama-3.2-11B · dev run · job 24256941 · 5 May 2026" and these bullets:
- Yes-rate 57–79% across all structures — model agrees with whatever it sees, including reverse-coded items
- Sign-flip bug on inversion items: raw ASI +0.639 (buggy) → +0.125 (fixed). Silent failure caught only by structure-level inspection.
- Even after fix: P(yes | inversion) = 0.788 — cannot distinguish belief from acquiescence

Leave a large empty image area labelled: [INSERT: chart_acquiescence.png]

---

## ADD SLIDE 15 — Corrected Metric: The Surviving Signal

Add a slide titled "Corrected Metric: The Surviving Signal" with subtitle "Llama-3.2-11B · dev run · job 24256941 · 5 May 2026" and these bullets:
- Three correction variants — all give the same qualitative pattern
- Surviving signal: HS−BS gap holds. Corrected HS = +0.044 · BS = −0.125 · gap ≈ 0.17
- RLHF suppresses obvious BS tropes but is weaker on subtler HS items

Leave a large empty image area labelled: [INSERT: chart_corrected.png]

---

## ADD SLIDE 16 — Next Steps

Add a slide titled "Next Steps" with two sections:

Immediate: Re-run Phase 1 on Llama-3.2-Vision 90B and Qwen2-VL 72B on Bunya. Apply acquiescence-corrected metric across all three models for cross-model comparison.

Then Phase 2: SIGIR 2018 extrinsic queries · δ_m · CALM/ΔRR · Spearman ρ between intrinsic and extrinsic rank

---

## ADD SLIDE 17 — Acknowledgements

Add a final slide titled "Acknowledgements" with:
- Milestone: Proposal had Phase 1 start date 25 May 2026. 11B dev run completed 5 May — 3 weeks ahead. Full 90B run pending, infrastructure complete.
- Thank you to Prof Gianluca Demartini — supervisor and co-author of SIGIR 2018, the human baseline this project benchmarks against.

Leave two placeholder boxes:
[INSERT: gianluca.jpeg — Prof Demartini headshot]
[INSERT: uqlogo.png — UQ logo]
