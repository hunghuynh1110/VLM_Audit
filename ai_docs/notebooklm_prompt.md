# NotebookLM Presentation Prompt
### Steps before pasting:
1. Upload `ai_docs/seminar_outline.md` as a source
2. Upload `ai_docs/proposal_summary.md` as a source
3. Click "Bản trình chiếu" (Presentation) in Studio
4. Paste everything below the line

---

Create a 17-slide academic progress seminar presentation based strictly on the uploaded seminar outline. Do not invent content — every bullet, number, and equation must come directly from the outline or proposal summary.

**Context:** REIT4841 honours progress seminar. Student: Gia Hung Huynh (49384848). Supervisor: Prof Gianluca Demartini. Audience: technical university peers and supervisor. Zoom delivery, ~15 minutes. Tone: professional, clean, confident — not a final results talk, this is a progress update.

**Design:** Dark navy background (#1E2761), amber accent (#F4B942), ice blue highlights (#CADCFC), white body text. Minimal text per slide — headlines and short bullets only. No speaker notes needed.

**Slide structure — follow exactly:**

- Slide 1: Title — "Auditing Gender Bias in Vision-Language Models via Implicit Logit Extraction", Gia Hung Huynh, Student No. 49384848, Supervisor: Prof Gianluca Demartini. UQ logo top corner. No date.
- Slide 2: Full-bleed image placeholder labelled [INSERT: Acknowledgement.jpg — UQ Acknowledgement of Country, full bleed, no text overlay, do not modify]
- Slide 3: Why This Project — 2 bullets from outline. No image. The personal story is spoken aloud, not on the slide.
- Slide 4: What Is Ambivalent Sexism? — HS section: Apple Card (2019) bullet + image placeholder [INSERT: applecard.jpg — DHH tweet screenshot]; Amazon hiring tool bullet + image [amazon2.jpeg — Reuters headline screenshot]. BS section: Google Translate bullet + image [gender-flip.webp — before/after screenshot]; Lu et al. 2018 captioning bullet (no image). Punchline at bottom.
- Slide 5: Why This Is Hard to Measure — 3 bullets from outline. No image.
- Slide 6: Research Questions — RQ1, RQ2, three target models. No image.
- Slide 7: The Core Method: Logit Extraction — bullet on model.forward() vs model.generate(), then Equations 3.1, 3.2, 3.3 formatted clearly. No image.
- Slide 8: Prompt Structures & Modality Conditions — 5 structures, 3 conditions, inference counts. Two image placeholders: [INSERT: gray patch PNG — experimental stimulus] and [INSERT: Gaussian noise PNG — experimental stimulus].
- Slide 9: Phase 2 & CALM — 3 bullets from outline. No image.
- Slide 10: Pipeline: What Shipped — bullet list from outline. Clearly state: 11B dev run completed 5 May 2026, 90B/72B pending. No image.
- Slide 11: Engineering Discoveries — exactly 3 bullets (GPFS mmap, transformers pin, bitsandbytes pin). No image. Tight, no explanation text.
- Slide 12: Gate 1: Stimulus Validation — 3 bullets from outline. Large image placeholder: [INSERT: chart_stimulus_validation.png — matplotlib chart, pre-made, insert as-is]
- Slide 13: Phase 1 Raw Results (Llama-3.2-11B dev run · job 24256941 · 5 May 2026) — 4 bullets from outline. Large image placeholder: [INSERT: chart_raw_scores.png — matplotlib chart, pre-made, insert as-is]
- Slide 14: Acquiescence Bias: The Confound — 3 bullets from outline. Large image placeholder: [INSERT: chart_acquiescence.png — matplotlib chart, pre-made, insert as-is]
- Slide 15: Corrected Metric: The Surviving Signal — 3 bullets from outline. Large image placeholder: [INSERT: chart_corrected.png — matplotlib chart, pre-made, insert as-is]
- Slide 16: Next Steps — 2 bullets (90B/72B immediate, Phase 2 after). No image.
- Slide 17: Timeline + Acknowledgements — timeline comparison bullet, acknowledgement of Prof Demartini and SIGIR 2018 [16]. Image placeholders: [INSERT: gianluca.jpeg — Prof Demartini headshot] and [INSERT: uqlogo.png — UQ logo].

**Image rules:**
- Items marked [INSERT: filename] are real images the presenter will drop in manually — leave a clearly visible, labelled placeholder box in that position.
- The 4 chart PNGs on slides 12–15 are pre-made matplotlib figures. Leave large placeholder boxes for them — do not attempt to recreate charts as slide graphics.
- Acknowledgement of Country on slide 2 must be full-bleed with no text on top.

**Do not:** add extra slides, invent results, use stock AI imagery for content areas, add speaker notes, or change any numbers or equations from the source documents.
