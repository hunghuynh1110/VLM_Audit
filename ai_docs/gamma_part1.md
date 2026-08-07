# Gamma Prompt — Part 1 (Slides 1–9, initial generation)
### Before pasting:
- Drag amazon2.jpeg, gender-flip.webp, uqlogo.png into the prompt box
- Use "Thesis Defense" template
- Copy everything below the line

---

Create a 9-slide academic progress seminar presentation. This is a mid-project progress update, not a thesis defense. Use the Thesis Defense template layout. Do not use stock or AI-generated images for content areas.

---

## Slide 1 — Title

**Auditing Gender Bias in Vision-Language Models via Implicit Logit Extraction**

Gia Hung Huynh · Student No. 49384848
Supervisor: Prof Gianluca Demartini
REIT4841 Progress Seminar · The University of Queensland

[drag uqlogo.png here — top corner]

---

## Slide 2 — Acknowledgement of Country

[PLACEHOLDER — full-bleed image. Insert Acknowledgement.jpg manually after export. No text overlay. No modifications.]

---

## Slide 3 — Why This Project

- Safety-aligned models sanitise their *outputs* — but nobody has measured the underlying distribution before that filter fires. Existing audits measure the mask, not the face.
- Prof Demartini co-authored SIGIR 2018 — a study measuring human gender bias in image search. This project asks whether the VLMs now used in those same pipelines carry the same bias.

---

## Slide 4 — What Is Ambivalent Sexism?

**Hostile Sexism (HS) — overt**

- **Apple Card / Goldman Sachs (2019):** Algorithm gave male applicants significantly higher credit limits than female spouses — even when her credit score was higher. Support response: "IT'S JUST THE ALGORITHM."
  [PLACEHOLDER — insert applecard.jpg manually]

- **Amazon's 2018 hiring tool:** Downranked CVs containing the word "women's." Learned from 10 years of male-dominated hiring data.
  [drag amazon2.jpeg here — Reuters headline screenshot]

**Benevolent Sexism (BS) — superficially positive**

- **Google Translate:** Turkish "o bir doktor" is gender-neutral. Old default: "he is a doctor." No gender in the source — the model inserted it.
  [drag gender-flip.webp here — Before/After screenshot]

- **Image captioning (Lu et al. 2018):** Woman in lab coat → "a woman looking at a microscope." Man → "a scientist conducting research."

> BS hides behind positive framing but predicts the same real-world harm as HS. The ASI (Glick & Fiske 1996) captures both.

---

## Slide 5 — Why This Is Hard to Measure

- **Algorithmic Defensiveness:** Safety-aligned models detect evaluative framing and return sanitised outputs — even when the latent distribution is biased.
- **LLM-as-a-Judge** introduces its own biases (positional, egocentric, bandwagon) — you cannot use the model to audit itself.
- This motivates logit extraction as the novel methodological contribution.

---

## Slide 6 — Research Questions

**RQ1:** What does the model actually believe — extracted from its internal probability distribution, before safety training intervenes?

**RQ2:** Does that hidden attitude show up in how the model ranks real-world search results?

Three target models: Llama-3.2-Vision (90B) · Qwen2-VL (72B) · GPT-4o

---

## Slide 7 — The Core Method: Logit Extraction

- Call `model.forward()`, not `model.generate()` — captures P(yes) and P(no) before sampling. Safety overlay never fires.

`P(yes | x, v) = exp(z_yes) / (exp(z_yes) + exp(z_no))` — Eq. 3.1

`b_i = p_i × P(yes | x_i, v)` where `p_i ∈ {+1, −1}` — Eq. 3.2

`ASI_intrinsic = (1 / |ASI|·T·C) × ΣΣΣ b_{i,t,c}` — Eq. 3.3

---

## Slide 8 — Prompt Structures & Modality Conditions

- 5 structures × 3 conditions = 15 probe variants per ASI item
- Structures: Direct · Inversion · Attribution · Hypothetical · Descriptive
- Conditions: Text-Only · Gaussian Noise · Gray Patch
- Per model: 22 × 5 × 3 = **330 inferences** · Across 3 models: **990 Phase 1 total**

[PLACEHOLDER — insert gray patch PNG and Gaussian noise PNG manually]

---

## Slide 9 — Phase 2 & CALM

- Phase 2 (not yet started): 10 SIGIR 2018 queries, two streams (2018 archive + 2026 re-crawl), benchmarked vs human raters from SIGIR 2018
- CALM positional shuffling isolates vision encoder: `ΔRR = RR_vision − RR_text-only`
- Total across both phases: **3,150 inferences**
