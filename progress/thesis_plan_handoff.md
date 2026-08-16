# Thesis plan handoff — project state as of 2026-08-15

Self-contained briefing for drafting the thesis plan. Every number below comes
from a completed run; nothing here is projected or estimated.

---

## 1. Project identity

| | |
|---|---|
| **Title** | *Auditing VLM Perception of Gender Bias using Human Sexism Scales in Image Search* |
| Author | Gia Hung Huynh |
| Supervisor | Prof. Gianluca Demartini |
| Course / institution | REIT4841, School of EECS, The University of Queensland |
| Proposal submitted | 23 April 2026 |
| Status | Data collection ~2/3 complete; analysis and write-up outstanding |

## 2. Research questions

- **RQ1** — What is the intrinsic psychological baseline attitude of the model
  when evaluated in a vacuum?
- **RQ2** — Is that intrinsic distributional skew descriptively consistent with
  the model's extrinsic, downstream behaviour when evaluating real-world search
  queries?

## 3. The gap the project addresses

Vision-language models are deployed as automated judges in retrieval and content
moderation, and inherit gender stereotypes from visual training distributions.
The barrier to auditing them is **algorithmic defensiveness**: safety-aligned
models detect evaluative framing and emit sanitised text, so audits based on
model self-report cannot separate genuine debiasing from defensive evasion.

**The contribution is methodological**: measure the distribution *before* the
safety overlay acts on it, by reading raw next-token logits from a single
forward pass rather than analysing generated text.

## 4. Method

**Implicit logit extraction.** `model.forward()` is called once per stimulus and
the full-vocabulary softmax over the final position is read directly.
`model.generate()` is never used. Target-token probabilities are pooled across
surface forms (`yes`/`Yes`/`YES`) before any ratio is taken.

**`captured_mass`** — the share of the full distribution sitting on the target
tokens — is recorded for every measurement. It is the project's core validity
statistic and has caught three separate measurement bugs (a surface-form bug at
0.09% mass, a chat-template bug at 0.43%, and a silent hardware fault producing
a perfectly uniform distribution).

### Phase 1 — intrinsic (RQ1)
22 Ambivalent Sexism Inventory items × 5 prompt structures (direct, descriptive,
attribution, hypothetical, inversion) × 3 image conditions (text-only, grey
patch, Gaussian noise) = **330 measurements per model**.
`bias_score = structure_sign × polarity × p_yes`; `ASI_intrinsic` is its mean.

### Phase 2 — extrinsic (RQ2)
SIGIR-2018 image-search results rated 1–7 for objectivity against human
baselines from low-ASI raters. 10 queries × 9 images × 2 scale orders × 2
conditions = **360 rows per model**.
- `delta_m` — mean signed deviation from the human low-ASI baseline
- `RR` (Robustness Rate) — share of judgements surviving **CALM scale reversal**
- `ΔRR = RR_vision − RR_text_only`

### Models
| Model | Status |
|---|---|
| Llama-3.2-11B-Vision | Development model, complete |
| Llama-3.2-90B-Vision | **Complete**, both phases, plus an independent replication |
| Qwen2-VL-72B | **Complete**, both phases |
| GPT-4o | **Not started** — no extractor written |

---

## 5. Results

### Phase 1 (intrinsic)

| | Llama-90B | Qwen-72B |
|---|---|---|
| ASI_intrinsic | **+0.1530** | **+0.0078** |
| mean captured_mass | 0.5823 | 0.9550 |
| mean p_yes | 0.5851 | 0.2506 |
| polarity separation | +0.064 | **+0.150** |
| subscale HS / BS | +0.159 / +0.147 | +0.030 / −0.015 |

By prompt structure:

| structure | Llama-90B | Qwen-72B |
|---|---|---|
| attribution | +0.299 | +0.046 |
| descriptive | +0.283 | +0.111 |
| direct | +0.245 | +0.002 |
| hypothetical | +0.157 | −0.003 |
| **inversion** | **−0.219** | **−0.117** |

`captured_mass` by image condition:

| condition | Llama-11B | Llama-90B | Qwen-72B |
|---|---|---|---|
| text_only | 0.979 | 0.951 | 0.952 |
| gray_patch | 0.237 | 0.467 | **0.962** |
| noise | 0.204 | 0.329 | **0.952** |

### Phase 2 (extrinsic)

| | Llama-90B | Qwen-72B |
|---|---|---|
| `delta_m` | **−1.2033** | **−0.6252** |
| RR text-only | 0.900 | 0.900 |
| RR with image | **0.044** | **0.156** |
| ΔRR | −0.856 | −0.744 |
| mean captured_mass | 0.9978 | 0.9995 |

Person vs non-person query dissociation:

| | person queries (mean) | non-person control | gap |
|---|---|---|---|
| Llama-90B | −1.333 (range −2.292 … −0.675) | −0.036 | 1.30 |
| Qwen-72B | −0.738 (range −1.244 … +0.126) | +0.386 | 1.12 |

---

## 6. What can be claimed

**Replicates across architectures (strongest results):**
1. **Visual input collapses scale-reversal robustness.** 0.900 text-only versus
   0.044 (Llama) and 0.156 (Qwen). Two independently trained models, same
   failure — a property of VLM visual grounding, not one model's quirk.
2. **Both models rate images of people as less objective than human raters do**,
   while the non-person control sits ~1.1–1.3 points higher. The dissociation
   rules out generic harshness or scale miscalibration.

**Dissociates between architectures:**
3. **Algorithmic defensiveness is Llama-specific.** Llama's `captured_mass`
   collapses when any image is present (0.95 → 0.33–0.47), including a
   featureless grey patch; Qwen holds ~0.95 across all conditions. This is a
   property of Llama's safety training, not of VLMs generally.

**Methodological finding:**
4. **`ASI_intrinsic` is confounded with response bias.** Because
   `bias_score` scales with `p_yes`, a model that says "no" to everything scores
   ~0 regardless of its actual attitudes, and one that says "yes" to everything
   scores by polarity balance. Llama's +0.153 is inflated by acquiescence (both
   polarities above 0.5); Qwen's +0.008 is deflated by nay-saying (mean 0.25).
   The **polarity separation** (difference in mean `p_yes` between
   positively- and reverse-scored items) is response-bias invariant, and by it
   Qwen discriminates 2.3× more strongly than Llama. The conclusion is *not*
   "Qwen is less biased" — it is that **Qwen answers the inventory more
   discriminatingly while Llama acquiesces**.

---

## 7. Threats to validity — must appear in the plan

1. **Acquiescence contaminates Phase 1.** The `inversion` structure asks each
   item backwards and should score like the others; instead it flips sign in
   both models (−0.219, −0.117). Part of the headline intrinsic bias is
   yea-saying rather than belief. Report polarity separation alongside.
2. **Text-only Phase 2 has effective n = 10, not 90.** With no image the input
   is fully determined by (query × scale_order), so the 180 text-only rows
   contain only 20 unique measurements, 9-fold duplicated. `RR_text = 0.900` is
   9 of 10 independent query-pairs. The ΔRR contrast survives comfortably, but
   any standard error computed over 90 text-only rows is wrong.
3. **Cross-condition `captured_mass` differs sharply for Llama** (0.95 text vs
   0.33 noise), so its `p_yes` is conditioned on a third of the model's
   behaviour in vision conditions versus nearly all of it in text. Cross-condition
   contrasts must report `captured_mass` alongside.
4. **Quantisation is not acceptable for the primary result** — measured shift is
   0.107 (4-bit) and 0.065 (8-bit) against an effect of ~0.15. All headline runs
   are bf16.
5. **One dataset, one language, two open-weight models.** GPT-4o (closed,
   API-only, no logit access) remains outstanding and may not support the same
   measurement at all — a scope limitation worth stating explicitly.

---

## 8. Infrastructure finding (methods/limitations material)

Mid-project, all Llama-90B runs were found to be returning a perfectly uniform
distribution (p = 1/128256 for every token) while exiting cleanly and reporting
plausible aggregate metrics. Root cause was **not** the model or the code: the
HPC cluster's PCIe GPU nodes were silently zero-filling every GPU-to-GPU copy,
so any model sharded across GPUs received zeros at each layer boundary. Verified
identical under three PyTorch versions on both A100 and H100 hardware.

Mitigation routes cross-GPU traffic through host memory. Validated as
bit-identical to a single-GPU reference. An independent replication on hardware
where the fault had since been repaired reproduced the results without the
workaround active (ASI_intrinsic +0.1530 → +0.1527, r = 0.9996, 220/330 rows
bit-identical), confirming the workaround does not distort the measurement. A
4-bit single-GPU run, structurally incapable of triggering the fault, agrees at
r = 0.767 with the difference matching what quantisation predicts.

**Relevance to the plan:** it justifies the pipeline's built-in degeneracy
self-test, and is a concrete example of why `captured_mass` is reported for
every measurement rather than only aggregate scores.

---

## 9. Remaining work

1. **GPT-4o extractor** — closed model; logit access is limited to top-N
   logprobs via API, so the method may need adaptation or the model may need to
   be dropped. This is the largest open scope question.
2. **Statistical analysis** — permutation test for directional skew against
   chance; per-item `b_i` heatmap; cross-model ranking.
3. **Add polarity separation** as a first-class metric alongside `ASI_intrinsic`.
4. **Recompute Phase 2 statistics** honouring the text-only n = 10.
5. **Figures** for the three headline results.
6. Optional: contemporary image stream (Phase 2 currently uses the historical
   SIGIR-2018 stream only).

## 10. Tools

Python, PyTorch 2.11, HuggingFace Transformers 4.57 / accelerate, safetensors,
bitsandbytes, pandas, matplotlib, pytest; SLURM multi-GPU HPC (UQ Bunya, H100 /
A100, bf16, 166 GB model sharding across 3 GPUs).
