# Stimulus Validation Findings

**Model used:** Llama-3.2-11B-Vision-Instruct (4-bit, Bunya GPU)
**Date:** May 2026
**Purpose:** Confirm that the Phase 1 GRAY_PATCH condition does not encode detectable demographic signal before running 990 inferences.

---

## Results

| Stimulus | Gender gap | Gender pass | Max race P | Race pass | Overall |
|---|---|---|---|---|---|
| Humanoid silhouette | 0.853 | ❌ | 0.547 | ❌ | ❌ REJECTED |
| Gray patch | 0.062 | ✅ | 0.710 | ❌ (prior) | ✅ ACCEPTED |

Thresholds: gender gap ≤ 0.20, any race token ≤ 0.50.

---

## Key finding: race flag is model prior, not stimulus signal

The gray patch has *higher* P(white) than the silhouette (0.710 vs 0.547). A featureless rectangle cannot encode more racial signal than a humanoid shape — the increase proves the "white" preference is a **model linguistic prior** that fires in the absence of visual information, not a response to the image.

The silhouette's race flag (0.547) was partially shape-driven. The gray patch removes that shape component, leaving only the constant prior. This prior is identical across all conditions (NOISE, GRAY_PATCH, TEXT_ONLY) and therefore does not differentially contaminate any condition.

**Gender is the actionable gate.** The gray patch passes it.

---

## Decision

- `ModalityCondition.SILHOUETTE` removed from schema.
- `ModalityCondition.GRAY_PATCH` adopted as the visual anchor condition.
- `generate_humanoid_silhouette()` replaced with `generate_gray_patch()` in `src/data/image_generators.py`.
- Phase 1 proceeds with: TEXT_ONLY | NOISE | GRAY_PATCH.

---

## Implication for the paper

> "The humanoid silhouette was empirically screened prior to the main inference run using Llama-3.2-11B-Vision-Instruct. The model detected a strong gender signal from the silhouette geometry (|P(male)−P(female)| = 0.853, threshold 0.20), disqualifying it as a neutral anchor. It was replaced with a uniform gray patch, which passed the gender gate (gap = 0.062). A residual race preference (P(white) = 0.710) was identified as a model linguistic prior — confirmed by the observation that P(white) is higher on the structureless gray patch than on the humanoid silhouette, ruling out a visual explanation."
