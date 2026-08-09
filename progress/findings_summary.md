# Findings summary

Distilled empirical claims with links to evidence. One claim per heading.
Update when understanding shifts; don't accumulate stale claims.

---

## 1. GPFS + safetensors mmap is unusable on Bunya for large model loads

**Evidence:** `findings/bunya_queue_wait_log.md`, smoke jobs `24190012` and `24190558` both hung 2+ hours at the weight-loading step despite 216 MB/s sequential read speed (`dd` benchmark from login node).

**Mechanism:** `safetensors` defaults to `mmap()`. On GPFS each 4 KB page fault is a network round-trip. A 23 GB 11B model = ~6M pages → ~100 min cold-cache load.

**Mitigation:** pre-copy model weights to node-local `$TMPDIR` before `from_pretrained`. Implemented in `scripts/bunya_smoke.sh`, `scripts/bunya_preflight_90b.sh`, `scripts/bunya_run_phase1.sh`.

**Implication for 90B:** 90 GB copy ≈ 7 min at 200 MB/s. Bake into walltime budget.

---

## 2. transformers 5.x changes default loader behavior

**Evidence:** smoke job log showed `transformers_version: "5.7.0"` and `TokenizersBackend(...)` (a 5.x abstraction). Loader hung where the same code path on transformers 4.x had previously worked (silhouette test).

**Mitigation:** pin `transformers>=4.45.0,<5.0` in `requirements.txt`. Commit `7f8754c`.

**Lesson:** open-ended `>=` pins on a major-version-eve package cost a day of debugging. Audit other open pins.

---

## 3. Llama-3.2-Vision native dtype is bfloat16, not fp16

**Evidence:** model config in smoke log shows `"dtype": "bfloat16"`. Code was passing `torch_dtype=torch.float16`, forcing a conversion per shard.

**Mitigation:** switched loader to `torch.bfloat16`. Commit `f89c796`. No accuracy cost; faster load.

---

## 4. Noise-condition anomaly was item-1-specific, not a general stimulus problem

**Evidence:** full 330-prompt run on Llama-3.2-11B (job 24256941, 2026-05-05). Per-condition mean p_yes across all 22 items × 5 structures:

| condition | mean p_yes |
|---|---|
| text_only | 0.66 |
| gray_patch | 0.66 |
| noise | 0.69 |

Compared to the dramatic 0.5 → 0.9 gap seen on item 1 alone in the smoke test (n=5), the full-dataset gap is ~0.03 — within noise. **The smoke-test anomaly was item-1 content driving high agreement, not the noise stimulus contaminating measurement.**

**Status:** RESOLVED. Multi-seed re-run no longer needed — dropped from next_steps.

---

## 4b. Polarity sign on inversion items was a bug (now fixed)

**Evidence:** runner used `bias_score = polarity * p_yes` regardless of structure. For inversion ("would it be incorrect to say...") "yes" means *disagreement* with the trait, so the sign needs to flip.

**Resolution:** added `structure_sign = -1 if INVERSION else 1` to `src/phase1/runner.py`. Smoke-test `asi_intrinsic = +0.6392` (buggy) → full-run `asi_intrinsic = +0.1251` (correct). The 5x reduction is the bug fix; remaining signal is real.

**Lesson:** a metric formula bug is silent — outputs still parsed and aggregated cleanly. Caught only by structure-level inspection (inversion mean dropped to –0.37 after fix, confirming the flip is now correct).

---

## 4c. Acquiescence bias dominates the Llama-3.2-11B Phase 1 signal

**Evidence:** full 330-prompt run (job 24256941). Mean p_yes per structure:

| structure | mean p_yes | median p_yes |
|---|---|---|
| descriptive | 0.793 | 0.893 |
| inversion | 0.788 | 0.852 |
| attribution | 0.648 | 0.692 |
| direct | 0.602 | 0.658 |
| hypothetical | 0.567 | 0.600 |

The model says "yes" 57–79% of the time across **every** structure — including reverse-coded items (6, 7, 9, 15, 20, 22) where "yes" endorses an *anti-sexist* statement (mean bias_scores: –0.23 to –0.59). The model agrees with whatever it is shown.

**Implication:** a yes/no probe with a ~0.7 baseline yes-rate cannot cleanly distinguish "model holds sexist beliefs" from "model says yes a lot." The reported `asi_intrinsic = +0.125` is small *and* confounded.

**Two RLHF guard-rail items.** Item 12 ("set on a pedestal", bias = –0.019) and item 14 ("deserving financial sacrifice from a male partner", bias = +0.025) are near zero — model declines to endorse these textbook benevolent-sexism tropes despite acquiescence baseline. Plausibly hit by safety RLHF.

**HS > BS gap is real but small.** HS = +0.21, BS = +0.04 overall. Restricted to polarity-+1 items only, HS ≈ +0.43 vs BS ≈ +0.24.

**Three forward paths considered:**
1. **(a) Acquiescence-controlled metric** — recompute `bias_score` relative to per-structure baseline yes-rate. Cheap; same parquet. ✓ implemented in `scripts/analyze_phase1.py`.
2. **(b) Forced-choice prompts** — "Is X more likely Y or Z?" eliminates yes-bias by construction (StereoSet, CrowS-Pairs, BBQ literature). Requires Phase 1 redesign + re-run.
3. **(c) Cross-model comparison** — same probes on Qwen2-VL and Llama-90B; differences across models are interpretable signal because method effect partially cancels. Aligned with VHELM / OpenVLM Leaderboard practice.

**Decision:** going with **(a) + (c)** — apply acquiescence correction now, then run cross-model comparison on Bunya. (b) deferred unless reviewers push back.

**Status:** RESOLVED as a confound. Survives as a methodological finding to report.

---

## 4d. Acquiescence-corrected results: HS vs BS subscale gap is the surviving signal

**Evidence:** `scripts/analyze_phase1.py` on `outputs/phase1/llama_dev.parquet` (n=330). Three correction variants computed (per-structure mean subtraction, per-(structure×condition) mean subtraction, log-odds version); all three give the same qualitative pattern.

**Naive vs corrected comparison:**

| Aggregate | raw | per-structure adj | log-odds adj |
|---|---|---|---|
| **Overall ASI** | +0.125 | **−0.040** | −0.289 |
| direct | +0.202 | −0.072 | −0.558 |
| descriptive | +0.328 | −0.032 | −0.046 |
| attribution | +0.258 | −0.037 | −0.246 |
| hypothetical | +0.208 | −0.050 | −0.361 |
| **inversion** | **−0.371** | **−0.012** | −0.236 |
| noise (cond) | +0.165 | −0.000 | −0.042 |
| gray_patch | +0.119 | −0.046 | −0.386 |
| text_only | +0.091 | −0.075 | −0.440 |
| **HS subscale** | **+0.209** | **+0.044** | **+0.263** |
| **BS subscale** | **+0.041** | **−0.125** | **−0.842** |

**What collapsed under correction:**
- The dramatic **inversion outlier (−0.371)** disappears (−0.012). It was acquiescence × sign-flip, not a structural insight.
- The **noise vs gray_patch / text_only gap** disappears. Noise lands closest to zero — validates the stimulus design as a bias-free baseline.
- The **per-structure variation** flattens (all five structures within ±0.07).

**What survived correction (the real finding):**
- **HS − BS gap ≈ 0.17 in the per-structure adjusted metric, and is robust to log-odds correction.** BS goes meaningfully *negative* (model resists endorsing benevolent sexism); HS stays slightly positive.
- Interpretation: RLHF safety training catches obvious benevolent-sexism tropes ("set on a pedestal", "deserving financial sacrifice from a male partner") but is weaker on subtler hostile items ("exaggerating problems at work", "too easily offended"). Items 12 and 14 (BS, near-zero raw scores) corroborate this.

**Implication for cross-model comparison:** the corrected metric is what should be reported across Qwen / 90B / etc. Per-model acquiescence baselines must be reported alongside, since acquiescence won't cancel cleanly across model families.

**Status:** RESOLVED for 11B. Pending replication on 90B and Qwen2-VL.

---

## 5. Bunya `gpu_cuda` queue wait dominates dev iteration cost

**Evidence:** `findings/bunya_queue_wait_log.md`. Smoke job submitted ~15:06 AEST 2026-05-05, scheduled to start 19:52:41 AEST. ~4 h 46 min wait.

**Reason:** `(Priority)` — fairshare queue, not resource-limited.

**Implication:** every debug iteration costs ~5 hours regardless of actual compute. Plan jobs to do as much as possible per submission.

---

## 6. bitsandbytes 0.44.x is broken on torch 2.11 / triton 3.6

**Evidence:** `ImportError: No module named 'triton.ops'` on `import bitsandbytes`. `triton 3.x` removed the `triton.ops` namespace.

**Mitigation:** pin `bitsandbytes>=0.45.0`. Commit `b8312e7`.

---

## 7. Bunya `/tmp` quota is 200 MB — too small for pip on torch + CUDA libs

**Evidence:** pip install of torch (530 MB) fails with `OSError: [Errno 122] Disk quota exceeded` on login node.

**Mitigation:** redirect with `TMPDIR=/QRISdata/Q9468/tmp PIP_CACHE_DIR=/QRISdata/Q9468/pip_cache` before any large pip install on Bunya. Persist in `~/.bashrc` if needed.

---

## 8. Surface-form artifact — corrected, and the HS/BS gap replicates

**Evidence:** job `27104037` (330 rows, 11B, bf16, case-pooled) vs the archived
lowercase-only run `24256941` (`outputs/phase1_lowercase_v1/`).

The probe previously read only lowercase `yes`/`no`, which hold ~0.09% of the
next-token distribution. Pooling `{yes,Yes,YES}` vs `{no,No,NO}` raises the
measured share to **47% on average** — a ~500x increase in captured mass.

| metric | lowercase-only | case-pooled |
|---|---|---|
| ASI_intrinsic (raw) | +0.1251 | **+0.0835** |
| ASI_intrinsic (per-structure adj) | −0.040 | **−0.042** |
| HS (adj) | +0.044 | **+0.033** |
| BS (adj) | −0.125 | **−0.117** |
| **HS − BS gap** | **0.169** | **0.151** |

**The central finding survives.** The corrected overall ASI is essentially
unchanged (−0.040 → −0.042) and the HS−BS gap moves only 0.169 → 0.151. It is
also robust to filtering on measurement quality:

| captured_mass ≥ | n | HS adj | BS adj | gap |
|---|---|---|---|---|
| 0.00 | 330 | +0.033 | −0.117 | 0.151 |
| 0.10 | 284 | +0.029 | −0.115 | 0.144 |
| 0.30 | 166 | +0.019 | −0.154 | 0.174 |
| 0.50 | 113 | +0.084 | −0.182 | 0.266 |

The gap never collapses and strengthens on the highest-quality rows.

**Status:** Finding 4d CONFIRMED under corrected measurement.

---

## 9. Acquiescence was real but substantially overstated (revises 4c)

Mean p_yes per structure, before and after the surface-form fix:

| structure | lowercase-only | case-pooled |
|---|---|---|
| descriptive | 0.793 | 0.650 |
| inversion | 0.788 | 0.628 |
| attribution | 0.648 | 0.503 |
| direct | 0.602 | 0.456 |
| hypothetical | 0.567 | 0.400 |

Overall mean p_yes falls from ~0.70 to **0.527**. Three of five structures now
sit at or below 0.5. The claim "the model says yes to whatever it is shown" does
not hold as stated — much of it was the lowercase tail, not the model.

Acquiescence correction still matters (raw +0.0835 → adjusted −0.042), but
Finding 4c must be reworded: a modest yes-lean, not a dominant one.

---

## 10. Visual input makes the model refuse the yes/no frame (new)

`captured_mass` by modality condition (job 27104037):

| condition | mean captured_mass | rows under 10% captured |
|---|---|---|
| text_only | **0.979** | 0.0% |
| gray_patch | 0.237 | 12.7% |
| noise | 0.204 | 29.1% |

With no image the model answers yes/no cleanly 98% of the time. **Add any
image — even a featureless gray patch — and ~80% of the probability mass moves
elsewhere**, typically onto `'I'` (the model beginning "I cannot determine…").
Gaussian noise provokes the most refusal.

**Two implications.**
1. This is algorithmic defensiveness measured directly, and it is triggered by
   the *presence* of visual input rather than by image content. That is on-thesis
   for RQ1 and is a result in its own right.
2. p_yes is not comparable across conditions without care: in the vision
   conditions it is conditioned on ~20% of the model's behaviour versus ~98% in
   text-only. Cross-condition contrasts must report captured_mass alongside.

---

## 11. The 90B was never broken: cross-GPU copies on Bunya deliver zeros (new)

The 90B's perfectly uniform output (`p = 1/128256` for every token, job 27105703;
`p_yes` pinned at exactly 0.5 on 247/330 rows, job 27105190) was not a 90B
problem, not a checkpoint problem, and not a code problem. **On Bunya's PCIe GPU
nodes a direct GPU-to-GPU copy does not move the data — the destination reads
back as all zeros, silently.**

```
src (on cuda:0)  : [0, 1, 2, 3, 4, 5, 6, 7, ...]
src.to("cuda:1") : [0, 0, 0, 0, 0, 0, 0, 0, ...]
n_zero in dst    : 1048576 / 1048576
```

`torch.cuda.can_device_access_peer()` returns True for every pair, so torch takes
the peer path and the copy is lost. No error, no warning.

### Why it looked like a 90B problem

Every 11B run used `--gres=gpu:l40:1` (`device_map={'': 0}`); every 90B run used
3×H100. **Model size and multi-GPU sharding were perfectly confounded across the
entire experiment history**, so "the same code works on the 11B" appeared to
exonerate the code and implicate the 90B checkpoint. It did neither. Forcing the
*11B* across 3 GPUs reproduces the failure exactly (job 27113606), on a 21 GB
model whose weights are demonstrably fine.

### Mechanism, end to end

`device_map="auto"` balances a model across every visible GPU, and accelerate
moves activations across device boundaries between layers. Each crossing
delivered zeros. Two fingerprints in the old data follow directly:

- A zero hidden state through the final RMSNorm and `lm_head` gives **exactly
  zero logits**, hence a perfectly uniform softmax — the Phase 2 signature.
- Where activations instead exploded, `hidden.pow(2)` overflows fp32,
  `rsqrt(inf)` is 0, and the state is zeroed anyway; where it reached `inf`,
  `inf * 0` gives the **NaN** rows seen in Phase 1.

It also explains the otherwise impossible detail in the layer trace: NaNs
produced on `cuda:0` were *absent* from the tensor read on `cuda:1`. Values do
not un-corrupt themselves — they were replaced by zeros in transit.

### Not a library version

Identical zero-fill under torch **2.7.1+cu126, 2.8.0+cu128 and 2.11.0+cu130**
(driver 595.58.03, CUDA 13.2), on both A100 and H100 PCIe nodes.
`NCCL_P2P_DISABLE=1` makes no difference — it governs NCCL, not the plain
`Tensor.to()` path accelerate uses. **This is the machine, and it should be
reported to UQ RCC.** Host↔device copies are unaffected.

### Fix

`src/models/p2p_workaround.py` routes every `cuda->cuda` copy through host
memory, which is the path that works. Only tensors crossing a device boundary
pay the cost — activations of a few MB, not the weights. `LlamaExtractor`
probes the node with `is_affected()` and enables it automatically whenever more
than one GPU is visible.

Validated on the 11B in a single job (27113674), all three arms on one node:

| arm | logits std (P1 text / P1 image / P2 digits) | P2 captured_mass |
|---|---|---|
| A — single GPU (ground truth) | 2.13938 / 2.10143 / 1.66373 | 0.981027 |
| B — sharded, no workaround | **0 / 0 / 0** | 5.45783e-05 (= uniform) |
| C — sharded + workaround | **2.13938 / 2.10143 / 1.66373** | **0.981027** |

Arm C is bit-identical to arm A. The workaround restores the reference result
exactly rather than merely producing something non-degenerate.

### What this invalidates

- **Invalid:** every 90B run to date. Archived as
  `outputs/phase{1,2}/*.invalid_multi_gpu.*` — evidence, not results.
- **Unaffected:** all 11B results, including Findings 4c and 10. Every 11B job
  ran on a single L40, so no cross-GPU copy ever occurred.
- **Watch:** any future model needing more than one GPU — Qwen2-VL-72B included.

### Method note

`captured_mass` did not catch this one on its own: the failing runs reported
plausible aggregates (`rating_expected` = 4.000, the exact midpoint of a 1–7
uniform). What exposed it was that **330 distinct prompts produced only 8
distinct values** — output carrying no information about its input. Worth
adding to the standard checks: a degenerate run can look calibrated in the mean.
