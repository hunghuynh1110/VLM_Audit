# Results archive

Every experimental run, kept whether it worked or not. Invalid runs are evidence
for the methodological findings and must not be deleted.

**Directory naming:** `<date>_<phase>_<model>_<precision>_<gpus>_job<slurm_id>_<STATUS>[-<reason>]`

**Status meanings**

| status | means |
|---|---|
| `VALID` | trustworthy; safe to cite |
| `FLAWED-<reason>` | ran correctly but measured the wrong thing; superseded by a fixed rerun |
| `INVALID-<reason>` | output is not a measurement at all; never cite a number from it |
| `FAILED-<reason>` | crashed; kept for the error signature |
| `SUPERSEDED` | early smoke, replaced by a later run |

Each directory holds the run's `parquet`, `checkpoint.jsonl`, `summary.json`
where they exist, plus `logs/` with the SLURM `.out` and `.err`.

**The single most important column is `captured_mass`** — the share of the
model's full next-token distribution sitting on the tokens being measured. Every
flaw below was caught by it, and none of them raised an error.

---

## Current best results

| what | directory |
|---|---|
| **Phase 1, 11B** | `2026-08-08_phase1_11B_bf16_1gpu_job27104037_VALID` |
| **Phase 2, 11B** | `2026-08-09_phase2_11B_bf16_1gpu_job27105702_VALID` |
| **Phase 1/2, 90B** | none yet — see invalid runs below |

Headline numbers from those two: `ASI_intrinsic` raw +0.0835, corrected −0.042,
**HS−BS gap 0.151**; Phase 2 `δ_m` −1.02 (but ~80% of that is a response-scale
prior — see below), `RR_vision` 0.022, `RR_text_only` 0.000.

---

## Every run

### `2026-05-03_gate1_stimulus-validation_11B_VALID`
Gate 1. Humanoid silhouette **rejected** — gender gap 0.853 against a 0.20
threshold. Gray patch **accepted** at 0.062. This is why the stimulus is a flat
gray rectangle.

### `2026-05_phase1_11B_smoke_job24230930_SUPERSEDED`
First working smoke after the 3-day GPFS/`transformers` 5.x debugging. Five
prompts. Predates the inversion sign fix, so its `asi_intrinsic` of 0.6392 is
wrong.

### `2026-05-05_phase1_11B_bf16_1gpu_job24256941_FLAWED-lowercase-tokens`
The original full 330-prompt run. **Every number in the progress seminar came
from here.** Measured only lowercase `yes`/`no`, which hold **0.09%** of the
distribution — the model replies in chat format and emits capitalised `Yes`/`No`
(`'No'` is ~457× likelier than `'no'`). Inflated `p_yes` by ~0.17 systematically.
Reported `ASI_intrinsic` +0.1251, HS +0.209, BS +0.041.

### `2026-08-07_preflight_90B_8bit_2xH100_job27086002_FAILED-bitsandbytes-int8`
Gate 2. **Proved the 90B fits**: loaded 8-bit in 154 s, 85.25 GB peak across
2×H100. Then died in the bitsandbytes int8 kernel (`ops.py:34`, 4-D vision
tensors). Log only — never produced output.

### `2026-08-07_smoke_11B_8bit_job27087751_FAILED-bitsandbytes-int8`
8-bit on the 11B. Died at `ops.py:145` (`.view()` on a non-contiguous tensor).
**SLURM reported COMPLETED** because the script lacked `set -e` — which is why
every job script now has it.

### `2026-08-07_smoke_11B_4bit_job27103376_VALID`
4-bit works (avoids the int8 kernels entirely). Also measured the quantisation
cost: mean |Δp_yes| 0.211 vs bf16 on identical prompts.

### `2026-08-08_diagnostic_quantisation-sweep_11B_job27103565_VALID`
The run that found the surface-form bug. Five quantisation variants; per-prompt
top-10 tokens and per-surface-form mass. Established `captured_mass` = 0.0877%,
that 41% of mass sits on `'I'` (the model starting *"I cannot determine…"*), that
`lm_head` must be excluded from quantisation, and that skipping the vision tower
fixes the 8-bit crash. Δ vs bf16: 8-bit 0.065, 4-bit 0.107.

### `2026-08-08_phase1_11B_bf16_1gpu_job27104037_VALID` ✅
Phase 1 rerun with surface-form pooling (`yes`/`Yes`/`YES` vs `no`/`No`/`NO`).
Captured mass rose 0.09% → 47%. **The HS−BS gap survived at 0.151** (was 0.169),
and strengthens to 0.266 on rows with `captured_mass` ≥ 0.5. Also showed
acquiescence was overstated (mean p_yes 0.70 → 0.527) and that **images trigger
refusal**: captured mass 0.98 text-only vs 0.24 gray-patch vs 0.20 noise.

### `2026-08-09_phase2_11B_bf16_1gpu_job27105400_FLAWED-prefix-in-user-turn`
First Phase 2. `"Rating: "` sat at the end of the *user* message, but the chat
template closes that turn and opens an empty assistant turn — so the model began
a fresh reply and the digit was never the next token. `captured_mass` **0.43%**.
Reported δ_m −1.2018 and `RR` = 1.0, both meaningless.

### `2026-08-09_phase2_11B_bf16_1gpu_job27105702_VALID` ✅
Phase 2 with `"Rating: "` prefilled into the assistant turn. `captured_mass`
**0.9944** (min 0.9545). δ_m −1.0216, `RR_vision` 0.0222, `RR_text_only` 0.0000,
ΔRR +0.0222.

> **Read δ_m carefully.** Split by scale order it is −1.92 (original) vs −0.12
> (reversed): the model outputs ~3 regardless of which end of the scale means
> "objective". A model with a pure low-digit prior and zero anchor sensitivity
> yields an order-averaged normalised rating of exactly 4.0, i.e. a null δ_m of
> 4.000 − 5.280 = **−1.280**. Observed is −1.022, so the genuine signal is only
> **+0.26**. CALM's scale reversal is the only reason this is detectable.

### `2026-08-09_phase1_90B_bf16_3xH100_job27105190_INVALID-multigpu-zerocopy` ❌
### `2026-08-09_phase2_90B_bf16_3xH100_job27105703_INVALID-multigpu-zerocopy` ❌
Both completed with exit 0 and no warnings, and both output **pure noise**.

Bunya's PCIe GPU nodes silently zero-fill every cross-GPU copy: `x.to("cuda:1")`
returns all zeros while `can_device_access_peer()` reports `True`. Sharded across
3 GPUs, activations crossing a device boundary arrive as zeros; a zero hidden
state through the final RMSNorm and `lm_head` gives exactly-zero logits and a
perfectly uniform softmax.

Signatures: Phase 2 token probabilities all exactly `7.79690617491724e-06`
(= 1/128256 = 1/vocab_size); Phase 1 `p_yes` exactly 0.500000 on 247 of 330 rows;
`captured_mass` ~4.5e-05.

**Never cite `ASI_intrinsic` +0.1493 or the apparent HS/BS reversal from these.**
Not a model finding — a platform fault. Reproduced on a 21 GB model with verified
weights, on torch 2.7.1/2.8.0/2.11.0, on A100 and H100. Fix in
`src/models/p2p_workaround.py`; report drafted in
`ai_docs/rcc_p2p_bug_report.md`.

---

## Confound worth remembering

Every 11B run used `--gres=gpu:l40:1` (single GPU); every 90B run used 3×H100.
**Model size and GPU count were perfectly confounded across the entire
experiment history**, which is why "the same code works on the 11B" looked like a
control and wasn't. Forcing the *11B* across 3 GPUs reproduced the failure
exactly.

## Related documents

- `progress/audit_2026-08-08.md` — full audit narrative
- `progress/findings_summary.md` — findings 8, 9, 10
- `ai_docs/investigation_90b_uniform_logits.md` — the handoff brief
- `ai_docs/rcc_p2p_bug_report.md` — draft report to UQ RCC
