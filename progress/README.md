# Progress journal

Living record of the project: what's been done, what's been learned, what's
pending. Append to this as work happens — don't reconstruct from memory later.

## Files

- [`timeline.md`](timeline.md) — chronological event log (date · event · outcome)
- [`findings_summary.md`](findings_summary.md) — synthesized empirical claims, links to evidence
- [`blockers_log.md`](blockers_log.md) — resolved + open blockers, with root causes
- [`next_steps.md`](next_steps.md) — prioritized work queue with dependencies

## How to maintain

- When you commit something non-trivial, append a line to `timeline.md` in the same change.
- When something blocks you for >30 min, log it in `blockers_log.md` as you debug, not after.
- When a finding shifts your understanding, update `findings_summary.md`.
- When the plan changes, edit `next_steps.md` — don't keep stale priorities.

## Status snapshot

**Current phase:** Phase 1 (intrinsic ASI bias measurement on VLMs).
**Latest milestone:** 2026-05-05 (late eve) — `analyze_phase1.py` applied to 11B parquet. Acquiescence-corrected ASI ≈ −0.04 (vs naive +0.125). Surviving signal: HS subscale +0.044 vs BS −0.125 (gap ≈ 0.17), robust across three correction variants. Likely RLHF asymmetry.
**Immediate next:** scaffold Qwen2-VL extractor for cross-model comparison (next_steps.md #6, now elevated).
