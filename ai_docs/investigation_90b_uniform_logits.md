# Investigation brief — Llama-3.2-90B-Vision returns a uniform distribution

**Paste this whole file into a fresh chat as the opening prompt.**

---

## Your task

Find out why `meta-llama/Llama-3.2-90B-Vision-Instruct` produces a *perfectly
uniform* next-token distribution on our HPC cluster, while the exact same code
produces correct output on `meta-llama/Llama-3.2-11B-Vision-Instruct`.

Decide whether the cause is **our code**, a **library/version conflict**, a
**multi-GPU sharding problem**, or **corrupted staged weights** — then fix it and
prove the fix with a run whose output is non-degenerate.

Do not accept "it ran without error" as success. It already does that.

---

## The symptom

Two SLURM jobs completed cleanly (exit 0, no exceptions, no warnings) and
produced numbers that are pure noise.

**Phase 2 (job 27105703)** — probability of each rating digit, one row:

```
{'1': 7.79690617491724e-06, '2': 7.79690617491724e-06, '3': 7.79690617491724e-06,
 '4': 7.79690617491724e-06, '5': 7.79690617491724e-06, '6': 7.79690617491724e-06,
 '7': 7.79690617491724e-06}
```

All bit-identical. **7.79690617491724e-06 = 1/128256 = 1/vocab_size.** The
softmax is uniform over the entire vocabulary, i.e. the logits are all equal
(almost certainly all zero). The forward pass computes nothing.

**Phase 1 (job 27105190)** — `p_yes` is *exactly* 0.500000 on 247 of 330 rows;
`captured_mass` (share of the distribution on the target tokens) averages
4.5e-05 where the 11B gets 0.47.

Both jobs report plausible-looking aggregate metrics. Only the `captured_mass`
instrumentation revealed the problem.

---

## What is already ruled out

| checked | result |
|---|---|
| Missing / unexpected / newly-initialised weight warnings | **none** in stdout or stderr |
| CPU / disk / meta offload in `hf_device_map` | **none** — all layers on GPU 0/1/2 |
| GPUs visible to the job | 3 × NVIDIA **H100 PCIe**, 81559 MiB each |
| dtype | `Instantiating MllamaForConditionalGeneration model under default dtype torch.bfloat16` (correct — native dtype) |
| Weight staging | `rsync -aL` completed, **44 min** for 166 GB, no error |
| Source shards present | 37/37 resolve, 166 GB total, non-sparse (raw bytes read back as plausible bf16) |
| Same code on the 11B | **works correctly** — jobs 27104037 (Phase 1) and 27105702 (Phase 2), `captured_mass` 0.47 and 0.99 |
| Disk space on scratch | 2.2 PB, 1.3 PB free |

**Important:** the 90B has *never* been observed producing a sane distribution.
An earlier 8-bit preflight (job 27086002) loaded successfully — 85.25 GB peak
across 2 GPUs, 154 s load — but crashed inside the bitsandbytes int8 kernel
before emitting any logits, so it never confirmed correct output either.

---

## Strongest lead: a checkpoint-key / module-name mismatch

The source weights are confirmed healthy (read directly from the shards):

```
language_model.model.embed_tokens.weight      std=0.009094  mean=-0.000014  zeros=0.0%
language_model.model.layers.0.mlp.gate_proj   std=0.065961  mean=-0.000004  zeros=0.0%
language_model.model.layers.10.mlp.down_proj  std=0.011695  mean=-0.000003  zeros=0.0%
```

So the data on disk is fine, and the problem is in **loading**, not storage.

Note the prefixes. The checkpoint stores **`language_model.model.layers.N...`**,
but the runtime `hf_device_map` from the earlier preflight lists modules as
**`model.language_model.layers.N`** — the `model.` prefix sits in a different
place. transformers restructured Mllama's module tree across versions, and if
the key remapping does not fire, parameters stay at their initialised values
while `from_pretrained` reports nothing.

That would produce exactly this symptom: constant logits, uniform softmax, no
error. It is also consistent with the 11B working — the two checkpoints were
exported at different times and may not share a key convention.

**Verify first:**

```python
import safetensors.torch as st
sd_keys = set(st.load_file(shard1, device="cpu").keys())
mod_keys = set(dict(model.named_parameters()).keys())
print("in checkpoint only:", list(sd_keys - mod_keys)[:10])
print("in model only:",      list(mod_keys - sd_keys)[:10])
print("lm_head std:", model.lm_head.weight.float().std())
```

Compare the same output for the 11B (which works) against the 90B (which does
not). If the 90B has a large symmetric difference and the 11B does not, that is
the answer. Check `transformers` release notes for Mllama key renaming and any
`_checkpoint_conversion_mapping` on `MllamaForConditionalGeneration`.

---

## Other hypotheses (not exhaustive, not ranked)

1. **Multi-GPU sharding of Mllama cross-attention.** `device_map="auto"` split
   the model across 3 GPUs (layers 0–42 → GPU0, 43–99 → GPU1 in the 2-GPU
   preflight). Mllama interleaves cross-attention layers; if hidden states or
   the cross-attention mask cross a device boundary incorrectly, output could
   collapse to constant. **Test:** compare against a single-GPU load (4-bit or
   8-bit fits on one H100) — if single-GPU is sane and multi-GPU is uniform,
   this is it.
2. **H100 PCIe without NVLink.** These are PCIe cards, not SXM. Peer-to-peer
   transfer behaviour differs and some accelerate paths assume NVLink.
3. **transformers 4.57.6 + Mllama-90B specific bug.** The 11B works, so any bug
   would be size- or config-specific (the 90B has a different
   `cross_attention_layers` list and `vision_output_dim` 7680). Search GitHub
   issues and HF forums for Llama-3.2-90B-Vision multi-GPU / garbage output.
4. **Silently truncated staged copy.** `rsync -aL` was not checksum-verified.
   `safetensors` would normally error on a truncated file, but verify:
   re-stage with `rsync -aLc` (checksum) or compare per-shard sizes and hashes
   between `/QRISdata/...` and `$TMPDIR` inside the job.
5. **lm_head / tied embeddings not loaded.** Zero `lm_head` weights give exactly
   constant logits. Print `model.lm_head.weight.std()` right after load — if it
   is 0, that is the answer.
6. **bf16 numerical collapse** in the 90B's vision tower or projector.

---

## Suggested first diagnostic (cheap, one GPU, `debug` QOS)

Load the model and inspect it **before** any prompt, so a bad load is separated
from a bad forward pass:

```python
print(model.lm_head.weight.float().std())          # 0 => weights never loaded
for n, p in model.named_parameters():
    s = p.float().std().item()
    if s == 0 or not math.isfinite(s):
        print("DEAD PARAM:", n, p.shape, p.dtype, p.device)
```

Then one forward pass and `outputs.logits[0, -1, :].std()` — 0 confirms
constant logits. Also print `model.hf_device_map` and check which device each
cross-attention layer landed on.

Run the identical script against the **11B as a control** in the same job.

---

## Environment

- **Cluster:** UQ Bunya. Repo at `/QRISdata/Q9468/VLM_Audit` (an RDM collection,
  NFSv4 — *git is pathologically slow there*; a `git pull` once ran >25 min while
  a bare clone to `/scratch` took 1 s. Use `scp`/`rsync` for file transfer).
- **Git remote:** `https://github.com/hunghuynh1110/VLM_Audit` (public).
- **venv:** `/QRISdata/Q9468/VLM_Audit/.venv`, Python 3.11.3
  - `torch 2.11.0+cu130`, `transformers 4.57.6`, `accelerate 1.0.1`,
    `bitsandbytes 0.49.2`
  - Pins in `requirements.txt`: `transformers>=4.45,<5.0`, `accelerate<1.1`,
    `bitsandbytes>=0.45`. These were hard-won — `transformers 5.x` hangs the
    loader on this filesystem. Do not bump casually.
- **Model cache:** `HF_HOME=/QRISdata/Q9468/huggingface_cache` (90B = 166 GB /
  37 shards; Qwen2-VL-72B = 137 GB; 11B = 20 GB in `~/.cache/huggingface`).
- **Weight staging is mandatory:** `safetensors` uses `mmap()`, which is
  pathological over this NFS mount (~100 min for a cold 23 GB load). All SLURM
  scripts pre-copy to `$TMPDIR` first. Budget ~45 min for the 90B.
- **SLURM:** account `a_ai_collab`, partition `gpu_cuda`.
  QOS: `debug` (priority 30, **1 h max**, 4 GPUs — use this for diagnostics),
  `short` (prio 20, 12 h, 2 GPUs), `gpu` (prio 10, no cap, 4 GPUs),
  `viz`, `mig`, `normal` (0 GPUs).
  All 21 H100s are usually allocated; waits can be days. `squeue` hides other
  users' jobs (`PrivateData=jobs`) — use `scontrol show node <n>` and read
  `AllocTRES` to see real contention.
- **Access:** DUO MFA required. The agent cannot type passwords. Ask the user to
  run this once, then reuse the socket:
  ```
  ssh -M -S ~/.ssh/bunya.sock -o ControlPersist=12h -o ServerAliveInterval=60 -fN s4938484@bunya.rcc.uq.edu.au
  ```
  then `ssh -S ~/.ssh/bunya.sock s4938484@bunya.rcc.uq.edu.au '<cmd>'`.
  **Bunya switches DUO → Okta after the 10–11 Aug maintenance.**
- **Gotcha:** a client-side SSH timeout does **not** mean the remote command
  didn't run. This has twice produced duplicate SLURM jobs. Always check
  `squeue` before resubmitting.

---

## Relevant files

| path | what |
|---|---|
| `src/models/llama_extractor.py` | model loading, `extract_probs()`, `_build_inputs()` |
| `src/phase1/runner.py` | Phase 1 loop; writes `captured_mass` |
| `src/phase2/runner.py` | Phase 2 loop |
| `scripts/bunya_run_phase1.sh` | the 90B bf16 3×H100 job that produced the bad output |
| `scripts/diagnose_measurement.py` | existing per-variant diagnostic; good starting point |
| `progress/audit_2026-08-08.md` | full audit history, all job IDs |
| logs | `/QRISdata/Q9468/VLM_Audit/logs/phase1_llama_27105190.{out,err}` and `phase2_90b_27105703.{out,err}` |

Outputs from the bad runs are kept at `outputs/phase1/llama.parquet` and
`outputs/phase2/llama_historical_2018.parquet` — **do not treat these as
results**, they are evidence.

---

## Project context (one paragraph)

Honours thesis auditing gender bias in vision-language models by extracting raw
next-token logits (`model.forward()`, never `generate()`) so the latent
distribution is measured before safety alignment sanitises the text output.
Phase 1 scores 22 Ambivalent Sexism Inventory items × 5 prompt structures × 3
image conditions. Phase 2 rates SIGIR-2018 image-search results 1–7 for
objectivity against human baselines, with CALM scale-reversal controls. The 11B
is the development model; the 90B and Qwen2-VL-72B are the thesis models. Two
measurement artifacts have already been found and fixed by tracking
`captured_mass` (the share of probability mass on the target tokens) — a
surface-form bug (`yes` vs `Yes`, 0.09% mass) and a chat-template bug (the cue
sat in the user turn, so the model started a fresh reply, 0.43% mass). **Always
report `captured_mass`; it is how both bugs were caught, and how this one was.**

---

## Definition of done

1. A named root cause with evidence, not a guess.
2. A fix in the repo (conventional one-line commit messages, no AI attribution).
3. A 90B run whose `captured_mass` is comparable to the 11B's (~0.5 for Phase 1
   yes/no, ~0.99 for Phase 2 digits) and whose `p_yes` is not pinned at 0.5.
4. A short write-up appended to `progress/findings_summary.md`.
