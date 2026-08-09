"""
Root-cause diagnostic for the 90B uniform-logit failure.

The 90B returns a perfectly uniform next-token distribution (p = 1/128256 for
every token) while the 11B, on identical code, is fine. "It ran without error"
is not evidence of anything here, so this script separates the three places the
signal can die and reports each independently:

  A. LOAD    -- did every parameter actually get weights? A tensor left at its
                initialised value, or an all-zero lm_head, gives exactly
                constant logits and transformers reports nothing.
  B. FORWARD -- where in the stack does the activation die? Hooks record the
                std / absmax / %zeros / finiteness of every layer's output, so
                a collapse is pinned to a layer index, which in turn maps onto
                a device boundary in hf_device_map.
  C. OUTPUT  -- final hidden state and logits std. std == 0 confirms constant
                logits rather than a merely bad answer.

It also runs the forward twice, with and without an image. The text-only pass
does not touch the vision tower, the projector, or any cross-attention layer,
so if text-only is sane and the image pass is uniform the fault is confined to
the vision/cross-attention path.

Usage (see scripts/bunya_diagnose_90b.sh):
    python scripts/diagnose_90b_logits.py --variant llama
    python scripts/diagnose_90b_logits.py --variant llama_dev   # control
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import OrderedDict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

import torch
from PIL import Image

from src.config import CFG

_MODEL_IDS = {"llama": CFG["models"]["llama"], "llama_dev": CFG["models"]["llama_dev"]}


# ----------------------------------------------------------------------
# A. load audit
# ----------------------------------------------------------------------

def tensor_stats(t: torch.Tensor) -> dict:
    """
    Cheap summary of a tensor, computed in fp32 on its own device.

    The masked copy is taken only when something is actually non-finite --
    the 90B's lm_head is 1.05 B elements and the GPUs are near full, so an
    unconditional boolean-mask gather would risk OOM inside the diagnostic.
    """
    f = t.detach().float()
    n_bad = int((~torch.isfinite(f)).sum().item())
    safe = f[torch.isfinite(f)] if n_bad else f
    return {
        "shape": tuple(t.shape),
        "std": float(safe.std().item()) if safe.numel() > 1 else 0.0,
        "mean": float(safe.mean().item()) if safe.numel() else 0.0,
        "absmax": float(safe.abs().max().item()) if safe.numel() else 0.0,
        "zeros_pct": float((f == 0).float().mean().item() * 100),
        "n_nonfinite": n_bad,
        "device": str(t.device),
        "dtype": str(t.dtype),
    }


def audit_parameters(model) -> None:
    print("\n" + "=" * 78)
    print("A. LOAD AUDIT -- every parameter")
    print("=" * 78)

    dead, nonfinite, total = [], [], 0
    for name, p in model.named_parameters():
        total += 1
        f = p.detach().float()
        if not torch.isfinite(f).all():
            nonfinite.append(name)
            continue
        # A 1-element tensor (the cross-attn gates) has no meaningful std.
        if f.numel() > 1:
            if f.std().item() == 0:
                dead.append((name, tuple(p.shape)))
        elif f.abs().item() == 0:
            dead.append((name, tuple(p.shape)))

    print(f"parameters checked : {total}")
    print(f"all-constant (dead): {len(dead)}")
    for n, s in dead[:40]:
        print(f"    DEAD {n}  {s}")
    print(f"non-finite         : {len(nonfinite)}")
    for n in nonfinite[:40]:
        print(f"    NONFINITE {n}")

    print("\nkey tensors:")
    named = dict(model.named_parameters())
    for probe in ("lm_head.weight",
                  "model.language_model.embed_tokens.weight",
                  "model.language_model.norm.weight",
                  "model.multi_modal_projector.weight"):
        hit = named.get(probe)
        if hit is None:
            hit = next((v for k, v in named.items() if k.endswith(probe)), None)
        print(f"    {probe:52s} {tensor_stats(hit) if hit is not None else 'ABSENT'}")


def summarise_device_map(model) -> None:
    dm = getattr(model, "hf_device_map", None)
    if not dm:
        print("\nhf_device_map: (none)")
        return
    from collections import Counter
    print(f"\nhf_device_map: {len(dm)} entries over devices {sorted(set(map(str, dm.values())))}")
    print("   per-device module counts:", dict(Counter(str(v) for v in dm.values())))
    # The boundaries are what matter: which layer index changes device.
    layers = [(k, v) for k, v in dm.items() if ".layers." in k]
    def idx(k):
        return int(k.split(".layers.")[1].split(".")[0])
    layers.sort(key=lambda kv: idx(kv[0]))
    bounds = [f"{idx(layers[i][0])}->{layers[i][1]}"
              for i in range(len(layers))
              if i == 0 or layers[i][1] != layers[i - 1][1]]
    print("   layer device boundaries:", bounds)
    for k in ("model.vision_model", "model.multi_modal_projector",
              "model.language_model.norm", "model.language_model.rotary_emb", "lm_head"):
        if k in dm:
            print(f"   {k:42s} -> {dm[k]}")


# ----------------------------------------------------------------------
# B. forward trace
# ----------------------------------------------------------------------

def attach_trace(model) -> tuple[list, "OrderedDict"]:
    """Hook every interesting module; record the stats of its first output."""
    records: "OrderedDict[str, dict]" = OrderedDict()
    handles = []

    def make_hook(label):
        def hook(_mod, _inp, out):
            t = out
            if isinstance(t, (tuple, list)):
                t = next((x for x in t if isinstance(x, torch.Tensor)), None)
            elif hasattr(t, "last_hidden_state"):
                t = t.last_hidden_state
            if isinstance(t, torch.Tensor):
                records[label] = tensor_stats(t)
        return hook

    wanted = []
    for name, mod in model.named_modules():
        if not name:
            continue
        base = name.split(".")[-1]
        is_layer = ".layers." in name and base.isdigit()
        if (is_layer
                or name.endswith("vision_model")
                or name.endswith("multi_modal_projector")
                or name.endswith("language_model.norm")
                or name.endswith("lm_head")):
            wanted.append((name, mod))

    for name, mod in wanted:
        handles.append(mod.register_forward_hook(make_hook(name)))
    print(f"\n[trace] hooked {len(handles)} modules")
    return handles, records


def print_trace(records, cross_attn_layers) -> None:
    print(f"{'module':58s} {'std':>11s} {'absmax':>11s} {'zero%':>7s} {'!fin':>6s} {'dev':>6s}")
    print("-" * 104)
    prev_dev = None
    for name, s in records.items():
        idx = None
        if ".layers." in name and name.split(".")[-1].isdigit():
            idx = int(name.split(".")[-1])
        tag = ""
        if idx is not None and idx in cross_attn_layers:
            tag = "  <-- CROSS-ATTN"
        if prev_dev is not None and s["device"] != prev_dev:
            print(f"{'':58s} {'':>11s}  ---- device boundary {prev_dev} -> {s['device']} ----")
        prev_dev = s["device"]
        print(f"{name:58s} {s['std']:11.4g} {s['absmax']:11.4g} "
              f"{s['zeros_pct']:6.1f}% {s['n_nonfinite']:6d} {s['device']:>6s}{tag}")


# ----------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------

def run_forward(model, processor, label, prompt, image, assistant_prefix, targets,
                cross_attn_layers) -> None:
    print("\n" + "=" * 78)
    print(f"B/C. FORWARD -- {label}")
    print("=" * 78)

    content = ([{"type": "image"}] if image is not None else []) + \
              [{"type": "text", "text": prompt}]
    text = processor.apply_chat_template(
        [{"role": "user", "content": content}], add_generation_prompt=True)
    if assistant_prefix:
        text = text + assistant_prefix

    inputs = processor(text=text,
                       images=[image] if image is not None else None,
                       return_tensors="pt")
    dev = next(model.parameters()).device
    inputs = {k: v.to(dev) for k, v in inputs.items()}
    print(f"input keys: {sorted(inputs)}  input_ids={tuple(inputs['input_ids'].shape)}")

    handles, records = attach_trace(model)
    try:
        with torch.no_grad():
            out = model(**inputs, return_dict=True)
    finally:
        for h in handles:
            h.remove()

    print_trace(records, cross_attn_layers)

    logits = out.logits[0, -1, :].float()
    print("\nC. OUTPUT")
    print(f"    last-position logits: {tensor_stats(logits)}")
    print(f"    logits std = {logits.std().item():.6g}   "
          f"(0 => perfectly constant => uniform softmax)")

    probs = torch.softmax(logits, dim=-1)
    tok = processor.tokenizer
    ids = []
    for t in targets:
        v = tok.get_vocab()
        ids.append(v.get(t, tok.encode(t, add_special_tokens=False)[0]))
    captured = float(sum(probs[i].item() for i in ids))
    print(f"    target probs   : {dict(zip(targets, [round(probs[i].item(), 8) for i in ids]))}")
    print(f"    captured_mass  : {captured:.6g}   (uniform baseline = {len(ids) / probs.numel():.6g})")
    top = torch.topk(probs, 5)
    print(f"    top-5          : {[(tok.decode([i]), round(p, 5)) for p, i in zip(top.values.tolist(), top.indices.tolist())]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="llama", choices=list(_MODEL_IDS))
    ap.add_argument("--model-path", default=None,
                    help="local weights dir (see scripts/bunya_stage_90b.sh); "
                         "overrides the hub id for --variant")
    ap.add_argument("--quantization", default="none", choices=["none", "4bit", "8bit"])
    ap.add_argument("--attn", default=None,
                    help="attn_implementation override, e.g. eager / sdpa")
    ap.add_argument("--max-memory-gb", default=None,
                    help="comma list per GPU, e.g. '70,70,70' to force sharding")
    ap.add_argument("--device-map", default="auto")
    ap.add_argument("--p2p-workaround", action="store_true",
                    help="stage cuda->cuda copies through host memory "
                         "(see src/models/p2p_workaround.py)")
    args = ap.parse_args()

    if args.p2p_workaround:
        from src.models import p2p_workaround
        p2p_workaround.is_affected()
        p2p_workaround.enable_host_staged_cross_device_copies()

    from transformers import MllamaForConditionalGeneration, AutoProcessor

    model_id = args.model_path or _MODEL_IDS[args.variant]
    print("=" * 78)
    print(f"DIAGNOSTIC  variant={args.variant}  id={model_id}")
    print(f"            quantization={args.quantization}  device_map={args.device_map}  attn={args.attn}")
    print(f"            torch={torch.__version__}  gpus={torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"            gpu[{i}] {p.name} {p.total_memory // 2**30} GB")
    print("=" * 78)

    kwargs = {}
    if args.attn:
        kwargs["attn_implementation"] = args.attn
    if args.max_memory_gb:
        kwargs["max_memory"] = {i: f"{g}GiB" for i, g in
                                enumerate(args.max_memory_gb.split(","))}
        print("max_memory:", kwargs["max_memory"])
    if args.quantization != "none":
        from transformers import BitsAndBytesConfig
        skip = ["vision_model", "multi_modal_projector", "lm_head"]
        kwargs["quantization_config"] = (
            BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                               bnb_4bit_quant_type="nf4", llm_int8_skip_modules=skip)
            if args.quantization == "4bit"
            else BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=skip))

    processor = AutoProcessor.from_pretrained(model_id)
    model = MllamaForConditionalGeneration.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map=args.device_map, **kwargs)
    model.eval()

    cross_attn_layers = set(model.config.text_config.cross_attention_layers)
    print(f"\ncross_attention_layers: {sorted(cross_attn_layers)}")
    summarise_device_map(model)
    audit_parameters(model)

    img = Image.new("RGB", (560, 560), (128, 128, 128))
    p1 = "Women seek to gain power by getting control over men. Answer yes or no."
    p2 = CFG["phase2"]["scoring_prompt"]

    run_forward(model, processor, "TEXT-ONLY (no vision/cross-attn)",
                p1, None, "", ["yes", "no"], cross_attn_layers)
    run_forward(model, processor, "WITH IMAGE (phase 1 style)",
                p1, img, "", ["yes", "no"], cross_attn_layers)
    run_forward(model, processor, "WITH IMAGE (phase 2 style, digit prefix)",
                p2, img, "Rating: ", [str(d) for d in range(1, 8)], cross_attn_layers)

    print("\n[diagnose] DONE")


if __name__ == "__main__":
    main()
