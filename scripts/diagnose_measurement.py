"""
Measurement-validity diagnostic for the Phase 1 logit probe.

This does NOT test "does the code run" -- the smoke tests cover that. It tests
whether the number we extract is the number we think we are extracting.

Four questions:

  Q1. Where does the model's next-token probability mass actually sit?
      We read logits for the tokens "yes"/"no". If the model overwhelmingly
      prefers "Yes"/"YES"/" yes", then p_yes is computed over two low-mass
      tokens and may be dominated by tail noise.

  Q2. How much of the full distribution do our two target tokens capture?
      Reported as captured_mass. If this is tiny, the softmax-over-two-logits
      in Eq. 3.1 is measuring a sliver of the model's actual behaviour.

  Q3. Which modules actually get quantised, and is lm_head among them?
      lm_head produces the dependent variable; quantising it injects noise
      straight into the measurement.

  Q4. How far does each quantisation setting move p_yes from the bf16
      reference on identical prompts?

Usage:
    python scripts/diagnose_measurement.py --quantization 4bit --limit 8
    python scripts/diagnose_measurement.py --quantization 4bit --no-skip --limit 8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

import torch

from src.data.asi_items import get_all_prompts
from src.data.image_generators import get_condition_image

CASE_VARIANTS = ["yes", "Yes", "YES", " yes", " Yes", "no", "No", "NO", " no", " No"]


def module_quantisation_report(model) -> dict:
    """Count Linear vs quantised-Linear per top-level submodule."""
    from collections import defaultdict
    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"quantised": 0, "full": 0})
    lm_head_type = None
    for name, mod in model.named_modules():
        cls = type(mod).__name__
        if cls in ("Linear", "Linear4bit", "Linear8bitLt", "Params4bit"):
            top = name.split(".")[0:3]
            key = ".".join(top)
            if cls in ("Linear4bit", "Linear8bitLt"):
                counts[key]["quantised"] += 1
            elif cls == "Linear":
                counts[key]["full"] += 1
        if name.endswith("lm_head"):
            lm_head_type = cls
    return {"lm_head_class": lm_head_type, "by_module": dict(counts)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama_dev")
    ap.add_argument("--quantization", default="4bit", choices=["none", "4bit", "8bit"])
    ap.add_argument("--no-skip", action="store_true",
                    help="Quantise everything, including vision tower and lm_head")
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--out", type=Path, default=Path("outputs/diagnostics"))
    args = ap.parse_args()

    from src.models.llama_extractor import LlamaExtractor

    skip = [] if args.no_skip else None
    tag = f"{args.quantization}{'_noskip' if args.no_skip else ''}"
    print(f"[diag] === {args.model} / {tag} ===", flush=True)

    ex = LlamaExtractor(variant=args.model, device="auto",
                        quantization=args.quantization, skip_modules=skip)
    print(f"[diag] skip_modules = {ex.skip_modules}", flush=True)

    # ---- Q3: what actually got quantised -----------------------------------
    qrep = module_quantisation_report(ex.model)
    print(f"[diag] lm_head class = {qrep['lm_head_class']}"
          f"   <-- must NOT be Linear4bit/Linear8bitLt", flush=True)
    for k, v in sorted(qrep["by_module"].items()):
        if v["quantised"] or v["full"]:
            print(f"[diag]   {k:52s} quantised={v['quantised']:<4d} full={v['full']}")

    records = get_all_prompts()[: args.limit]
    rows = []

    for rec in records:
        image = get_condition_image(rec.condition)

        # ---- Q1/Q2: full next-token distribution --------------------------
        inputs = ex._build_inputs(rec.prompt, image)
        with torch.no_grad():
            out = ex.model(**inputs, return_dict=True)
        logits = out.logits[0, -1, :].float()
        probs = torch.softmax(logits, dim=-1)

        topk = torch.topk(probs, 10)
        top_tokens = [
            (ex.processor.tokenizer.decode([i]), round(p.item(), 4))
            for i, p in zip(topk.indices.tolist(), topk.values)
        ]

        variant_mass = {}
        for v in CASE_VARIANTS:
            ids = ex.processor.tokenizer.encode(v, add_special_tokens=False)
            if len(ids) == 1:
                variant_mass[v] = round(probs[ids[0]].item(), 6)

        yes_id = ex._resolve_token_id("yes")
        no_id = ex._resolve_token_id("no")
        captured = probs[yes_id].item() + probs[no_id].item()

        p_yes_pair = probs[yes_id].item() / captured if captured > 0 else float("nan")

        # p_yes if we instead pooled every case variant
        yes_pool = sum(m for v, m in variant_mass.items() if v.strip().lower() == "yes")
        no_pool = sum(m for v, m in variant_mass.items() if v.strip().lower() == "no")
        p_yes_pooled = yes_pool / (yes_pool + no_pool) if (yes_pool + no_pool) > 0 else float("nan")

        rows.append({
            "item_id": rec.item_id,
            "structure": rec.structure.value,
            "condition": rec.condition.value,
            "captured_mass": round(captured, 6),
            "p_yes_pair": round(p_yes_pair, 6),
            "p_yes_pooled": round(p_yes_pooled, 6),
            "top10": top_tokens,
            "variant_mass": variant_mass,
        })

        print(f"\n[diag] item={rec.item_id} {rec.structure.value}/{rec.condition.value}")
        print(f"[diag]   top10        : {top_tokens}")
        print(f"[diag]   variant mass : {variant_mass}")
        print(f"[diag]   captured by ('yes','no') = {captured:.4%}")
        print(f"[diag]   p_yes pair={p_yes_pair:.4f}  pooled-case={p_yes_pooled:.4f}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"diag_{args.model}_{tag}.json"
    with open(path, "w") as f:
        json.dump({"skip_modules": ex.skip_modules,
                   "quantisation_report": qrep,
                   "rows": rows}, f, indent=2)

    mean_captured = sum(r["captured_mass"] for r in rows) / len(rows)
    print(f"\n[diag] MEAN captured_mass over {len(rows)} prompts = {mean_captured:.4%}")
    print(f"[diag] saved -> {path}")


if __name__ == "__main__":
    main()
