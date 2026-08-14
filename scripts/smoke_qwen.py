"""
Qwen2-VL smoke test: does the extractor measure what we think it measures?

Deliberately not a "did it load" check -- that is exactly the standard that let
the 90B ship two runs of uniform noise. This exercises the real prompts, the
real stimulus images and the real target tokens, and reports captured_mass,
which is how every measurement bug in this project has been caught so far.

Checks, in order:
  1. the model shards across GPUs and the self-test passes (that is inside
     QwenExtractor.__init__, so an unhealthy load raises before we get here)
  2. the pinned visual token budget actually holds across image conditions --
     Qwen2-VL uses dynamic resolution, and a varying token count across
     conditions would be a confound, not a detail
  3. Phase 1: yes/no probability mass per modality condition
  4. Phase 2: the rating digits, which depend on the trailing-space
     tokenisation assumption inherited from the Llama prompt

Usage:
    python scripts/smoke_qwen.py --weights-path <dir> --limit 6
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

import torch

from src.config import CFG
from src.data.asi_items import ModalityCondition, get_all_prompts
from src.data.image_generators import get_condition_image

YES_FORMS = ["yes", "Yes", "YES"]
NO_FORMS = ["no", "No", "NO"]
DIGITS = [str(d) for d in range(1, 8)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights-path", default=None)
    ap.add_argument("--quantization", default="none", choices=["none", "4bit", "8bit"])
    ap.add_argument("--limit", type=int, default=6)
    args = ap.parse_args()

    from src.models.qwen_extractor import QwenExtractor

    print("=" * 78)
    print(f"QWEN SMOKE TEST  torch={torch.__version__}  gpus={torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  gpu[{i}] {p.name} {p.total_memory // 2**30} GB")
    print("=" * 78)

    ex = QwenExtractor(variant="qwen", device="auto",
                       quantization=args.quantization,
                       weights_path=args.weights_path)

    dm = getattr(ex.model, "hf_device_map", None)
    if dm:
        from collections import Counter
        print(f"\n[smoke] device map spans {sorted(set(map(str, dm.values())))} "
              f"({dict(Counter(str(v) for v in dm.values()))})")

    # --- 2. visual token budget ------------------------------------------
    print("\n" + "=" * 78)
    print("VISUAL TOKEN BUDGET (must be identical across image conditions)")
    print("=" * 78)
    for cond in ModalityCondition:
        img = get_condition_image(cond)
        inputs = ex._build_inputs("describe this", img)
        n_tokens = int(inputs["input_ids"].shape[-1])
        grid = inputs.get("image_grid_thw")
        vis = int(grid.prod(dim=-1).sum().item() // 4) if grid is not None else 0
        print(f"  {cond.value:12s} total_input_tokens={n_tokens:6d}  visual_tokens={vis:6d}")

    # --- 3. Phase 1 ------------------------------------------------------
    print("\n" + "=" * 78)
    print("PHASE 1  (captured_mass = share of distribution on yes/no)")
    print("=" * 78)
    records = get_all_prompts()[: args.limit]
    for rec in records:
        img = get_condition_image(rec.condition)
        probs = ex.extract_probs(rec.prompt, img, YES_FORMS + NO_FORMS)
        yes = sum(probs[t] for t in YES_FORMS)
        no = sum(probs[t] for t in NO_FORMS)
        cap = yes + no
        p_yes = yes / cap if cap > 0 else float("nan")
        print(f"  item{rec.item_id:>3} {rec.structure.value:<12} {rec.condition.value:<11} "
              f"captured_mass={cap:.4f}  p_yes={p_yes:.4f}")

    # --- 4. Phase 2 ------------------------------------------------------
    print("\n" + "=" * 78)
    print("PHASE 2  (digits; validates the trailing-space tokenisation)")
    print("=" * 78)
    scoring = CFG["phase2"]["scoring_prompt"]
    for cond in ModalityCondition:
        img = get_condition_image(cond)
        probs = ex.extract_probs(scoring, img, DIGITS, assistant_prefix="Rating: ")
        cap = sum(probs.values())
        exp = (sum(int(d) * p for d, p in probs.items()) / cap) if cap > 0 else float("nan")
        print(f"  {cond.value:12s} captured_mass={cap:.4f}  expected_rating={exp:.3f}")
        print(f"               {({d: round(p, 4) for d, p in probs.items()})}")

    print("\n[smoke] DONE")


if __name__ == "__main__":
    main()
