"""
Phase 1 CLI entry point.

Usage:
    # Local Mac smoke test (Gate 2.5):
    .venv/bin/python scripts/run_phase1.py --model llama_dev --quantization 4bit --limit 5

    # Bunya production run on the 90B:
    .venv/bin/python scripts/run_phase1.py --model llama --quantization 8bit
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.phase1 import metrics
from src.phase1.runner import RunConfig, default_output_dir, run_phase1


def _build_extractor(model: str, device: str, quantization: str):
    if model in ("llama", "llama_dev"):
        from src.models.llama_extractor import LlamaExtractor
        return LlamaExtractor(variant=model, device=device, quantization=quantization)
    if model == "qwen":
        # Implemented later (Step 6 in TODO.md). Fail fast for now.
        raise NotImplementedError("Qwen extractor not implemented yet (TODO.md Step 6)")
    raise ValueError(f"Unknown --model '{model}'")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 inference runner")
    parser.add_argument("--model", required=True, choices=["llama", "llama_dev", "qwen"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--quantization", choices=["none", "4bit", "8bit"], default="none")
    parser.add_argument("--limit", type=int, default=None,
                        help="Run only the first N prompts (smoke test)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Override default output dir (outputs/phase1/)")
    args = parser.parse_args()

    output_dir = args.output_dir or default_output_dir()

    print(f"[run_phase1] model={args.model} device={args.device} "
          f"quantization={args.quantization} limit={args.limit}")
    print(f"[run_phase1] output_dir={output_dir}")
    extractor = _build_extractor(args.model, args.device, args.quantization)

    cfg = RunConfig(model_name=args.model, output_dir=output_dir, limit=args.limit)
    parquet_path = run_phase1(extractor, cfg)

    import pandas as pd
    df = pd.read_parquet(parquet_path)
    summary = metrics.summary(df)

    summary_path = output_dir / f"{args.model}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print("=" * 60)
    print("PHASE 1 SUMMARY")
    print("=" * 60)
    print(f"Rows:           {summary['n_rows']}")
    print(f"ASI_intrinsic:  {summary['asi_intrinsic']:+.4f}")
    print(f"By structure:   {summary['by_structure']}")
    print(f"By condition:   {summary['by_condition']}")
    print(f"By subscale:    {summary['by_subscale']}")
    print(f"\nSaved → {parquet_path}")
    print(f"Saved → {summary_path}")


if __name__ == "__main__":
    main()
