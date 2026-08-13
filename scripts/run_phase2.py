"""
Phase 2 CLI entry point.

Usage:
    python scripts/run_phase2.py --model llama_dev --quantization none --limit 8
    python scripts/run_phase2.py --model llama     --quantization none
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.phase2 import metrics
from src.phase2.runner import (
    STREAM_HISTORICAL,
    Phase2Config,
    default_output_dir,
    run_phase2,
)


def _build_extractor(model: str, device: str, quantization: str,
                     weights_path: str | None = None):
    if model in ("llama", "llama_dev"):
        from src.models.llama_extractor import LlamaExtractor
        return LlamaExtractor(variant=model, device=device, quantization=quantization,
                              weights_path=weights_path)
    if model == "qwen":
        from src.models.qwen_extractor import QwenExtractor
        return QwenExtractor(variant=model, device=device, quantization=quantization,
                             weights_path=weights_path)
    raise ValueError(f"Unknown --model {model!r}")


def main() -> None:
    p = argparse.ArgumentParser(description="Phase 2 extrinsic runner")
    p.add_argument("--model", required=True, choices=["llama", "llama_dev", "qwen"])
    p.add_argument("--device", default="auto")
    p.add_argument("--quantization", choices=["none", "4bit", "8bit"], default="none")
    p.add_argument("--weights-path", default=None,
                   help="Load weights from a local dir (see "
                        "scripts/bunya_stage_90b.sh) instead of the HF cache. "
                        "model_id in the output is unaffected.")
    p.add_argument("--stream", default=STREAM_HISTORICAL)
    p.add_argument("--k", type=int, default=9, help="images per query (SIGIR grid = 9)")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    args = p.parse_args()

    output_dir = args.output_dir or default_output_dir()

    import torch
    print(f"[run_phase2] model={args.model} quantization={args.quantization} "
          f"stream={args.stream} k={args.k} limit={args.limit}")
    print(f"[run_phase2] torch={torch.__version__} cuda={torch.cuda.is_available()} "
          f"gpus={torch.cuda.device_count()}")
    print("[run_phase2] loading model ...")
    extractor = _build_extractor(args.model, args.device, args.quantization,
                                 args.weights_path)
    print("[run_phase2] model loaded")

    cfg = Phase2Config(model_name=args.model, output_dir=output_dir,
                       stream=args.stream, k=args.k, limit=args.limit)
    parquet_path = run_phase2(extractor, cfg)

    import pandas as pd
    df = pd.read_parquet(parquet_path)
    summary = metrics.summary(df)

    summary_path = output_dir / f"{args.model}_{args.stream}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print("=" * 60)
    print("PHASE 2 SUMMARY")
    print("=" * 60)
    print(f"Rows:                {summary['n_rows']}")
    print(f"mean captured_mass:  {summary['mean_captured_mass']:.4f}")
    print(f"delta_m:             {summary['delta_m']:+.4f}   "
          "(negative = model rates less objective than low-ASI humans)")
    print(f"RR_vision:           {summary['rr_vision']:.4f}")
    print(f"RR_text_only:        {summary['rr_text_only']:.4f}")
    print(f"delta_RR:            {summary['delta_rr']:+.4f}   "
          "(negative = vision adds instability)")
    print(f"\nSaved → {parquet_path}")
    print(f"Saved → {summary_path}")


if __name__ == "__main__":
    main()
