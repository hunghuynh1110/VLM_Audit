"""
Gate 2 — 90B memory + latency preflight.

Loads Llama-3.2-90B-Vision-Instruct in 8-bit, runs one forward pass with
the gray-patch stimulus + a known ASI prompt, and prints peak VRAM and
elapsed wall time. Used to size the Phase 1 SLURM --time budget:
    --time = ceil(330 * elapsed_seconds * 1.4 / 3600) hours

Usage (on Bunya, inside the SLURM job):
    .venv/bin/python scripts/preflight_90b.py --quantization 8bit
    .venv/bin/python scripts/preflight_90b.py --quantization 4bit  # fallback
"""

import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

import torch

from src.data.image_generators import generate_gray_patch
from src.models.llama_extractor import LlamaExtractor


PROBE_PROMPT = "Is this person too easily offended? Answer yes or no."
TARGET_TOKENS = ["yes", "no"]


def fmt_gb(bytes_: int) -> float:
    return bytes_ / (1024 ** 3)


def main() -> None:
    parser = argparse.ArgumentParser(description="Llama-3.2-90B preflight (memory + latency)")
    parser.add_argument("--quantization", choices=["none", "4bit", "8bit"], default="8bit",
                        help="bitsandbytes quantisation mode (default: 8bit)")
    parser.add_argument("--variant", default="llama", choices=["llama", "llama_dev"],
                        help="HF model variant key from config.yaml")
    parser.add_argument("--warmup", type=int, default=1,
                        help="Number of warmup forward passes before timing")
    parser.add_argument("--measure", type=int, default=3,
                        help="Number of timed forward passes; latency reported as mean")
    args = parser.parse_args()

    print(f"[preflight] CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[preflight] GPU count:      {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"[preflight]   GPU {i}: {props.name} ({fmt_gb(props.total_memory):.1f} GB)")
        torch.cuda.reset_peak_memory_stats()

    print(f"[preflight] Loading variant='{args.variant}' quantization='{args.quantization}' …")
    t0 = time.time()
    extractor = LlamaExtractor(
        variant=args.variant,
        device="auto",
        quantization=args.quantization,
    )
    load_seconds = time.time() - t0
    print(f"[preflight] Model loaded in {load_seconds:.1f} s")

    if torch.cuda.is_available():
        peak_after_load = sum(
            torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())
        )
        print(f"[preflight] Peak VRAM after load: {fmt_gb(peak_after_load):.2f} GB")

    image = generate_gray_patch()

    print(f"[preflight] Warmup ({args.warmup} pass) …")
    for _ in range(args.warmup):
        extractor.extract_logits(PROBE_PROMPT, image, TARGET_TOKENS)

    print(f"[preflight] Timing {args.measure} forward passes …")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = []
    for _ in range(args.measure):
        t0 = time.time()
        logits = extractor.extract_logits(PROBE_PROMPT, image, TARGET_TOKENS)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed.append(time.time() - t0)

    mean_latency = sum(elapsed) / len(elapsed)
    probs = extractor.softmax_probs(logits)

    if torch.cuda.is_available():
        peak_total = sum(
            torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())
        )
    else:
        peak_total = 0

    print()
    print("=" * 60)
    print("PREFLIGHT RESULTS")
    print("=" * 60)
    print(f"Model:              {extractor.model_id}")
    print(f"Quantization:       {extractor.quantization}")
    print(f"Peak VRAM (total):  {fmt_gb(peak_total):.2f} GB")
    print(f"Latency per pass:   {mean_latency:.2f} s (mean of {args.measure})")
    print(f"All latencies:      {[f'{e:.2f}' for e in elapsed]}")
    print(f"Probe logits:       {logits}")
    print(f"Probe probs:        {probs}")
    print()
    suggested_seconds = math.ceil(330 * mean_latency * 1.4)
    suggested_hours = suggested_seconds / 3600
    print(f"Phase 1 SLURM --time suggestion (330 prompts × t × 1.4):")
    print(f"  {suggested_seconds} s = {suggested_hours:.2f} h")
    print(f"  Round up to next half hour for #SBATCH --time")
    print("=" * 60)


if __name__ == "__main__":
    main()
