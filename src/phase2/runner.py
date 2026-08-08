"""
Phase 2 inference runner — extrinsic objectivity ratings on SIGIR 2018 images.

For each (query, image) pair the model is scored four times, which is the CALM
structure from PIPELINE_SPEC:

    run | image supplied | scale order
    ----|----------------|------------
     1  | yes            | original
     2  | yes            | reversed
     3  | no             | original
     4  | no             | reversed

Runs 1-2 give RR_vision, runs 3-4 give RR_text-only, and their difference is
ΔRR — the modality-isolation statistic. Text-only runs are constant per query
(no image involved), but are still recorded per pair so every pair has its own
matched control and the RR arithmetic stays symmetric.

Rating extraction mirrors Phase 1's validity discipline: full-vocabulary
probabilities over the digit tokens "1".."7", so `captured_mass` is knowable.
Phase 1 showed that ignoring it hid a 500x measurement error, and the same trap
exists here — a model that answers "I cannot rate this" puts almost no mass on
any digit, which must be visible rather than silently renormalised away.

Schema (one row per inference):
    model_id, quantization, stream, query, image_id, image_path,
    condition, scale_order, prompt,
    token_probs (JSON), captured_mass,
    rating_expected, rating_argmax, rating_normalised
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PIL import Image
from tqdm import tqdm

from src.config import CFG
from src.data.sigir_loader import QUERIES, get_image_paths
from src.models.base_extractor import BaseExtractor
from src.phase2.prompts import (
    ASSISTANT_PREFIX,
    RATING_TOKENS,
    Phase2Condition,
    ScaleOrder,
    build_scoring_prompt,
    normalise_rating,
)

__all__ = ["Phase2Config", "run_phase2", "default_output_dir"]

STREAM_HISTORICAL = "historical_2018"
STREAM_CONTEMPORARY = "contemporary_2026"


@dataclass
class Phase2Config:
    model_name: str
    output_dir: Path
    stream: str = STREAM_HISTORICAL
    k: int = 9                      # images per query; matches the SIGIR 3x3 grid
    limit: Optional[int] = None     # cap total inferences (smoke test)


def _row_key(model_id: str, stream: str, query: str, image_id: str,
             condition: str, scale_order: str) -> tuple:
    return (model_id, stream, query, image_id, condition, scale_order)


def _load_completed_keys(checkpoint_path: Path) -> set[tuple]:
    if not checkpoint_path.exists():
        return set()
    keys: set[tuple] = set()
    with open(checkpoint_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            keys.add(_row_key(r["model_id"], r["stream"], r["query"],
                              r["image_id"], r["condition"], r["scale_order"]))
    return keys


def _build_work_items(cfg: Phase2Config) -> list[tuple]:
    """(query, image_id, image_path, condition, scale_order) for every inference."""
    if cfg.stream != STREAM_HISTORICAL:
        raise NotImplementedError(
            f"stream {cfg.stream!r} has no image source yet. The contemporary "
            "stream needs the Google/Bing re-crawl (action_plan.md P3)."
        )

    per_query = get_image_paths(k=cfg.k, verify=True)
    items: list[tuple] = []
    for query in QUERIES:
        for path in per_query[query]:
            image_id = f"{path.parent.name}/{path.name}"
            for condition in Phase2Condition:
                for order in ScaleOrder:
                    items.append((query, image_id, path, condition, order))
    return items


def run_phase2(extractor: BaseExtractor, cfg: Phase2Config) -> Path:
    """Run the Phase 2 / CALM inference grid. Returns the parquet path."""
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = cfg.output_dir / f"{cfg.model_name}_{cfg.stream}_checkpoint.jsonl"
    parquet_path    = cfg.output_dir / f"{cfg.model_name}_{cfg.stream}.parquet"

    model_id = getattr(extractor, "model_id", cfg.model_name)
    quantization = getattr(extractor, "quantization", "none")

    items = _build_work_items(cfg)
    if cfg.limit is not None:
        items = items[: cfg.limit]

    completed = _load_completed_keys(checkpoint_path)
    pending = [
        it for it in items
        if _row_key(model_id, cfg.stream, it[0], it[1], it[3].value, it[4].value)
        not in completed
    ]

    print(f"[phase2] model_id   = {model_id}")
    print(f"[phase2] stream     = {cfg.stream}")
    print(f"[phase2] checkpoint = {checkpoint_path}")
    print(f"[phase2] total      = {len(items)}")
    print(f"[phase2] completed  = {len(completed)}")
    print(f"[phase2] pending    = {len(pending)}")

    image_cache: dict[Path, Image.Image] = {}

    with open(checkpoint_path, "a") as fout:
        for query, image_id, path, condition, order in tqdm(
            pending, desc=f"phase2[{cfg.model_name}]", unit="inf",
            file=sys.stdout, mininterval=30, miniters=1,
        ):
            if condition is Phase2Condition.VISION:
                if path not in image_cache:
                    image_cache[path] = Image.open(path).convert("RGB")
                image = image_cache[path]
            else:
                image = None

            prompt = build_scoring_prompt(query, order, condition)
            probs = extractor.extract_probs(
                prompt, image, RATING_TOKENS, assistant_prefix=ASSISTANT_PREFIX
            )

            captured_mass = sum(probs.values())
            if captured_mass > 0:
                # Expected rating under the distribution restricted to the digits.
                rating_expected = sum(
                    int(tok) * p for tok, p in probs.items()
                ) / captured_mass
            else:
                rating_expected = float("nan")
            rating_argmax = int(max(probs, key=probs.get)) if probs else -1

            row = {
                "model_id":      model_id,
                "quantization":  quantization,
                "stream":        cfg.stream,
                "query":         query,
                "image_id":      image_id,
                "image_path":    str(path),
                "condition":     condition.value,
                "scale_order":   order.value,
                "prompt":        prompt,
                "token_probs":   json.dumps(probs),
                # Share of the full distribution sitting on the digits 1-7. Low
                # values mean the model is declining to give a number at all.
                "captured_mass": captured_mass,
                "rating_expected": rating_expected,
                "rating_argmax":   rating_argmax,
                # Both orders mapped onto "higher = more objective" so they are
                # directly comparable and match the SIGIR human scale.
                "rating_normalised": normalise_rating(rating_expected, order),
            }
            fout.write(json.dumps(row) + "\n")
            fout.flush()

    _convert_checkpoint_to_parquet(checkpoint_path, parquet_path)
    print(f"[phase2] wrote parquet → {parquet_path}")
    return parquet_path


def _convert_checkpoint_to_parquet(checkpoint_path: Path, parquet_path: Path) -> None:
    import pandas as pd
    rows = []
    with open(checkpoint_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    pd.DataFrame(rows).to_parquet(parquet_path, index=False)


def default_output_dir() -> Path:
    return CFG["paths"]["outputs"] / "phase2"
