"""
Phase 1 inference runner.

Iterates over the 330 PromptRecord instances from get_all_prompts(), runs
one forward pass per record, and writes one row to a JSONL checkpoint
immediately after each inference. On restart, already-completed rows are
skipped (matched on model_id + item_id + structure + condition). At the
end the JSONL checkpoint is converted to a parquet file for analysis.

Schema (per row):
    model_id       str   HF repo string (e.g. meta-llama/Llama-3.2-90B-Vision-Instruct)
    quantization   str   "none" | "4bit" | "8bit"
    seed           int   stimulus RNG seed (Gaussian noise)
    item_id        int   1–22, ASI item number
    subscale       str   "HS" | "BS"
    polarity       int   +1 | -1
    structure      str   PromptStructure value
    condition      str   ModalityCondition value
    prompt         str   final prompt text including suffix
    logit_yes      float raw next-token logit for "yes"
    logit_no       float raw next-token logit for "no"
    p_yes          float softmax over (logit_yes, logit_no)
    p_no           float
    bias_score     float polarity * p_yes
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from tqdm import tqdm

from src.config import CFG
from src.data.asi_items import (
    ModalityCondition,
    PromptRecord,
    PromptStructure,
    get_all_prompts,
)
from src.data.image_generators import get_condition_image
from src.models.base_extractor import BaseExtractor

TARGET_TOKENS = ["yes", "no"]
NOISE_SEED = 42  # matches generate_gaussian_noise default


@dataclass
class RunConfig:
    model_name: str           # short key used in output filenames ("llama", "llama_dev", "qwen")
    output_dir: Path          # outputs/phase1/
    limit: Optional[int] = None  # if set, only run the first N prompts (smoke test)


def _row_key(model_id: str, item_id: int, structure: str, condition: str) -> tuple:
    return (model_id, item_id, structure, condition)


def _load_completed_keys(checkpoint_path: Path) -> set[tuple]:
    """Return the set of row keys already recorded in the JSONL checkpoint."""
    if not checkpoint_path.exists():
        return set()
    keys: set[tuple] = set()
    with open(checkpoint_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            keys.add(_row_key(row["model_id"], row["item_id"], row["structure"], row["condition"]))
    return keys


def _records_to_run(records: Iterable[PromptRecord], limit: Optional[int]) -> list[PromptRecord]:
    records = list(records)
    if limit is not None:
        records = records[:limit]
    return records


def run_phase1(extractor: BaseExtractor, cfg: RunConfig) -> Path:
    """
    Run Phase 1 inference loop. Returns the final parquet path.
    """
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = cfg.output_dir / f"{cfg.model_name}_checkpoint.jsonl"
    parquet_path    = cfg.output_dir / f"{cfg.model_name}.parquet"

    model_id = getattr(extractor, "model_id", cfg.model_name)
    quantization = getattr(extractor, "quantization", "none")

    all_records = _records_to_run(get_all_prompts(), cfg.limit)
    completed = _load_completed_keys(checkpoint_path)
    pending = [
        r for r in all_records
        if _row_key(model_id, r.item_id, r.structure.value, r.condition.value) not in completed
    ]

    print(f"[runner] model_id     = {model_id}")
    print(f"[runner] quantization = {quantization}")
    print(f"[runner] checkpoint   = {checkpoint_path}")
    print(f"[runner] total        = {len(all_records)}")
    print(f"[runner] completed    = {len(completed)}")
    print(f"[runner] pending      = {len(pending)}")

    image_cache: dict[ModalityCondition, object] = {}

    with open(checkpoint_path, "a") as fout:
        for record in tqdm(pending, desc=f"phase1[{cfg.model_name}]", unit="prompt",
                           file=sys.stdout, disable=False, mininterval=60, miniters=1):
            if record.condition not in image_cache:
                image_cache[record.condition] = get_condition_image(record.condition)
            image = image_cache[record.condition]

            logits = extractor.extract_logits(record.prompt, image, TARGET_TOKENS)
            probs  = extractor.softmax_probs(logits)

            p_yes = probs["yes"]
            p_no  = probs["no"]
            bias_score = record.polarity * p_yes

            row = {
                "model_id":     model_id,
                "quantization": quantization,
                "seed":         NOISE_SEED,
                "item_id":      record.item_id,
                "subscale":     record.subscale,
                "polarity":     record.polarity,
                "structure":    record.structure.value,
                "condition":    record.condition.value,
                "prompt":       record.prompt,
                "logit_yes":    logits["yes"],
                "logit_no":     logits["no"],
                "p_yes":        p_yes,
                "p_no":         p_no,
                "bias_score":   bias_score,
            }
            fout.write(json.dumps(row) + "\n")
            fout.flush()

    _convert_checkpoint_to_parquet(checkpoint_path, parquet_path)
    print(f"[runner] wrote parquet → {parquet_path}")
    return parquet_path


def _convert_checkpoint_to_parquet(checkpoint_path: Path, parquet_path: Path) -> None:
    import pandas as pd
    rows = []
    with open(checkpoint_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    df.to_parquet(parquet_path, index=False)


def default_output_dir() -> Path:
    return CFG["paths"]["outputs"] / "phase1"
