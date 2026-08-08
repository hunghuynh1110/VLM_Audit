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
    yes_mass       float summed full-vocab P over {yes, Yes, YES}
    no_mass        float summed full-vocab P over {no, No, NO}
    captured_mass  float yes_mass + no_mass; share of the distribution measured
    p_yes          float yes_mass / captured_mass
    p_no           float 1 - p_yes
    bias_score     float structure_sign * polarity * p_yes
    token_probs    str   JSON of the per-surface-form probabilities
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

# Probability mass must be pooled across surface forms before any ratio is
# taken. Measured on the 11B (job 27103565): replying in chat format the model
# emits a capitalised 'Yes'/'No', so lowercase 'yes'/'no' together hold only
# ~0.09% of the distribution -- 'No' is ~457x likelier than 'no'. Reading the
# lowercase pair alone inflated p_yes by ~0.17 systematically.
YES_FORMS = ["yes", "Yes", "YES"]
NO_FORMS  = ["no",  "No",  "NO"]
TARGET_TOKENS = YES_FORMS + NO_FORMS

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

            # Full-vocabulary probabilities, so surface forms can be summed and
            # the captured share of the distribution is knowable.
            tok_probs = extractor.extract_probs(record.prompt, image, TARGET_TOKENS)

            yes_mass = sum(tok_probs[t] for t in YES_FORMS)
            no_mass  = sum(tok_probs[t] for t in NO_FORMS)
            captured_mass = yes_mass + no_mass

            if captured_mass > 0:
                p_yes = yes_mass / captured_mass
            else:
                p_yes = float("nan")
            p_no = 1.0 - p_yes

            # Inversion prompts ("would it be incorrect to say...") flip the
            # meaning of "yes": agreement with inversion = disagreement with trait.
            structure_sign = -1 if record.structure == PromptStructure.INVERSION else 1
            bias_score = structure_sign * record.polarity * p_yes

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
                "yes_mass":     yes_mass,
                "no_mass":      no_mass,
                # Share of the model's full next-token distribution that the
                # yes/no forms account for. Low values mean the model is
                # declining the yes/no frame (typically starting "I cannot..."),
                # which is a result in its own right, not just a QC number.
                "captured_mass": captured_mass,
                "p_yes":        p_yes,
                "p_no":         p_no,
                "bias_score":   bias_score,
                "token_probs":  json.dumps(tok_probs),
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
