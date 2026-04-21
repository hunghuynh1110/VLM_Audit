"""
Loads SIGIR 2018 dataset and computes human ASI-group baselines.

Dataset notes:
  - ASI items (asi1..asi22) are on a 0–5 scale (not the standard 1–7).
    Reverse-coding is already applied in the CSV for con-trait items
    (items 2, 3, 7 and HSS items 3, 6, 9). Do NOT re-reverse.
  - `specific_bias_0..9` — per-query objectivity rating [1–7].
    This is the human equivalent of our Phase 2 scoring prompt and is
    the correct column for computing r̄_lowASI,q.
  - `objective` — participant's global view of search engines in general.
    Do NOT use this for per-query baselines.
  - Low/high-ASI split uses median of asi_score (≈ 2.82 on 0–5 scale).
"""

import ast
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import CFG

__all__ = [
    "QUERIES",
    "load_raw",
    "get_low_asi_group",
    "compute_low_asi_baselines",
    "get_image_paths",
    "get_asi_items_from_data",
]

QUERIES: list[str] = [
    "aggressive person",
    "anxious person",
    "bossy person",
    "calm person",
    "casual person",
    "hot air baloon",
    "interested person",
    "smart person",
    "warm person",
    "working person",
]

_DATA_DIR: Path = CFG["paths"]["sigir_data"]


def load_raw(data_dir: Path = _DATA_DIR) -> pd.DataFrame:
    df = pd.read_csv(data_dir / "final_anonymised.csv")
    _validate_asi_range(df)
    return df


def _validate_asi_range(df: pd.DataFrame) -> None:
    asi_cols = [f"asi{i}" for i in range(1, 23)]
    for col in asi_cols:
        bad = df[col].dropna()
        if not bad.between(0, 5).all():
            raise ValueError(f"{col} contains values outside [0, 5]: {bad[~bad.between(0,5)].unique()}")


def get_low_asi_group(df: pd.DataFrame, threshold: float | None = None) -> pd.DataFrame:
    """Return rows where asi_score ≤ threshold (default: median)."""
    cut = threshold if threshold is not None else df["asi_score"].median()
    return df[df["asi_score"] <= cut].copy()


def compute_low_asi_baselines(
    data_dir: Path = _DATA_DIR,
    asi_threshold: float | None = None,
    queries: list[str] = QUERIES,
) -> dict[str, float]:
    """
    Compute r̄_lowASI,q — mean human low-ASI objectivity rating per query.

    Uses `specific_bias_i` at the position each query appeared for each participant.
    Returns {query: mean_rating} on the [1–7] objectivity scale.
    """
    df = load_raw(data_dir)
    low = get_low_asi_group(df, asi_threshold)

    query_set = set(queries)
    ratings: dict[str, list[float]] = {q: [] for q in queries}

    for i in range(10):
        q_col = f"{i}_query"
        b_col = f"specific_bias_{i}"
        chunk = low[[q_col, b_col]].dropna()
        for _, row in chunk.iterrows():
            q = row[q_col]
            if q in query_set:
                ratings[q].append(float(row[b_col]))

    return {q: float(np.mean(v)) if v else float("nan") for q, v in ratings.items()}


def get_image_paths(
    data_dir: Path = _DATA_DIR,
    images_subdir: str = "images",
    queries: list[str] = QUERIES,
) -> dict[str, list[Path]]:
    """
    Return per-query image file paths from the extracted images.zip archive.

    Prerequisite: images.zip must be extracted at data_dir/images/ before calling.
    Paths are returned regardless of whether files exist — callers should verify.

    Returns {query: [Path, ...]} with up to k=9 unique images per query.
    """
    df = load_raw(data_dir)
    images_dir = data_dir / images_subdir
    query_set = set(queries)

    query_images: dict[str, list[Path]] = {q: [] for q in queries}
    seen: dict[str, set[str]] = {q: set() for q in queries}

    for _, row in df.iterrows():
        for i in range(10):
            query = row.get(f"{i}_query")
            if not isinstance(query, str) or query not in query_set:
                continue
            raw = row.get(f"{i}_imageurl")
            if pd.isna(raw):
                continue
            try:
                urls = ast.literal_eval(raw) if isinstance(raw, str) else raw
            except (ValueError, SyntaxError):
                urls = [raw]
            for url in urls:
                fname = Path(str(url)).name
                if fname and fname not in seen[query]:
                    seen[query].add(fname)
                    query_images[query].append(images_dir / fname)

    return query_images


def get_asi_items_from_data(data_dir: Path = _DATA_DIR) -> pd.DataFrame:
    """
    Return raw ASI item responses for all participants.

    Scale: 0–5 (already reverse-coded for con-trait items).
    Columns: _worker_id, asi_score, asi1..asi22
    """
    df = load_raw(data_dir)
    asi_cols = [f"asi{i}" for i in range(1, 23)]
    return df[["_worker_id", "asi_score"] + asi_cols].copy()
