"""
Phase 1 aggregation metrics.

ASI_intrinsic = mean of bias_score across all rows.
Breakdowns let you inspect whether the bias score is stable across prompt
structures, modality conditions, or subscales (HS vs BS).
"""

from __future__ import annotations

import pandas as pd


def compute_asi_intrinsic(df: pd.DataFrame) -> float:
    """Mean bias_score over all rows."""
    return float(df["bias_score"].mean())


def breakdown_by_structure(df: pd.DataFrame) -> dict[str, float]:
    """Mean bias_score grouped by prompt structure (5 keys)."""
    return df.groupby("structure")["bias_score"].mean().to_dict()


def breakdown_by_condition(df: pd.DataFrame) -> dict[str, float]:
    """Mean bias_score grouped by modality condition (3 keys)."""
    return df.groupby("condition")["bias_score"].mean().to_dict()


def breakdown_by_subscale(df: pd.DataFrame) -> dict[str, float]:
    """Mean bias_score grouped by ASI subscale (HS, BS)."""
    return df.groupby("subscale")["bias_score"].mean().to_dict()


def summary(df: pd.DataFrame) -> dict:
    """Single-call dump of every aggregation a caller is likely to want."""
    return {
        "n_rows": int(len(df)),
        "asi_intrinsic": compute_asi_intrinsic(df),
        "by_structure": breakdown_by_structure(df),
        "by_condition": breakdown_by_condition(df),
        "by_subscale":  breakdown_by_subscale(df),
    }
