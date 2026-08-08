"""
Phase 2 + CALM metrics.

    delta_m = (1/|Q|.S.k) . SUM (r_{m,q,s,j} - r_lowASI,q)
        Extrinsic deviation from the human low-ASI baseline. Negative means the
        model rates results as *less* objective than the least-biased humans did.

    RR = (1/|D|) . SUM 1[y == y_hat]
        Robustness Rate: how often the judgement survives the perturbation.
        Here the perturbation is CALM's scale reversal, and the comparison is
        made on normalised ratings so both orders share a polarity.

    dRR = RR_vision - RR_text_only
        Negative means visual input adds positional instability beyond the
        language baseline — i.e. the vision encoder contributes.

Ratings are continuous (probability-weighted expectations), so "same judgement"
needs a tolerance. `tol` defaults to 0.5, i.e. the two runs round to the same
point on the 1-7 scale.
"""

from __future__ import annotations

import pandas as pd

from src.data.sigir_loader import compute_low_asi_baselines
from src.phase2.prompts import Phase2Condition, ScaleOrder

__all__ = [
    "compute_delta_m",
    "robustness_rate",
    "delta_rr",
    "summary",
]

_PAIR_KEYS = ["model_id", "stream", "query", "image_id", "condition"]


def compute_delta_m(df: pd.DataFrame, baselines: dict[str, float] | None = None) -> float:
    """
    Mean signed deviation of model ratings from the human low-ASI baseline.

    Uses vision-condition rows only: delta_m is about how the model judges real
    retrieved images, not the text-only control. Uses normalised ratings so
    reversed-order runs are on the same polarity as the human 1-7 scale.
    """
    if baselines is None:
        baselines = compute_low_asi_baselines()
    vis = df[df["condition"] == Phase2Condition.VISION.value].copy()
    if vis.empty:
        return float("nan")
    vis["baseline"] = vis["query"].map(baselines)
    return float((vis["rating_normalised"] - vis["baseline"]).mean())


def _pairs(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (pair, condition) with the original and reversed ratings."""
    orig = df[df["scale_order"] == ScaleOrder.ORIGINAL.value]
    rev  = df[df["scale_order"] == ScaleOrder.REVERSED.value]
    return orig.merge(
        rev, on=_PAIR_KEYS, suffixes=("_orig", "_rev"), validate="one_to_one"
    )


def robustness_rate(df: pd.DataFrame, condition: str, tol: float = 0.5) -> float:
    """Fraction of pairs whose normalised rating survives the scale reversal."""
    merged = _pairs(df[df["condition"] == condition])
    if merged.empty:
        return float("nan")
    agree = (
        merged["rating_normalised_orig"] - merged["rating_normalised_rev"]
    ).abs() <= tol
    return float(agree.mean())


def delta_rr(df: pd.DataFrame, tol: float = 0.5) -> float:
    """dRR = RR_vision - RR_text_only. Negative implicates the vision encoder."""
    return (
        robustness_rate(df, Phase2Condition.VISION.value, tol)
        - robustness_rate(df, Phase2Condition.TEXT_ONLY.value, tol)
    )


def summary(df: pd.DataFrame, tol: float = 0.5) -> dict:
    baselines = compute_low_asi_baselines()
    vis = df[df["condition"] == Phase2Condition.VISION.value]
    per_query = (
        vis.groupby("query")["rating_normalised"].mean().to_dict() if not vis.empty else {}
    )
    return {
        "n_rows": int(len(df)),
        "delta_m": compute_delta_m(df, baselines),
        "rr_vision": robustness_rate(df, Phase2Condition.VISION.value, tol),
        "rr_text_only": robustness_rate(df, Phase2Condition.TEXT_ONLY.value, tol),
        "delta_rr": delta_rr(df, tol),
        # Reported alongside every rating: Phase 1 showed that a probe can look
        # healthy while sitting on a negligible slice of the distribution.
        "mean_captured_mass": float(df["captured_mass"].mean()),
        "human_low_asi_baselines": baselines,
        "model_mean_rating_by_query": per_query,
    }
