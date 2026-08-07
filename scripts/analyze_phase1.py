"""
Phase 1 analysis: raw + acquiescence-corrected ASI scores.

Acquiescence correction: for each row, subtract the model's per-structure
baseline yes-rate from p_yes before applying polarity and structure_sign.
This estimates "what bias_score would we see if the model were equally
likely to say yes/no by default within this structure?"

We also report log-odds-corrected scores (logit(p_yes) - mean(logit(p_yes))
within structure), which is the more standard psychometric correction
because it doesn't compress at the tails.

Usage:
    python scripts/analyze_phase1.py --parquet outputs/phase1/llama_dev.parquet
    python scripts/analyze_phase1.py  # uses default path
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

INVERSION = "inversion"


def _structure_sign(structure: str) -> int:
    return -1 if structure == INVERSION else 1


def _logit(p: float, eps: float = 1e-6) -> float:
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))


def add_corrections(df: pd.DataFrame) -> pd.DataFrame:
    """Add three new columns:
       - p_yes_adj_struct:    p_yes - mean p_yes within (model_id, structure)
       - p_yes_adj_struct_cond: p_yes - mean p_yes within (model_id, structure, condition)
       - logit_yes_adj:       logit(p_yes) - mean logit(p_yes) within (model_id, structure)
       And three corresponding bias_score variants.
    """
    df = df.copy()

    # Per-structure baseline (the simplest acquiescence correction)
    base_struct = df.groupby(["model_id", "structure"])["p_yes"].transform("mean")
    df["p_yes_baseline_struct"] = base_struct
    df["p_yes_adj_struct"] = df["p_yes"] - base_struct

    # Per-(structure, condition) baseline — also controls for image-condition effects
    base_sc = df.groupby(["model_id", "structure", "condition"])["p_yes"].transform("mean")
    df["p_yes_baseline_struct_cond"] = base_sc
    df["p_yes_adj_struct_cond"] = df["p_yes"] - base_sc

    # Log-odds correction (more standard psychometrically)
    df["logit_yes"] = df["p_yes"].apply(_logit)
    base_logit = df.groupby(["model_id", "structure"])["logit_yes"].transform("mean")
    df["logit_yes_baseline_struct"] = base_logit
    df["logit_yes_adj"] = df["logit_yes"] - base_logit

    # Apply polarity * structure_sign to each correction
    sign = df["polarity"] * df["structure"].map(_structure_sign)
    df["bias_score_raw"] = sign * df["p_yes"]            # should match existing bias_score
    df["bias_score_adj_struct"] = sign * df["p_yes_adj_struct"]
    df["bias_score_adj_struct_cond"] = sign * df["p_yes_adj_struct_cond"]
    df["bias_score_adj_logit"] = sign * df["logit_yes_adj"]

    return df


def summarize(df: pd.DataFrame) -> dict:
    score_cols = [
        "bias_score_raw",
        "bias_score_adj_struct",
        "bias_score_adj_struct_cond",
        "bias_score_adj_logit",
    ]
    out: dict = {"n_rows": len(df)}

    out["overall"] = {c: float(df[c].mean()) for c in score_cols}

    out["mean_p_yes_overall"] = float(df["p_yes"].mean())
    out["mean_p_yes_by_structure"] = (
        df.groupby("structure")["p_yes"].mean().round(4).to_dict()
    )
    out["mean_p_yes_by_structure_condition"] = {
        f"{s}|{c}": round(v, 4)
        for (s, c), v in df.groupby(["structure", "condition"])["p_yes"].mean().items()
    }

    out["by_structure"] = {
        c: df.groupby("structure")[c].mean().round(4).to_dict() for c in score_cols
    }
    out["by_condition"] = {
        c: df.groupby("condition")[c].mean().round(4).to_dict() for c in score_cols
    }
    out["by_subscale"] = {
        c: df.groupby("subscale")[c].mean().round(4).to_dict() for c in score_cols
    }
    return out


def print_report(summary: dict) -> None:
    print("=" * 70)
    print(f"PHASE 1 ANALYSIS  (n={summary['n_rows']} rows)")
    print("=" * 70)

    print("\n-- Acquiescence baseline --")
    print(f"  overall mean p_yes = {summary['mean_p_yes_overall']:.3f}")
    print("  per-structure mean p_yes:")
    for s, v in summary["mean_p_yes_by_structure"].items():
        print(f"    {s:14s} {v:.3f}")

    print("\n-- Overall ASI scores --")
    for k, v in summary["overall"].items():
        print(f"  {k:32s} = {v:+.4f}")

    def _print_table(title: str, table: dict):
        print(f"\n-- {title} --")
        cols = list(table.keys())
        rows = sorted({r for c in cols for r in table[c].keys()})
        header = f"  {'group':14s}" + "".join(f"{c.replace('bias_score_',''):>22s}" for c in cols)
        print(header)
        for r in rows:
            line = f"  {r:14s}" + "".join(f"{table[c].get(r, float('nan')):>22.4f}" for c in cols)
            print(line)

    _print_table("Mean bias_score by structure", summary["by_structure"])
    _print_table("Mean bias_score by condition", summary["by_condition"])
    _print_table("Mean bias_score by subscale",  summary["by_subscale"])
    print()


def _load_dataframe(parquet_path: Path) -> pd.DataFrame:
    """Try parquet; fall back to JSONL checkpoint if pyarrow is missing."""
    try:
        return pd.read_parquet(parquet_path)
    except (ImportError, ValueError) as e:
        jsonl_path = parquet_path.with_name(parquet_path.stem + "_checkpoint.jsonl")
        if not jsonl_path.exists():
            raise SystemExit(f"parquet failed ({e}) and no fallback at {jsonl_path}")
        print(f"[load] parquet engine unavailable, using {jsonl_path}")
        rows = [json.loads(l) for l in open(jsonl_path) if l.strip()]
        return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="outputs/phase1/llama_dev.parquet")
    ap.add_argument("--out-json", default=None,
                    help="Optional path to write summary JSON. Defaults to <parquet>_analysis.json")
    args = ap.parse_args()

    parquet_path = Path(args.parquet)
    if not parquet_path.exists() and not parquet_path.with_name(parquet_path.stem + "_checkpoint.jsonl").exists():
        raise SystemExit(f"neither parquet nor checkpoint found at {parquet_path.parent}")

    df = _load_dataframe(parquet_path)
    df = add_corrections(df)

    summary = summarize(df)
    print_report(summary)

    out_json = Path(args.out_json) if args.out_json else parquet_path.with_name(
        parquet_path.stem + "_analysis.json"
    )
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote summary → {out_json}")


if __name__ == "__main__":
    main()
