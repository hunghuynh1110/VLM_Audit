"""
Charts for the 90B results after the cross-GPU fix.

Every number is read from the run outputs -- nothing here is typed in by hand.

    python scripts/make_90b_result_charts.py --data-dir <dir> --out-dir <dir>

Produces three figures:
    1_validity.png   the broken vs fixed runs, showing the old output carried
                     no information about its input
    2_phase2.png     model vs human objectivity ratings, and scale-reversal
                     robustness with and without an image
    3_phase1.png     bias score by prompt structure, with the captured_mass
                     caveat that governs how far the condition split can be read
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Validated with the dataviz skill's validate_palette.js (light, surface #fcfcfb):
# blue/orange and blue/red both PASS all six checks.
BLUE, ORANGE, RED = "#2a78d6", "#eb6834", "#e34948"
SURFACE = "#fcfcfb"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.size": 9, "text.color": INK,
    "axes.labelcolor": INK2, "axes.edgecolor": BASELINE,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": GRID, "grid.linewidth": 0.6,
})


def _style(ax, xgrid=False, ygrid=False):
    ax.set_axisbelow(True)
    ax.grid(axis="x" if xgrid else "y", visible=xgrid or ygrid)
    ax.tick_params(length=0)


# ----------------------------------------------------------------- figure 1
def figure_validity(d: Path, out: Path) -> None:
    p1_new = pd.read_parquet(d / "llama.parquet")
    p1_old = pd.read_parquet(d / "llama.invalid_multi_gpu.parquet")
    p2_new = pd.read_parquet(d / "llama_historical_2018.parquet")
    p2_old = pd.read_parquet(d / "llama_historical_2018.invalid_multi_gpu.parquet")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1))
    fig.suptitle("The old runs produced the same answer regardless of the prompt",
                 fontsize=13, fontweight="bold", x=0.5, y=0.99)

    for ax, old, new, col, lo, hi, title, ylab, ref, reflab in [
        (axes[0], p1_old["p_yes"], p1_new["p_yes"], "p_yes", 0, 1,
         "Phase 1  ·  330 prompts", "p(yes)", 0.5, "0.5 = no information"),
        (axes[1], p2_old["rating_expected"], p2_new["rating_expected"],
         "rating_expected", 1, 7, "Phase 2  ·  360 image ratings",
         "expected rating (1-7)", 4.0, "4.0 = scale midpoint"),
    ]:
        x = np.arange(len(new))
        # The reference line is described in the legend rather than annotated on
        # the plot: these panels are dense enough that any inline label lands on
        # top of data.
        ax.axhline(ref, color=BASELINE, lw=1, ls="--", zorder=1, label=reflab)
        ax.scatter(x, old, s=7, color=RED, alpha=0.85, zorder=3,
                   label=f"before fix ({old.nunique()} distinct)")
        ax.scatter(x, new, s=7, color=BLUE, alpha=0.55, zorder=2,
                   label=f"after fix ({new.nunique()} distinct)")
        ax.set_title(title, fontsize=10, color=INK, loc="left", pad=34)
        ax.set_xlabel("prompt (in run order)")
        ax.set_ylabel(ylab)
        ax.set_ylim(lo - (hi - lo) * 0.04, hi + (hi - lo) * 0.04)
        h, l = ax.get_legend_handles_labels()
        order = [1, 2, 0]                      # before, after, then the reference line
        ax.legend([h[i] for i in order], [l[i] for i in order],
                  loc="lower left", bbox_to_anchor=(0, 1.005), ncol=2,
                  frameon=False, fontsize=7.8, handletextpad=0.4,
                  columnspacing=1.4, labelcolor=INK2)
        _style(ax, ygrid=True)

    fig.tight_layout(rect=(0, 0.02, 1, 0.93))
    fig.savefig(out / "1_validity.png", dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------- figure 2
def figure_phase2(d: Path, out: Path) -> None:
    df = pd.read_parquet(d / "llama_historical_2018.parquet")
    summary = json.loads((d / "llama_historical_2018_summary.json").read_text())
    human = summary["human_low_asi_baselines"]

    vis = df[df["condition"] == "vision"].copy()
    vis["human"] = vis["query"].map(human)
    g = (vis.groupby("query")
            .agg(model=("rating_normalised", "mean"), human=("human", "first")))
    g["delta"] = g["model"] - g["human"]
    g = g.sort_values("delta")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6),
                             gridspec_kw={"width_ratios": [1.75, 1]})

    # --- panel A: paired dot plot, model vs human, per query
    ax = axes[0]
    y = np.arange(len(g))
    ax.hlines(y, g["human"], g["model"], color=BASELINE, lw=1.6, zorder=1)
    ax.scatter(g["human"], y, s=52, color=ORANGE, zorder=3, label="human (low-ASI)")
    ax.scatter(g["model"], y, s=52, color=BLUE, zorder=3, label="Llama-3.2-90B")
    ax.set_yticks(y)
    ax.set_yticklabels(
        [q + ("  ← control" if q == "hot air baloon" else "") for q in g.index],
        fontsize=8.5, color=INK2)
    for i, dv in enumerate(g["delta"]):
        ax.annotate(f"{dv:+.2f}", xy=(max(g['human'].max(), g['model'].max()) + 0.25, i),
                    va="center", fontsize=7.5,
                    color=INK2 if abs(dv) > 0.2 else MUTED)
    ax.set_xlim(2.3, 7.15)
    ax.set_xlabel("mean objectivity rating  (1-7, higher = more objective)")
    ax.set_title("The model rates people as less objective than humans do —\n"
                 "but agrees on the non-person control",
                 fontsize=10.5, color=INK, loc="left", pad=10)
    ax.legend(loc="upper left", frameon=False, fontsize=8.5, labelcolor=INK2)
    _style(ax, xgrid=True)

    # --- panel B: robustness rate, text-only vs vision
    ax = axes[1]
    rr = [summary["rr_text_only"], summary["rr_vision"]]
    bars = ax.bar(["text only", "with image"], rr, width=0.5,
                  color=[BLUE, RED], zorder=3)
    for b, v in zip(bars, rr):
        ax.annotate(f"{v:.0%}", xy=(b.get_x() + b.get_width() / 2, v),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=11, fontweight="bold", color=INK)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("ratings that survive flipping the scale")
    ax.set_title("Adding an image destroys\nscale comprehension",
                 fontsize=10.5, color=INK, loc="left", pad=10)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    _style(ax, ygrid=True)

    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(out / "2_phase2.png", dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------- figure 3
def figure_phase1(d: Path, out: Path) -> None:
    df = pd.read_parquet(d / "llama.parquet")
    by_s = df.groupby("structure")["bias_score"].mean().sort_values()
    overall = df["bias_score"].mean()
    cap = df.groupby("condition")["captured_mass"].mean().sort_values(ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3),
                             gridspec_kw={"width_ratios": [1.5, 1]})

    # --- panel A: bias score by structure (sign carries the meaning -> diverging)
    ax = axes[0]
    colors = [RED if v < 0 else BLUE for v in by_s]
    bars = ax.barh(by_s.index, by_s.values, color=colors, height=0.6, zorder=3)
    ax.axvline(0, color=BASELINE, lw=1.2, zorder=2)
    ax.axvline(overall, color=MUTED, lw=1, ls="--", zorder=2)
    # Anchored in axis-fraction y so it cannot be clipped off the top.
    ax.text(overall, 1.01, f"overall {overall:+.3f}",
            transform=ax.get_xaxis_transform(), ha="center", va="bottom",
            fontsize=7.5, color=MUTED)
    for b, v in zip(bars, by_s.values):
        ax.annotate(f"{v:+.3f}", xy=(v, b.get_y() + b.get_height() / 2),
                    xytext=(5 if v >= 0 else -5, 0), textcoords="offset points",
                    va="center", ha="left" if v >= 0 else "right",
                    fontsize=8.5, color=INK2)
    ax.set_xlim(-0.34, 0.40)
    ax.set_xlabel("mean bias score   (0 = no lean)")
    ax.set_title("'Inversion' asks the same thing backwards —\n"
                 "and flips sign, which is acquiescence, not belief",
                 fontsize=10.5, color=INK, loc="left", pad=10)
    ax.tick_params(axis="y", labelsize=9)
    _style(ax, xgrid=True)

    # --- panel B: captured_mass caveat
    ax = axes[1]
    bars = ax.bar(cap.index, cap.values, width=0.55, color=BLUE, zorder=3)
    for b, v in zip(bars, cap.values):
        ax.annotate(f"{v:.2f}", xy=(b.get_x() + b.get_width() / 2, v),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=9.5, color=INK)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("mean captured_mass")
    ax.set_title("Caveat: the vision rows measure\nfar less of the model's behaviour",
                 fontsize=10.5, color=INK, loc="left", pad=10)
    _style(ax, ygrid=True)

    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(out / "3_phase1.png", dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    figure_validity(a.data_dir, a.out_dir)
    figure_phase2(a.data_dir, a.out_dir)
    figure_phase1(a.data_dir, a.out_dir)
    print(f"wrote 3 figures to {a.out_dir}")


if __name__ == "__main__":
    main()
