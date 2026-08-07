"""Generate publication-quality charts for the progress seminar slides.

Reads from outputs/phase1/*.json and findings/stimulus_validation/*.json,
writes transparent PNGs to outputs/phase1/charts/.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl

# ------------------------------------------------------------------ palette
NAVY      = "#1E2761"
NAVY_MID  = "#2B4396"
AMBER     = "#F4B942"
ICE_BLUE  = "#CADCFC"
GREEN     = "#16A34A"
RED       = "#DC2626"
GRAY_TEXT = "#475569"
GRAY_LINE = "#CBD5E1"

# ------------------------------------------------------------------ globals
mpl.rcParams["font.family"] = ["DejaVu Sans"]
mpl.rcParams["axes.edgecolor"] = GRAY_LINE
mpl.rcParams["axes.labelcolor"] = GRAY_TEXT
mpl.rcParams["xtick.color"] = GRAY_TEXT
mpl.rcParams["ytick.color"] = GRAY_TEXT

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "phase1" / "charts"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRAY_LINE)
    ax.spines["bottom"].set_color(GRAY_LINE)
    ax.tick_params(labelsize=11, colors=GRAY_TEXT)
    ax.patch.set_alpha(0)


def _save(fig, path: Path) -> None:
    fig.patch.set_alpha(0)
    fig.savefig(path, dpi=200, transparent=True, bbox_inches="tight")
    plt.close(fig)
    sz = path.stat().st_size
    print(f"  wrote {path.name} ({sz:,} bytes)")


def _load(p: str | Path) -> dict:
    return json.loads(Path(p).read_text())


# ------------------------------------------------------------------ chart 1
def chart_stimulus_validation():
    print("\n[chart 1] Stimulus validation")
    sil = _load(ROOT / "findings/stimulus_validation/silhouette_result.json")
    gray = _load(ROOT / "findings/stimulus_validation/gray_patch_result.json")
    print(f"  silhouette gender metric: {sil['gender']['metric']}")
    print(f"  gray patch gender metric: {gray['gender']['metric']}")

    labels = ["Humanoid Silhouette", "Gray Patch"]
    values = [sil["gender"]["metric"], gray["gender"]["metric"]]
    colors = [RED, GREEN]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.barh(labels, values, color=colors, edgecolor="none", height=0.55)

    ax.axvline(0.20, color=AMBER, linestyle="--", linewidth=1.5,
               label="Threshold (0.20)", zorder=0)

    # Value label: inside the bar (white) if there's room, else outside
    for bar, val in zip(bars, values):
        if val >= 0.18:
            ax.text(val - 0.015, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", ha="right",
                    fontsize=10, fontweight="bold", color="white")
        else:
            ax.text(val + 0.015, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", ha="left",
                    fontsize=10, fontweight="bold", color=GRAY_TEXT)

    annots = [("✗ REJECTED", RED), ("✓ ACCEPTED", GREEN)]
    for bar, (txt, c) in zip(bars, annots):
        ax.text(0.99, bar.get_y() + bar.get_height() / 2, txt,
                transform=ax.get_yaxis_transform(),
                va="center", ha="right",
                fontsize=11, fontweight="bold", color=c)

    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Gender Gap Score", fontsize=12)
    ax.set_title("Stimulus Validation: Gender Gap by Condition",
                 fontsize=14, fontweight="bold", color=NAVY, pad=14)
    ax.legend(loc="lower right", frameon=False, fontsize=10)
    _style_axes(ax)
    ax.invert_yaxis()
    _save(fig, OUT_DIR / "chart_stimulus_validation.png")


# ------------------------------------------------------------------ chart 2
def chart_raw_scores():
    print("\n[chart 2] Raw ASI scores")
    s = _load(ROOT / "outputs/phase1/llama_dev_summary.json")
    by_sub = s["by_subscale"]
    by_struct = s["by_structure"]
    print(f"  by_subscale = {by_sub}")
    print(f"  by_structure = {by_struct}")

    sub_labels = ["HS", "BS"]
    sub_vals = [by_sub["HS"], by_sub["BS"]]

    struct_order = ["direct", "inversion", "attribution", "hypothetical", "descriptive"]
    struct_labels = [s.title() for s in struct_order]
    struct_vals = [by_struct[s] for s in struct_order]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(9, 5),
        gridspec_kw={"width_ratios": [1, 2.4], "wspace": 0.35},
    )

    def _bar(ax, labels, vals, title):
        colors = [NAVY if v >= 0 else RED for v in vals]
        bars = ax.bar(labels, vals, color=colors, edgecolor="none", width=0.6)
        ax.axhline(0, color=GRAY_LINE, linestyle="--", linewidth=1.0, zorder=0)
        for b, v in zip(bars, vals):
            offset = 0.012 if v >= 0 else -0.012
            va = "bottom" if v >= 0 else "top"
            ax.text(b.get_x() + b.get_width() / 2, v + offset,
                    f"{v:+.3f}", ha="center", va=va,
                    fontsize=10, fontweight="bold", color=GRAY_TEXT)
        ax.set_title(title, fontsize=12, color=NAVY, fontweight="bold", pad=10)
        _style_axes(ax)

    _bar(ax1, sub_labels, sub_vals, "By Subscale")
    _bar(ax2, struct_labels, struct_vals, "By Prompt Structure")

    all_vals = sub_vals + struct_vals
    lo, hi = min(all_vals), max(all_vals)
    pad = max(abs(lo), abs(hi)) * 0.25 + 0.04
    for ax in (ax1, ax2):
        ax.set_ylim(min(lo, 0) - pad, max(hi, 0) + pad)
        ax.set_ylabel("Bias score (raw)", fontsize=11)

    for tick in ax2.get_xticklabels():
        tick.set_rotation(15)
        tick.set_ha("right")

    fig.suptitle("Phase 1 Raw Bias Scores — Llama-3.2-11B",
                 fontsize=14, fontweight="bold", color=NAVY, y=1.02)
    _save(fig, OUT_DIR / "chart_raw_scores.png")


# ------------------------------------------------------------------ chart 3
def chart_acquiescence():
    print("\n[chart 3] Acquiescence bias")
    a = _load(ROOT / "outputs/phase1/llama_dev_analysis.json")
    pyes = a["mean_p_yes_by_structure"]
    print(f"  mean_p_yes_by_structure = {pyes}")

    items = sorted(pyes.items(), key=lambda kv: kv[1], reverse=True)
    labels = [k.title() for k, _ in items]
    vals = [v for _, v in items]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.barh(labels, vals, color=NAVY_MID, edgecolor="none", height=0.6)

    ax.axvline(0.5, color=AMBER, linestyle="--", linewidth=1.5,
               label="Unbiased baseline (0.50)", zorder=0)

    for bar, val in zip(bars, vals):
        ax.text(val + 0.012, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", ha="left",
                fontsize=10, fontweight="bold", color=GRAY_TEXT)

    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Mean P(yes)", fontsize=12)
    ax.set_title("Acquiescence Bias: Mean P(yes) by Prompt Structure",
                 fontsize=14, fontweight="bold", color=NAVY, pad=14)
    ax.legend(loc="lower right", frameon=False, fontsize=10)
    _style_axes(ax)
    ax.invert_yaxis()

    fig.text(0.5, -0.02,
             "All structures show >50% agreement rate, including reverse-coded items.",
             ha="center", fontsize=9, color=GRAY_TEXT, style="italic")
    _save(fig, OUT_DIR / "chart_acquiescence.png")


# ------------------------------------------------------------------ chart 4
def chart_corrected():
    print("\n[chart 4] Corrected scores")
    a = _load(ROOT / "outputs/phase1/llama_dev_analysis.json")
    by_sub = a["by_subscale"]["bias_score_adj_struct"]
    print(f"  bias_score_adj_struct = {by_sub}")

    labels = ["HS", "BS"]
    vals = [by_sub["HS"], by_sub["BS"]]
    colors = [NAVY if v >= 0 else RED for v in vals]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, vals, color=colors, edgecolor="none", width=0.5)
    ax.axhline(0, color=GRAY_LINE, linestyle="--", linewidth=1.0, zorder=0)

    for b, v in zip(bars, vals):
        offset = 0.008 if v >= 0 else -0.008
        va = "bottom" if v >= 0 else "top"
        ax.text(b.get_x() + b.get_width() / 2, v + offset,
                f"{v:+.3f}", ha="center", va=va,
                fontsize=11, fontweight="bold", color=GRAY_TEXT)

    gap = abs(vals[0] - vals[1])
    y_top = max(vals) + 0.04
    ax.annotate(
        "", xy=(0, y_top), xytext=(1, y_top),
        arrowprops=dict(arrowstyle="<->", color=AMBER, lw=1.5),
    )
    ax.text(0.5, y_top + 0.01, f"Gap ≈ {gap:.2f}",
            ha="center", va="bottom",
            fontsize=11, fontweight="bold", color=AMBER)

    lo, hi = min(vals), max(vals)
    span = max(abs(lo), abs(hi))
    ax.set_ylim(lo - 0.06, hi + span * 0.6 + 0.06)
    ax.set_ylabel("Corrected Bias Score", fontsize=12)

    ax.set_title("Acquiescence-Corrected ASI Scores — Llama-3.2-11B",
                 fontsize=14, fontweight="bold", color=NAVY, pad=22)
    ax.text(0.5, 1.02,
            "Positive = sexist alignment; Negative = counter-sexist",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=11, color=GRAY_TEXT, style="italic")
    _style_axes(ax)
    _save(fig, OUT_DIR / "chart_corrected.png")


def main():
    print(f"Output dir: {OUT_DIR}")
    chart_stimulus_validation()
    chart_raw_scores()
    chart_acquiescence()
    chart_corrected()
    print("\nAll charts written.")


if __name__ == "__main__":
    main()
