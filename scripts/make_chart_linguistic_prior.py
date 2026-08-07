import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

labels = ["Humanoid Silhouette", "Gray Patch"]
values = [0.547, 0.710]
colors = ["steelblue", "#E8A020"]

fig, ax = plt.subplots(figsize=(9, 4), facecolor="white")
ax.set_facecolor("white")

bars = ax.barh(labels, values, color=colors, height=0.5, zorder=3)

ax.axvline(x=0.50, color="gray", linestyle="--", linewidth=1.2, zorder=4)
ax.text(0.502, 1.52, "threshold", color="gray", fontsize=9, va="bottom")

# Value labels on bars
for bar, val in zip(bars, values):
    ax.text(val - 0.015, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", ha="right", va="center", fontsize=11,
            color="white", fontweight="bold")

# Annotation arrow on gray patch bar (index 1, top bar)
gray_bar = bars[1]
arrow_y = gray_bar.get_y() + gray_bar.get_height() / 2
ax.annotate(
    "higher on featureless patch →\nlanguage prior, not visual signal",
    xy=(values[1], arrow_y),
    xytext=(values[1] + 0.04, arrow_y),
    fontsize=8.5,
    color="#B05000",
    va="center",
    arrowprops=dict(arrowstyle="->", color="#B05000", lw=1.4),
)

ax.set_xlim(0.0, 1.0)
ax.set_xlabel("P(white)", fontsize=11)
ax.set_xticks([0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0])
ax.tick_params(axis="y", labelsize=11)
ax.grid(axis="x", linestyle=":", alpha=0.5, zorder=0)
ax.set_axisbelow(True)

fig.suptitle("P(white) by Stimulus — Linguistic Prior", fontsize=13, fontweight="bold", y=1.01)
ax.set_title(
    "Llama-3.2-11B · dev run · job 24256941 · 5 May 2026",
    fontsize=9, color="gray", pad=6,
)

plt.tight_layout()
plt.savefig(
    "/Users/hunghuynh/Code/VLM_Audit/outputs/phase1/charts/chart_linguistic_prior.png",
    dpi=200, bbox_inches="tight", facecolor="white",
)
print("Saved.")
