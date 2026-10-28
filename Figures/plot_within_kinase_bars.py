"""
DiffPocket Table 3 — Horizontal-bar figure (Figure 7-style layout)

Creates a 4 × 2 grid:
  Rows: EGFR, KIT, FLT3, ABL1
  Left column: Pearson correlation (higher is better)
  Right column: RMSE (lower is better)

Outputs:
  diffpocket_table3_horizontal_bars.png  (600 DPI)
  diffpocket_table3_horizontal_bars.pdf  (vector PDF)

Install:
  pip install matplotlib numpy pandas
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

OUT_DIR = Path("diffpocket_table3_horizontal_bars")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Table 3 values used in the previous figure-gallery script
models = [
    "Ligand-Blind",
    "DeepDTA-style",
    "Independent-GAT",
    "No-Delta",
    "DiffPocket cross-attention",
    "DiffPocket rigid mean-pooling",
]
kinases = ["EGFR", "KIT", "FLT3", "ABL1"]

pearson = pd.DataFrame({
    "EGFR": [-0.087, 0.028, 0.034, 0.288, 0.490, 0.577],
    "KIT":  [ 0.339, 0.380, 0.461, 0.195, 0.351, 0.539],
    "FLT3": [ 0.246, 0.051, 0.064, 0.229, 0.183, 0.232],
    "ABL1": [ 0.529, 0.412, 0.465, 0.047, 0.285, 0.548],
}, index=models)

rmse = pd.DataFrame({
    "EGFR": [0.865, 0.857, 0.857, 0.820, 0.746, 0.698],
    "KIT":  [1.294, 1.485, 1.492, 1.342, 1.315, 1.130],
    "FLT3": [0.923, 0.980, 0.983, 0.929, 0.957, 0.936],
    "ABL1": [0.890, 0.967, 1.063, 1.045, 0.994, 0.858],
}, index=models)

# Restrained academic palette: muted baselines, teal for DiffPocket.
COLORS = {
    "Ligand-Blind": "#C49A6C",                   # caramel
    "DeepDTA-style": "#BDBDBD",                  # grey
    "Independent-GAT": "#7891A8",                # muted blue
    "No-Delta": "#A58AA0",                       # dusty plum
    "DiffPocket cross-attention": "#8BA6A0",      # grey-green
    "DiffPocket rigid mean-pooling": "#24476B",  # deep navy
}
TEXT = "#000000"
GRID = "#DCE2E7"
AXIS = "#000000"
ZERO = "#000000"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8.5,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.titlesize": 13,
    "text.color": TEXT,
    "axes.labelcolor": TEXT,
    "xtick.color": TEXT,
    "ytick.color": TEXT,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.75,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.edgecolor": "white",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

fig, axes = plt.subplots(
    4, 2, figsize=(14, 11.2),
    gridspec_kw={"width_ratios": [1.0, 1.0]},
)
y_positions = np.arange(len(models))
bar_height = 0.70

for row, kinase in enumerate(kinases):
    # Pearson correlation panel
    ax = axes[row, 0]
    vals = pearson[kinase].to_numpy()
    ax.barh(y_positions, vals, height=bar_height,
            color=[COLORS[m] for m in models],
            edgecolor="white", linewidth=0.5, zorder=3)
    ax.axvline(0, color=ZERO, linewidth=0.9, zorder=2)
    ax.set_xlim(-0.2, 0.72)
    ax.xaxis.set_major_locator(MultipleLocator(0.1))
    ax.set_yticks(y_positions)
    ax.set_yticklabels(models)
    ax.invert_yaxis()
    ax.set_title(f"{kinase}",
                  loc="center", fontweight="bold", pad=7, fontsize=13)
    ax.set_xlabel("Pearson correlation ↑", fontsize=12, labelpad=7)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=GRID, linewidth=0.65, zorder=0)
    ax.tick_params(axis="y", labelsize=10, length=0, pad=6)

    for y, value in zip(y_positions, vals):
        if value >= 0:
            ax.annotate(f"{value:.3f}", (value, y), xytext=(4, 0),
                        textcoords="offset points", ha="left", va="center",
                        fontsize=11, color=TEXT, clip_on=False)
        else:
            ax.annotate(f"{value:.3f}", (value, y), xytext=(-4, 0),
                        textcoords="offset points", ha="right", va="center",
                        fontsize=11, color=TEXT, clip_on=False)

    # RMSE panel
    ax = axes[row, 1]
    vals = rmse[kinase].to_numpy()
    ax.barh(y_positions, vals, height=bar_height,
            color=[COLORS[m] for m in models],
            edgecolor="white", linewidth=0.5, zorder=3)
    ax.set_xlim(0, 1.62)
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.set_yticks(y_positions)
    ax.set_yticklabels(models)
    ax.invert_yaxis()
    ax.set_title(f"{kinase} ",
                  loc="center", fontweight="bold", pad=7,fontsize=13)
    ax.set_xlabel("RMSE (kcal/mol) ↓", fontsize=12, labelpad=7)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=GRID, linewidth=0.65, zorder=0)
    ax.tick_params(axis="y", labelsize=10, length=0, pad=6)
    

    for y, value in zip(y_positions, vals):
        ax.annotate(f"{value:.3f}", (value, y), xytext=(4, 0),
                    textcoords="offset points", ha="left", va="center",
                    fontsize=11, color=TEXT, clip_on=False)

fig.tight_layout(rect=(0.015, 0.015, 0.985, 0.975), h_pad=1.65, w_pad=1.4)

out_png = OUT_DIR / "diffpocket_table3_horizontal_bars.png"
out_pdf = OUT_DIR / "diffpocket_table3_horizontal_bars.pdf"
fig.savefig(out_png, dpi=2100, bbox_inches="tight")
fig.savefig(out_pdf, dpi=2100, bbox_inches="tight")
plt.close(fig)

print(f"PNG saved to: {out_png.resolve()}")
print(f"PDF saved to: {out_pdf.resolve()}")
print("Use the PDF for vector-quality manuscript submission where accepted.")
