import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

# Data
methods = [
    "Agentless (Embedding)",
    "Agentless (LLM)",
    "Agentless (Combined)",
    "CoSIL",
    "BugCerberus",
    "RAGent",
]
top1 = [0.407, 0.567, 0.623, 0.607, 0.651, 0.710]
top5 = [0.743, 0.757, 0.850, 0.827, 0.754, 0.903]
top10 = [0.827, np.nan, 0.857, np.nan, 0.791, 0.943]
metrics = [top1, top5, top10]
titles = ["Top-1", "Top-5", "Top-10"]

# Style
plt.rcParams.update(
    {
        "font.size": 10,
        "font.family": "serif",
        "axes.linewidth": 0.8,
    }
)

# Colors for each method (not each metric)
method_colors = [
    "#4C72B0",
    "#DD8452",
    "#55A868",
    "#C44E52",
    "#8172B3",
    "#937860",
]
bar_width = 0.7
y_lims = [(0.3, 0.8), (0.4, 1.0), (0.5, 1.0)]

fig, axes = plt.subplots(1, 3, figsize=(8, 2.8))

x = np.arange(len(methods))

# Plot bars for each subplot
for i, ax in enumerate(axes):
    # Sort data by score for this metric
    metric_data = np.array(metrics[i])
    sorted_indices = np.argsort(metric_data)  # Sort ascending
    sorted_values = metric_data[sorted_indices]
    sorted_colors = [method_colors[idx] for idx in sorted_indices]

    # Create bars with sorted colors
    bars = ax.bar(
        x,
        sorted_values,
        color=sorted_colors,
        edgecolor="black",
        linewidth=0.4,
        width=bar_width,
    )
    ax.set_title(titles[i], fontsize=10)
    ax.set_xticks([])
    ax.set_ylabel("Score", fontsize=10)
    ax.set_ylim(y_lims[i])
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# Create legend handles for methods
legend_handles = [
    Patch(facecolor=method_colors[i], edgecolor="black", linewidth=0.4)
    for i in range(len(methods))
]

# Unified legend showing method names (centered below plots)
fig.legend(
    legend_handles,
    methods,
    loc="lower center",
    ncol=3,
    frameon=False,
    fontsize=8,
    bbox_to_anchor=(0.5, -0.05),
)

plt.tight_layout()
fig.subplots_adjust(bottom=0.25)  # Make space for legend
plt.savefig("loc-results-compact.pdf", dpi=600, bbox_inches="tight")
plt.close()
