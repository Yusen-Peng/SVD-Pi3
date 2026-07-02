import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

csv_path = "ablation.csv"

keep_methods = ["Original", "Augmented", "fine-grained"]
metrics = ["Rel", "δ<1.25", "ATE", "RPE trans", "RPE rot"]
lower_better = {"Rel", "ATE", "RPE trans", "RPE rot"}

metric_labels = {
    "Rel": "Rel",
    "δ<1.25": r"$\delta < 1.25$",
    "ATE": "ATE",
    "RPE trans": "RPE-t",
    "RPE rot": "RPE-r",
}

method_labels = {
    "Original": "Default",
    "Augmented": "+ Visual Abstract",
    "fine-grained": "+ Continuous Mapping",
}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 11,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

df = pd.read_csv(csv_path)
df.columns = df.columns.str.strip()

df = df[df["Model"].isin(keep_methods)].copy()
df["Model"] = pd.Categorical(df["Model"], categories=keep_methods, ordered=True)
df = df.sort_values("Model")

base = df[df["Model"] == "Original"].iloc[0]

norm = df.copy()
for m in metrics:
    if m in lower_better:
        norm[m] = base[m] / df[m]
    else:
        norm[m] = df[m] / base[m]

# Auto radius range -- fixes the weird Visual Abstract issue
all_scores = norm[metrics].to_numpy(dtype=float).ravel()
r_min = np.floor((all_scores.min() - 0.01) * 100) / 100
r_max = 1.005

N = len(metrics)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
angles = np.concatenate([angles, [angles[0]]])

fig, ax = plt.subplots(figsize=(4.4, 4.1), subplot_kw={"polar": True})

ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

ax.set_ylim(r_min, r_max)
ax.set_yticks(np.round(np.linspace(r_min, 1.00, 4), 2))
ax.set_yticklabels([])
# ax.set_yticklabels([f"{x:.2f}" for x in np.round(np.linspace(r_min, 1.00, 4), 2)], fontsize=8)
ax.set_rlabel_position(88)

ax.set_xticks(angles[:-1])
ax.set_xticklabels([metric_labels[m] for m in metrics], fontsize=12)

ax.spines["polar"].set_color("0.55")
ax.spines["polar"].set_linewidth(0.9)
ax.grid(True, color="0.72", linewidth=0.75, alpha=0.8)

style_map = {
    "Original": dict(linewidth=2.8, marker="o", markersize=5.0, zorder=5),
    "Augmented": dict(linewidth=2.2, marker="s", markersize=4.5, zorder=4),
    "fine-grained": dict(linewidth=2.2, marker="^", markersize=5.0, zorder=3),
}

for _, row in norm.iterrows():
    method = row["Model"]
    values = row[metrics].to_numpy(dtype=float)
    values = np.concatenate([values, [values[0]]])

    ax.plot(
        angles,
        values,
        label=method_labels[method],
        **style_map[method],
    )

    # Very subtle fill; avoids ugly overlapping blobs
    ax.fill(
        angles,
        values,
        alpha=0.035 if method == "Original" else 0.025,
        zorder=style_map[method]["zorder"] - 1,
    )

ax.legend(
    loc="lower center",
    bbox_to_anchor=(0.5, -0.16),
    ncol=3,
    frameon=False,
    fontsize=10,
    handlelength=2.1,
    columnspacing=1.1,
)

plt.tight_layout()
plt.savefig("ablation_radar.pdf", bbox_inches="tight")
plt.savefig("ablation_radar.png", dpi=600, bbox_inches="tight")
plt.show()
