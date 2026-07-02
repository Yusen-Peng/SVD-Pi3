import os
import pandas as pd
import matplotlib.pyplot as plt

CSV_FILES = {
    "Pi3": "depth_pi3.csv",
    "VGGT": "depth_vggt.csv",
}

OUT_DIR = "."
OUT_PDF = os.path.join(OUT_DIR, "svd3_depth_tradeoff_side_by_side.pdf")
OUT_PNG = os.path.join(OUT_DIR, "svd3_depth_tradeoff_side_by_side.png")

groups = {
    "Original": ["Original"],
    "SVD": ["SVD-60", "SVD-70", "SVD-80"],
    "W-SVD": ["W-SVD-60", "W-SVD-70", "W-SVD-80"],
    "SVD$^3$": ["Ours-60", "Ours-70", "Ours-80"],
}

markers = {
    "Original": "o",
    "SVD": "s",
    "W-SVD": "^",
    "SVD$^3$": "D",
}

def load_and_score(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    base = df[df["Model"] == "Original"].iloc[0]

    rel_cols = [c for c in df.columns if c.endswith("-rel")]
    delta_cols = [c for c in df.columns if c.endswith("-delta")]

    df["Speedup"] = base["GFLOPs"] / df["GFLOPs"]

    for c in rel_cols:
        df[c + "_score"] = base[c] / df[c]      # lower is better

    for c in delta_cols:
        df[c + "_score"] = df[c] / base[c]      # higher is better

    score_cols = [c + "_score" for c in rel_cols + delta_cols]
    df["Overall"] = df[score_cols].mean(axis=1)

    return df

fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), sharey=True)

for ax, (model_name, csv_path) in zip(axes, CSV_FILES.items()):
    df = load_and_score(csv_path)

    for name, models in groups.items():
        sub = df[df["Model"].isin(models)].sort_values("Speedup")

        ax.plot(
            sub["Speedup"],
            sub["Overall"],
            marker=markers[name],
            linewidth=2.5,
            markersize=8,
            alpha=0.85,
            label=name,
        )

        for _, r in sub.iterrows():
            label = r["Model"].replace("Ours", "SVD$^3$")
            ax.annotate(
                label,
                (r["Speedup"], r["Overall"]),
                textcoords="offset points",
                xytext=(8, -6),
                fontsize=11,
            )

    ax.axhline(1.0, linestyle="--", linewidth=1.3, alpha=0.6)
    ax.set_title(f"{model_name} Depth Performance", fontsize=18, fontweight="bold")
    ax.set_xlabel("TFLOP Speedup over Original", fontsize=15)
    ax.grid(True, alpha=0.25)

axes[0].set_ylabel("Average Relative Performance", fontsize=15)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="lower center",
    ncol=4,
    fontsize=13,
    bbox_to_anchor=(0.5, -0.08),
    frameon=True,
)

plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig(OUT_PDF, bbox_inches="tight")
plt.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
plt.show()