"""
Task 4: Visual Integrity Audit
Reject 3D bar chart; implement Small Multiples approach.
Pollution vs Population Density vs Region — Sequential colormap.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

Path("outputs").mkdir(exist_ok=True)


def run_task4(df: pd.DataFrame, station_meta: pd.DataFrame):
    print("\n[Task 4] Visual Integrity Audit — Small Multiples ...")

    # Station-level summary
    station_summary = df.groupby("station_id")["PM25"].mean().reset_index()
    station_summary.columns = ["station_id", "mean_PM25"]
    merged = station_summary.merge(station_meta, on="station_id")

    regions = sorted(merged["region"].unique())

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor("white")
    axes = axes.flatten()

    cmap = plt.get_cmap("YlOrRd")  # Sequential — perceptually uniform luminance
    norm = mcolors.Normalize(vmin=merged["mean_PM25"].min(), vmax=merged["mean_PM25"].max())

    for i, region in enumerate(regions):
        ax = axes[i]
        subset = merged[merged["region"] == region]

        sc = ax.scatter(
            subset["population_density"],
            subset["mean_PM25"],
            c=subset["mean_PM25"],
            cmap=cmap, norm=norm,
            s=80, alpha=0.85, edgecolors="white", linewidths=0.5
        )

        # Zone labels
        for zone_label, marker in [("Industrial", "^"), ("Residential", "o")]:
            z = subset[subset["zone"] == zone_label]
            ax.scatter(z["population_density"], z["mean_PM25"],
                       marker=marker, c=z["mean_PM25"], cmap=cmap, norm=norm,
                       s=90, alpha=0.9, edgecolors="#333", linewidths=0.5,
                       label=zone_label)

        ax.set_title(f"Region: {region}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Population Density (people/km²)", fontsize=9)
        ax.set_ylabel("Mean PM2.5 (µg/m³)", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_facecolor("#f9f9f9")
        ax.legend(fontsize=8, markerscale=1)
        ax.axhline(35, color="#d62728", lw=1, linestyle="--", alpha=0.6)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.02)
    cb.set_label("Mean PM2.5 (µg/m³)", fontsize=10)

    fig.suptitle(
        "Pollution vs Population Density vs Region\n"
        "Small Multiples — Sequential (YlOrRd) Color Scale\n"
        "[Replaces rejected 3D bar chart — eliminates occlusion and scale distortion]",
        fontsize=12, fontweight="bold", y=1.01
    )

    plt.tight_layout()
    out = "outputs/task4_small_multiples.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")
    print("  Justification: Sequential 'YlOrRd' preserves perceptual luminance gradient.")
    print("  Rainbow colormaps create false discontinuities — rejected.")


if __name__ == "__main__":
    from data import load_dataset
    import pandas as pd
    df = load_dataset()
    meta = pd.read_csv("data/station_metadata.csv")
    run_task4(df, meta)