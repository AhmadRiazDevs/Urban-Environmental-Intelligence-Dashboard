"""
Task 1: Dimensionality Challenge — PCA on 6 environmental variables
Visualize Industrial vs Residential zone clustering in 2D PCA space.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pathlib import Path

Path("outputs").mkdir(exist_ok=True)


def run_task1(df: pd.DataFrame):
    print("\n[Task 1] PCA Dimensionality Reduction ...")

    features = ["PM25", "PM10", "NO2", "Ozone", "Temperature", "Humidity"]

    # --- Aggregate to station-level means to reduce to 100 points for clarity
    station_agg = df.groupby(["station_id", "zone"])[features].mean().reset_index()

    X = station_agg[features].values
    zones = station_agg["zone"].values

    # Step 1 — Standardise
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Step 2 — PCA
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    # Loadings
    loadings = pd.DataFrame(
        pca.components_.T,
        index=features,
        columns=["PC1", "PC2"]
    )

    explained = pca.explained_variance_ratio_ * 100

    # ---- Plot ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("white")

    colors = {"Industrial": "#d62728", "Residential": "#1f77b4"}
    ax = axes[0]
    for zone, color in colors.items():
        mask = zones == zone
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=color, label=zone, alpha=0.85, edgecolors="white",
                   linewidths=0.5, s=80)
    ax.set_xlabel(f"PC1 ({explained[0]:.1f}% variance)", fontsize=11)
    ax.set_ylabel(f"PC2 ({explained[1]:.1f}% variance)", fontsize=11)
    ax.set_title("PCA — Industrial vs Residential Zones", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_facecolor("#f9f9f9")

    # Biplot — loadings
    ax2 = axes[1]
    for i, feat in enumerate(features):
        ax2.annotate("", xy=(pca.components_[0, i] * 3, pca.components_[1, i] * 3),
                     xytext=(0, 0),
                     arrowprops=dict(arrowstyle="->", color="#333333", lw=1.8))
        ax2.text(pca.components_[0, i] * 3.2, pca.components_[1, i] * 3.2,
                 feat, fontsize=10, ha="center", va="center", color="#222222")
    ax2.set_xlim(-4, 4)
    ax2.set_ylim(-4, 4)
    ax2.axhline(0, color="#cccccc", lw=0.8)
    ax2.axvline(0, color="#cccccc", lw=0.8)
    ax2.set_xlabel(f"PC1 ({explained[0]:.1f}%)", fontsize=11)
    ax2.set_ylabel(f"PC2 ({explained[1]:.1f}%)", fontsize=11)
    ax2.set_title("PCA Loadings Biplot", fontsize=13, fontweight="bold")
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.set_facecolor("#f9f9f9")

    plt.tight_layout()
    out = "outputs/task1_pca.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")

    print("\n  PCA Loadings:")
    print(loadings.round(3))
    print(f"\n  Explained Variance — PC1: {explained[0]:.1f}%  PC2: {explained[1]:.1f}%")

    return loadings, explained


if __name__ == "__main__":
    from data import load_dataset
    df = load_dataset()
    run_task1(df)
    