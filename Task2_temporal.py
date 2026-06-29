"""
Task 2: High-Density Temporal Analysis
Heatmap of PM2.5 Health Threshold Violations across 100 sensors over 2025.
Avoids overplotting 100 lines — uses a compact horizon/heatmap.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

Path("outputs").mkdir(exist_ok=True)

THRESHOLD = 35.0  # µg/m³


def run_task2(df: pd.DataFrame):
    print("\n[Task 2] High-Density Temporal Heatmap ...")

    df["date"] = pd.to_datetime(df["timestamp"]).dt.date
    df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour

    # --- Daily max PM2.5 per station
    daily = df.groupby(["station_id", "date"])["PM25"].max().reset_index()
    daily["violation"] = (daily["PM25"] > THRESHOLD).astype(float)

    # Pivot: stations × days
    pivot = daily.pivot(index="station_id", columns="date", values="PM25").fillna(0)

    # Sort stations by mean PM25 (worst on top)
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]

    # ---- Plot 1: PM2.5 Heatmap ----
    fig, axes = plt.subplots(2, 1, figsize=(18, 10))
    fig.patch.set_facecolor("white")

    cmap = plt.get_cmap("YlOrRd")
    norm = mcolors.Normalize(vmin=0, vmax=150)

    ax = axes[0]
    im = ax.imshow(pivot.values, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_yticks(range(0, 100, 10))
    ax.set_yticklabels([pivot.index[i] for i in range(0, 100, 10)], fontsize=7)
    # X ticks — monthly
    dates = list(pivot.columns)
    month_ticks = [i for i, d in enumerate(dates) if pd.Timestamp(d).day == 1]
    month_labels = [pd.Timestamp(dates[i]).strftime("%b") for i in month_ticks]
    ax.set_xticks(month_ticks)
    ax.set_xticklabels(month_labels, fontsize=9)
    ax.set_title("PM2.5 Daily Max — All 100 Stations (2025)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Station (sorted by avg PM2.5)", fontsize=10)
    cb = plt.colorbar(im, ax=ax, fraction=0.015, pad=0.01)
    cb.set_label("PM2.5 (µg/m³)", fontsize=9)
    ax.axhline(49.5, color="white", lw=1.5, linestyle="--")
    ax.text(5, 52, "← Residential | Industrial →", color="white", fontsize=8)
    ax.spines[["top", "right", "bottom", "left"]].set_visible(False)

    # ---- Plot 2: Hourly violation frequency (24h pattern) ----
    hourly_viol = df[df["PM25"] > THRESHOLD].groupby("hour").size() / df.groupby("hour").size()
    ax2 = axes[1]
    ax2.bar(hourly_viol.index, hourly_viol.values, color="#d62728", alpha=0.8, width=0.8)
    ax2.set_xlabel("Hour of Day (0–23)", fontsize=10)
    ax2.set_ylabel("Violation Rate", fontsize=10)
    ax2.set_title("Hourly PM2.5 Violation Rate — Reveals 24-Hour Diurnal Signature", fontsize=12, fontweight="bold")
    ax2.set_xticks(range(0, 24))
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.set_facecolor("#f9f9f9")

    plt.tight_layout()
    out = "outputs/task2_temporal_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")

    # Periodic signature analysis
    peak_hour = hourly_viol.idxmax()
    print(f"  Peak violation hour: {peak_hour}:00  — Driven by DAILY 24h traffic cycle")


if __name__ == "__main__":
    from data import load_dataset
    df = load_dataset()
    run_task2(df)