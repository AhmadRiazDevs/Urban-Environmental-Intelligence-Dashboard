"""
Task 3: Distribution Modeling & Tail Integrity
Two plots for industrial zone PM2.5:
  1. KDE (reveals peaks)
  2. ECDF / Log-scale histogram (reveals tails)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

Path("outputs").mkdir(exist_ok=True)


def run_task3(df: pd.DataFrame):
    print("\n[Task 3] Distribution Modeling & Tail Integrity ...")

    # Select industrial zone stations
    industrial = df[df["zone"] == "Industrial"]["PM25"].values
    industrial = industrial[industrial > 0]

    # 99th percentile
    p99 = np.percentile(industrial, 99)
    extreme_hazard_prob = np.mean(industrial > 200) * 100
    print(f"  99th Percentile PM2.5: {p99:.1f} µg/m³")
    print(f"  P(PM2.5 > 200 µg/m³): {extreme_hazard_prob:.4f}%")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("white")

    # ---- Plot A: KDE — reveals peaks ----
    ax1 = axes[0]
    kde = stats.gaussian_kde(industrial, bw_method="scott")
    x_range = np.linspace(0, 300, 500)
    kde_vals = kde(x_range)

    ax1.fill_between(x_range, kde_vals, alpha=0.3, color="#1f77b4")
    ax1.plot(x_range, kde_vals, color="#1f77b4", lw=2)
    ax1.axvline(35, color="#d62728", lw=1.5, linestyle="--", label="Health Threshold (35)")
    ax1.axvline(p99, color="#ff7f0e", lw=1.5, linestyle="--", label=f"99th pct ({p99:.0f} µg/m³)")
    ax1.set_xlabel("PM2.5 (µg/m³)", fontsize=11)
    ax1.set_ylabel("Density", fontsize=11)
    ax1.set_title("KDE — Reveals Distribution Peak\n(Industrial Zone)", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.set_facecolor("#f9f9f9")
    ax1.set_xlim(0, 300)

    # ---- Plot B: Log-scale histogram — reveals tails ----
    ax2 = axes[1]
    bins = np.logspace(np.log10(1), np.log10(industrial.max() + 1), 80)
    counts, edges, _ = ax2.hist(industrial, bins=bins, color="#2ca02c", alpha=0.75,
                                edgecolor="white", linewidth=0.4)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.axvline(200, color="#d62728", lw=2, linestyle="--", label="Extreme Hazard (200)")
    ax2.axvline(p99, color="#ff7f0e", lw=1.5, linestyle="--", label=f"99th pct ({p99:.0f})")
    ax2.set_xlabel("PM2.5 (µg/m³) — Log Scale", fontsize=11)
    ax2.set_ylabel("Count — Log Scale", fontsize=11)
    ax2.set_title("Log-Log Histogram — Reveals Tail Behaviour\n(Industrial Zone)", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.set_facecolor("#f9f9f9")

    # Annotate extreme hazard zone
    ax2.axvspan(200, industrial.max(), alpha=0.08, color="#d62728", label="Extreme Hazard Zone")

    plt.tight_layout()
    out = "outputs/task3_distribution.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")

    return p99, extreme_hazard_prob


if __name__ == "__main__":
    from data import load_dataset
    df = load_dataset()
    run_task3(df)