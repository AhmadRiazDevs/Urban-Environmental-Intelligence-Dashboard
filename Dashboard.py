"""
Streamlit Interactive Dashboard — Urban Environmental Intelligence
Run: streamlit run dashboard.py
"""

import sys
import os
from pathlib import Path

# --- Ensure the project folder is on the path so modules can be imported
PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

# --- Change working directory so relative data paths work correctly
os.chdir(PROJECT_DIR)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import stats

st.set_page_config(page_title="Urban Air Quality Intelligence", layout="wide")


@st.cache_data
def load_data():
    from data import build_dataset, load_dataset, STATION_META
    p = str(PROJECT_DIR / "data" / "air_quality_2025.pkl")
    if not Path(p).exists():
        with st.spinner("Generating dataset for the first time (~30 seconds)..."):
            df = build_dataset(p)
    else:
        df = load_dataset(p)
    return df, STATION_META


df, meta = load_data()

st.title("🌆 Urban Environmental Intelligence Dashboard")
st.markdown("**Data Science Assignment 2** — 100 Global Air Quality Sensors, Year 2025")

tab1, tab2, tab3, tab4 = st.tabs(["Task 1: PCA", "Task 2: Temporal", "Task 3: Distribution", "Task 4: Visual Audit"])

# ===== TASK 1 =====
with tab1:
    st.header("Task 1: PCA Dimensionality Reduction")
    features = ["PM25", "PM10", "NO2", "Ozone", "Temperature", "Humidity"]
    station_agg = df.groupby(["station_id", "zone"])[features].mean().reset_index()
    X = station_agg[features].values
    zones = station_agg["zone"].values
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    explained = pca.explained_variance_ratio_ * 100

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6, 5))
        for zone, color in {"Industrial": "#d62728", "Residential": "#1f77b4"}.items():
            mask = zones == zone
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=color, label=zone, alpha=0.85, s=70, edgecolors="white")
        ax.set_xlabel(f"PC1 ({explained[0]:.1f}%)")
        ax.set_ylabel(f"PC2 ({explained[1]:.1f}%)")
        ax.set_title("PCA Cluster Plot")
        ax.legend()
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig)
    with col2:
        loadings = pd.DataFrame(pca.components_.T, index=features, columns=["PC1", "PC2"])
        st.markdown("### PCA Loadings")
        st.dataframe(loadings.style.background_gradient(cmap="RdBu_r", axis=None))
        st.markdown(f"""
**Explained Variance:** PC1 = {explained[0]:.1f}% | PC2 = {explained[1]:.1f}%

**Justification:** PCA is linear and interpretable via loadings. PC1 is dominated by
pollution variables (PM2.5, PM10, NO2) and separates Industrial from Residential zones.
PC2 is driven by weather (Temperature, Humidity).
        """)

# ===== TASK 2 =====
with tab2:
    st.header("Task 2: High-Density Temporal Heatmap")
    df_t = df.copy()
    df_t["date"] = pd.to_datetime(df_t["timestamp"]).dt.date
    df_t["hour"] = pd.to_datetime(df_t["timestamp"]).dt.hour
    daily = df_t.groupby(["station_id", "date"])["PM25"].max().reset_index()
    pivot = daily.pivot(index="station_id", columns="date", values="PM25").fillna(0)
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(16, 6))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd",
                   norm=mcolors.Normalize(0, 150), interpolation="nearest")
    dates = list(pivot.columns)
    month_ticks = [i for i, d in enumerate(dates) if pd.Timestamp(d).day == 1]
    ax.set_xticks(month_ticks)
    ax.set_xticklabels([pd.Timestamp(dates[i]).strftime("%b") for i in month_ticks])
    ax.set_yticks(range(0, 100, 10))
    ax.set_yticklabels([pivot.index[i] for i in range(0, 100, 10)], fontsize=7)
    ax.set_title("PM2.5 Daily Max — 100 Stations (sorted by mean)")
    plt.colorbar(im, ax=ax, label="PM2.5 µg/m³")
    ax.spines[["top", "right", "bottom", "left"]].set_visible(False)
    st.pyplot(fig)

    hourly_viol = df_t[df_t["PM25"] > 35].groupby("hour").size() / df_t.groupby("hour").size()
    fig2, ax2 = plt.subplots(figsize=(10, 3))
    ax2.bar(hourly_viol.index, hourly_viol.values, color="#d62728", alpha=0.8)
    ax2.set_xlabel("Hour of Day")
    ax2.set_ylabel("Violation Rate")
    ax2.set_title("Hourly PM2.5 Violation Rate — Diurnal Signature")
    ax2.set_xticks(range(0, 24))
    ax2.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig2)
    st.info(f"**Peak violation hour: {hourly_viol.idxmax()}:00** — Driven by **daily 24-hour traffic cycle**.")

# ===== TASK 3 =====
with tab3:
    st.header("Task 3: Distribution Modeling & Tail Integrity")
    industrial = df[df["zone"] == "Industrial"]["PM25"].values
    industrial = industrial[industrial > 0]
    p99 = np.percentile(industrial, 99)
    extreme_prob = np.mean(industrial > 200) * 100

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6, 5))
        kde = stats.gaussian_kde(industrial, bw_method="scott")
        x_r = np.linspace(0, 300, 500)
        ax.fill_between(x_r, kde(x_r), alpha=0.3, color="#1f77b4")
        ax.plot(x_r, kde(x_r), color="#1f77b4", lw=2)
        ax.axvline(35, color="#d62728", lw=1.5, linestyle="--", label="Threshold (35)")
        ax.axvline(p99, color="#ff7f0e", lw=1.5, linestyle="--", label=f"99th pct ({p99:.0f})")
        ax.set_title("KDE — Peak Detection")
        ax.set_xlabel("PM2.5 (µg/m³)")
        ax.set_ylabel("Density")
        ax.legend()
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig)
        st.markdown("**KDE** reveals where most readings cluster — ideal for peak pollution modes.")
    with col2:
        fig, ax = plt.subplots(figsize=(6, 5))
        bins = np.logspace(np.log10(1), np.log10(float(industrial.max()) + 1), 80)
        ax.hist(industrial, bins=bins, color="#2ca02c", alpha=0.75, edgecolor="white", lw=0.4)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.axvline(200, color="#d62728", lw=2, linestyle="--", label="Extreme Hazard (200)")
        ax.axvline(p99, color="#ff7f0e", lw=1.5, linestyle="--", label=f"99th pct ({p99:.0f})")
        ax.set_title("Log-Log Histogram — Tail Detection")
        ax.set_xlabel("PM2.5 (µg/m³) — Log Scale")
        ax.set_ylabel("Count — Log Scale")
        ax.legend()
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig)
        st.markdown("**Log-log histogram** is more *honest* for rare events — tails stay visible without bin-size distortion.")

    st.metric("99th Percentile PM2.5", f"{p99:.1f} µg/m³")
    st.metric("P(PM2.5 > 200 µg/m³)", f"{extreme_prob:.4f}%")

# ===== TASK 4 =====
with tab4:
    st.header("Task 4: Visual Integrity Audit")
   # st.error("❌ **3D Bar Chart REJECTED** — Perspective distortion → Lie Factor > 1. Depth/shadows add non-data ink → low Data-Ink Ratio.")
   # st.success("✅ **Small Multiples adopted** — One scatter panel per region. Sequential color scale.")

    station_summary = df.groupby("station_id")["PM25"].mean().reset_index()
    station_summary.columns = ["station_id", "mean_PM25"]
    merged = station_summary.merge(meta, on="station_id")
    regions = sorted(merged["region"].unique())
    cmap = plt.get_cmap("YlOrRd")
    norm = mcolors.Normalize(vmin=merged["mean_PM25"].min(), vmax=merged["mean_PM25"].max())

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    for i, region in enumerate(regions):
        ax = axes[i]
        subset = merged[merged["region"] == region]
        for zone, marker in [("Industrial", "^"), ("Residential", "o")]:
            z = subset[subset["zone"] == zone]
            ax.scatter(z["population_density"], z["mean_PM25"],
                       marker=marker, c=z["mean_PM25"], cmap=cmap, norm=norm,
                       s=80, alpha=0.9, edgecolors="#333", lw=0.5, label=zone)
        ax.set_title(f"Region: {region}", fontweight="bold")
        ax.set_xlabel("Pop. Density (people/km²)")
        ax.set_ylabel("Mean PM2.5 (µg/m³)")
        ax.axhline(35, color="#d62728", lw=1, linestyle="--", alpha=0.5)
        ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_facecolor("#f9f9f9")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.02, label="Mean PM2.5")
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("""
**Color Scale Justification:**
- ✅ **Sequential (YlOrRd)** — luminance increases monotonically; human vision reads darker = higher pollution correctly.
- ❌ **Rainbow** — creates false perceptual edges at hue boundaries, misleading viewers about real data transitions.
    """)