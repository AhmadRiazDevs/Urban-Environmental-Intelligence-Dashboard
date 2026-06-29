"""
Main Pipeline — Urban Environmental Intelligence Challenge
Runs all 4 tasks in sequence.
Usage: python main.py
"""

import pandas as pd
from pathlib import Path

from data import build_dataset, load_dataset, STATION_META
from Task1_pca import run_task1
from Task2_temporal import run_task2
from Task3_distribution import run_task3
from Task4_visual_audit import run_task4


def main():
    print("=" * 60)
    print("  Urban Environmental Intelligence Pipeline")
    print("=" * 60)

    # --- Generate or load dataset
    data_path = "data/air_quality_2025.pkl"
    if not Path(data_path).exists():
        df = build_dataset(data_path)
    else:
        print(f"\nDataset found — loading from {data_path}")
        df = load_dataset(data_path)
        print(f"Loaded shape: {df.shape}")

    # --- Run all tasks
    run_task1(df)
    run_task2(df)
    run_task3(df)
    run_task4(df, STATION_META)

    print("\n" + "=" * 60)
    print("  All tasks complete. Outputs saved in /outputs/")
    print("=" * 60)


if __name__ == "__main__":
    main()