import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from journal_alignment.statistics_analysis import (
    compute_trend_statistics,
    compare_journals,
)


def main():
    results_tables_dir = PROJECT_ROOT / "results" / "tables"
    alignment_path = results_tables_dir / "alignment_scores.csv"

    print("Loading alignment scores...")
    df = pd.read_csv(alignment_path)
    print(f"Loaded rows: {len(df)}")

    trend_stats = compute_trend_statistics(df)
    trend_output_path = results_tables_dir / "trend_statistics.csv"
    trend_stats.to_csv(trend_output_path, index=False, encoding="utf-8")

    journal_comparison = compare_journals(df)
    comparison_output_path = results_tables_dir / "journal_comparison_statistics.csv"
    journal_comparison.to_csv(comparison_output_path, index=False, encoding="utf-8")

    print("\nTrend statistics:")
    print(trend_stats)

    print("\nJournal comparison:")
    print(journal_comparison)

    print("\nStatistical analysis completed.")


if __name__ == "__main__":
    main()