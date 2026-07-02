import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from journal_alignment.visualization import (
    plot_alignment_distribution,
    plot_yearly_alignment_trend,
    plot_alignment_boxplot,
    save_outlier_tables,
)


def main():
    results_tables_dir = PROJECT_ROOT / "results" / "tables"
    results_figures_dir = PROJECT_ROOT / "results" / "figures"

    alignment_path = results_tables_dir / "alignment_scores.csv"
    yearly_path = results_tables_dir / "yearly_alignment_summary.csv"

    print("Loading alignment results...")
    alignment_df = pd.read_csv(alignment_path)
    yearly_df = pd.read_csv(yearly_path)

    print(f"Alignment rows: {len(alignment_df)}")
    print(f"Yearly summary rows: {len(yearly_df)}")

    plot_alignment_distribution(
        alignment_df,
        results_figures_dir / "alignment_distribution_by_journal.png",
    )

    plot_yearly_alignment_trend(
        yearly_df,
        results_figures_dir / "yearly_alignment_trend.png",
    )

    plot_alignment_boxplot(
        alignment_df,
        results_figures_dir / "alignment_boxplot_by_journal.png",
    )

    save_outlier_tables(
        alignment_df,
        results_tables_dir,
        n=20,
    )

    print("\nVisualization step completed.")


if __name__ == "__main__":
    main()