from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_alignment_distribution(df: pd.DataFrame, output_path: Path) -> None:


    journals = df["journal_short_name"].unique()

    plt.figure(figsize=(10, 6))

    for journal in journals:
        subset = df[df["journal_short_name"] == journal]
        plt.hist(
            subset["alignment_score"],
            bins=30,
            alpha=0.5,
            label=journal.upper(),
        )

    plt.xlabel("Alignment score")
    plt.ylabel("Number of papers")
    plt.title("Distribution of thematic alignment scores by journal")
    plt.legend()
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved figure: {output_path}")


def plot_yearly_alignment_trend(yearly_df: pd.DataFrame, output_path: Path) -> None:


    journals = yearly_df["journal_short_name"].unique()

    plt.figure(figsize=(10, 6))

    for journal in journals:
        subset = yearly_df[yearly_df["journal_short_name"] == journal]
        plt.plot(
            subset["year"],
            subset["mean_alignment"],
            marker="o",
            label=journal.upper(),
        )

    plt.xlabel("Publication year")
    plt.ylabel("Mean alignment score")
    plt.title("Yearly thematic alignment trend")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved figure: {output_path}")


def plot_alignment_boxplot(df: pd.DataFrame, output_path: Path) -> None:


    journals = sorted(df["journal_short_name"].unique())
    data = [
        df[df["journal_short_name"] == journal]["alignment_score"].dropna()
        for journal in journals
    ]

    plt.figure(figsize=(8, 6))
    plt.boxplot(data, tick_labels=[journal.upper() for journal in journals])

    plt.xlabel("Journal")
    plt.ylabel("Alignment score")
    plt.title("Alignment score variation by journal")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved figure: {output_path}")


def save_outlier_tables(df: pd.DataFrame, output_dir: Path, n: int = 20) -> None:

    #Save highest-alignment and lowest-alignment papers for qualitative inspection.


    output_dir.mkdir(parents=True, exist_ok=True)

    columns = [
        "journal_short_name",
        "year",
        "title",
        "alignment_score",
        "abstract",
        "doi",
        "url",
    ]

    available_columns = [col for col in columns if col in df.columns]

    highest = (
        df.sort_values("alignment_score", ascending=False)
        .head(n)[available_columns]
    )

    lowest = (
        df.sort_values("alignment_score", ascending=True)
        .head(n)[available_columns]
    )

    highest_path = output_dir / "highest_alignment_papers.csv"
    lowest_path = output_dir / "lowest_alignment_papers.csv"

    highest.to_csv(highest_path, index=False, encoding="utf-8")
    lowest.to_csv(lowest_path, index=False, encoding="utf-8")

    print(f"Saved table: {highest_path}")
    print(f"Saved table: {lowest_path}")