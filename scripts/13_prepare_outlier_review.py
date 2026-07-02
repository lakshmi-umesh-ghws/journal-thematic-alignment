from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = PROJECT_ROOT / "results" / "tables"


def shorten_text(text, max_words=60):

    if pd.isna(text):
        return ""

    words = str(text).split()
    if len(words) <= max_words:
        return str(text)

    return " ".join(words[:max_words]) + "..."


def select_outliers_by_journal(df, n=5):


    rows = []

    for journal in sorted(df["journal_short_name"].unique()):
        subset = df[df["journal_short_name"] == journal].copy()

        highest = subset.sort_values("alignment_score", ascending=False).head(n)
        lowest = subset.sort_values("alignment_score", ascending=True).head(n)

        for _, row in highest.iterrows():
            rows.append(
                {
                    "journal": journal,
                    "outlier_type": "high_alignment",
                    "year": row["year"],
                    "title": row["title"],
                    "alignment_score": row["alignment_score"],
                    "abstract_preview": shorten_text(row["abstract"]),
                    "manual_interpretation": "",
                }
            )

        for _, row in lowest.iterrows():
            rows.append(
                {
                    "journal": journal,
                    "outlier_type": "low_alignment",
                    "year": row["year"],
                    "title": row["title"],
                    "alignment_score": row["alignment_score"],
                    "abstract_preview": shorten_text(row["abstract"]),
                    "manual_interpretation": "",
                }
            )

    return pd.DataFrame(rows)


def main():
    alignment_path = TABLES_DIR / "alignment_scores.csv"

    print("Loading alignment scores...")
    df = pd.read_csv(alignment_path)

    outlier_review = select_outliers_by_journal(df, n=5)

    output_path = TABLES_DIR / "outlier_review_for_report.csv"
    outlier_review.to_csv(output_path, index=False, encoding="utf-8")

    print(f"Saved outlier review table to: {output_path}")

    print("\nOutlier review preview:")
    print(
        outlier_review[
            [
                "journal",
                "outlier_type",
                "year",
                "title",
                "alignment_score",
            ]
        ]
    )


if __name__ == "__main__":
    main()