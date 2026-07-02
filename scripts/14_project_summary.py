from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
TOPICS_DIR = RESULTS_DIR / "topics"
REPORT_DIR = PROJECT_ROOT / "report"


def read_csv_if_exists(path: Path):

    if path.exists():
        return pd.read_csv(path)

    print(f"Warning: file not found: {path}")
    return None


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    dataset_summary = read_csv_if_exists(TABLES_DIR / "dataset_summary_by_year.csv")
    journal_alignment = read_csv_if_exists(TABLES_DIR / "journal_alignment_summary.csv")
    yearly_alignment = read_csv_if_exists(TABLES_DIR / "yearly_alignment_summary.csv")
    trend_stats = read_csv_if_exists(TABLES_DIR / "trend_statistics.csv")
    journal_comparison = read_csv_if_exists(TABLES_DIR / "journal_comparison_statistics.csv")
    classifier_report = read_csv_if_exists(TABLES_DIR / "classifier_report.csv")
    topic_table = read_csv_if_exists(TABLES_DIR / "final_topic_interpretation_table.csv")
    outliers = read_csv_if_exists(TABLES_DIR / "outlier_review_for_report.csv")

    output_path = REPORT_DIR / "project_summary.md"

    with open(output_path, "w", encoding="utf-8") as file:
        file.write("# Project Summary\n\n")

        file.write("## 1. Dataset Summary\n\n")
        if dataset_summary is not None:
            total_papers = dataset_summary["n_papers"].sum()
            file.write(f"Total cleaned papers: **{total_papers}**\n\n")

            file.write("Papers per journal:\n\n")
            papers_per_journal = (
                dataset_summary
                .groupby("journal_short_name")["n_papers"]
                .sum()
                .reset_index()
            )
            file.write(papers_per_journal.to_markdown(index=False))
            file.write("\n\n")

            file.write("Papers by year and journal:\n\n")
            file.write(dataset_summary.to_markdown(index=False))
            file.write("\n\n")

        file.write("## 2. Overall Alignment Summary\n\n")
        if journal_alignment is not None:
            file.write(journal_alignment.to_markdown(index=False))
            file.write("\n\n")

        file.write("## 3. Yearly Alignment Summary\n\n")
        if yearly_alignment is not None:
            file.write(yearly_alignment.to_markdown(index=False))
            file.write("\n\n")

        file.write("## 4. Trend Statistics\n\n")
        if trend_stats is not None:
            file.write(trend_stats.to_markdown(index=False))
            file.write("\n\n")

        file.write("## 5. Journal Comparison Statistics\n\n")
        if journal_comparison is not None:
            file.write(journal_comparison.to_markdown(index=False))
            file.write("\n\n")

        file.write("## 6. Classifier Baseline\n\n")
        if classifier_report is not None:
            file.write("TF-IDF + Logistic Regression classification report:\n\n")
            file.write(classifier_report.to_markdown(index=False))
            file.write("\n\n")

        file.write("## 7. Topic Interpretation Table\n\n")
        if topic_table is not None:
            file.write(topic_table.to_markdown(index=False))
            file.write("\n\n")

        file.write("## 8. Outlier Review\n\n")
        if outliers is not None:
            file.write(outliers.to_markdown(index=False))
            file.write("\n\n")

        file.write("## 9. Preliminary Interpretation\n\n")
        file.write(
            "- The cleaned dataset contains abstracts from Expert Systems with Applications "
            "and Knowledge-Based Systems between 2015 and 2025.\n"
        )
        file.write(
            "- Sentence-BERT alignment scores were used to measure semantic similarity "
            "between each article abstract and the corresponding journal Aims & Scope.\n"
        )
        file.write(
            "- The yearly trend analysis suggests a weak but statistically significant "
            "decrease in thematic alignment over time.\n"
        )
        file.write(
            "- BERTopic identified major thematic areas and helped explain which topics "
            "are more or less aligned with the journal scope.\n"
        )
        file.write(
            "- The TF-IDF + Logistic Regression classifier provides additional evidence "
            "that the two journals are thematically distinguishable, while still overlapping.\n"
        )
        file.write(
            "- Outlier inspection shows that low-alignment papers are often specialized "
            "technical AI papers rather than necessarily irrelevant articles.\n"
        )

    print(f"Saved project summary to: {output_path}")


if __name__ == "__main__":
    main()