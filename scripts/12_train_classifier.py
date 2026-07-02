import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from journal_alignment.classification import (
    train_journal_classifier,
    save_classification_report,
    save_confusion_matrix,
    extract_top_terms,
)


def main():
    results_tables_dir = PROJECT_ROOT / "results" / "tables"
    results_figures_dir = PROJECT_ROOT / "results" / "figures"

    data_path = results_tables_dir / "alignment_scores.csv"

    print("Loading alignment dataset...")
    df = pd.read_csv(data_path)
    print(f"Loaded rows: {len(df)}")

    results = train_journal_classifier(df)

    print("\nClassifier accuracy:")
    print(results["accuracy"])

    report_path = results_tables_dir / "classifier_report.csv"
    confusion_matrix_csv_path = results_tables_dir / "classifier_confusion_matrix.csv"
    confusion_matrix_fig_path = results_figures_dir / "classifier_confusion_matrix.png"
    top_terms_path = results_tables_dir / "classifier_top_terms.csv"

    save_classification_report(
        results["classification_report"],
        report_path,
    )

    save_confusion_matrix(
        results["confusion_matrix"],
        results["labels"],
        confusion_matrix_csv_path,
        confusion_matrix_fig_path,
    )

    extract_top_terms(
        results["classifier"],
        top_terms_path,
        top_n=30,
    )

    print("\nClassification baseline completed.")


if __name__ == "__main__":
    main()