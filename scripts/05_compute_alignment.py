import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from journal_alignment.alignment import (
    load_scope_texts,
    compute_alignment_scores,
    create_yearly_alignment_summary,
    create_journal_alignment_summary,
)


def main():
    processed_data_path = PROJECT_ROOT / "data" / "processed" / "papers_clean.csv"
    scopes_dir = PROJECT_ROOT / "data" / "scopes"
    results_tables_dir = PROJECT_ROOT / "results" / "tables"

    results_tables_dir.mkdir(parents=True, exist_ok=True)

    print("Loading cleaned papers...")
    papers_df = pd.read_csv(processed_data_path)
    print(f"Loaded papers: {len(papers_df)}")

    print("Loading journal scope texts...")
    scope_texts = load_scope_texts(scopes_dir)
    print(f"Loaded scopes: {list(scope_texts.keys())}")

    aligned_df = compute_alignment_scores(
        papers_df=papers_df,
        scope_texts=scope_texts,
    )

    alignment_output_path = results_tables_dir / "alignment_scores.csv"
    aligned_df.to_csv(alignment_output_path, index=False, encoding="utf-8")
    print(f"Saved alignment scores to: {alignment_output_path}")

    yearly_summary = create_yearly_alignment_summary(aligned_df)
    yearly_summary_path = results_tables_dir / "yearly_alignment_summary.csv"
    yearly_summary.to_csv(yearly_summary_path, index=False, encoding="utf-8")
    print(f"Saved yearly summary to: {yearly_summary_path}")

    journal_summary = create_journal_alignment_summary(aligned_df)
    journal_summary_path = results_tables_dir / "journal_alignment_summary.csv"
    journal_summary.to_csv(journal_summary_path, index=False, encoding="utf-8")
    print(f"Saved journal summary to: {journal_summary_path}")

    print("\nOverall journal alignment summary:")
    print(journal_summary)

    print("\nYearly alignment summary:")
    print(yearly_summary)


if __name__ == "__main__":
    main()
