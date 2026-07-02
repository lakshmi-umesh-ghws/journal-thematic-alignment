import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from journal_alignment.preprocessing import (
    load_raw_files,
    filter_by_journal_match,
    clean_papers,
    create_dataset_summary,
    save_processed_data,
)


def load_config(config_path: Path):
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def main():
    config = load_config(PROJECT_ROOT / "config.yaml")

    start_year = config["project"]["start_year"]
    end_year = config["project"]["end_year"]

    raw_data_dir = PROJECT_ROOT / "data" / "raw"
    processed_data_dir = PROJECT_ROOT / "data" / "processed"
    results_tables_dir = PROJECT_ROOT / "results" / "tables"

    raw_df = load_raw_files(raw_data_dir)

    matched_df = filter_by_journal_match(raw_df)

    clean_df = clean_papers(
        matched_df,
        start_year=start_year,
        end_year=end_year,
    )

    save_processed_data(
        clean_df,
        processed_data_dir / "papers_clean.csv",
    )

    summary_df = create_dataset_summary(clean_df)
    results_tables_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(
        results_tables_dir / "dataset_summary_by_year.csv",
        index=False,
        encoding="utf-8",
    )

    print("\nDataset summary by journal and year:")
    print(summary_df)

    print("\nPapers per journal:")
    print(clean_df["journal_short_name"].value_counts())


if __name__ == "__main__":
    main()