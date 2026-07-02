import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def inspect_file(file_path: Path):
    print("\n" + "=" * 80)
    print(f"Inspecting file: {file_path.name}")
    print("=" * 80)

    df = pd.read_csv(file_path)

    print(f"Total rows: {len(df)}")
    print(f"Total columns: {len(df.columns)}")

    if "paper_id" in df.columns:
        print(f"Unique paper IDs: {df['paper_id'].nunique()}")

    if "abstract" in df.columns:
        missing_abstracts = df["abstract"].isna().sum()
        print(f"Missing abstracts: {missing_abstracts}")
        print(f"Rows with abstracts: {len(df) - missing_abstracts}")

    if "year" in df.columns:
        print("\nRows by year:")
        print(df["year"].value_counts().sort_index())

    if "venue" in df.columns:
        print("\nTop venues:")
        print(df["venue"].value_counts().head(15))

    print("\nSample rows:")
    sample_columns = [
        "journal_query",
        "title",
        "year",
        "venue",
        "journal_name_from_api",
        "citation_count",
    ]

    available_columns = [col for col in sample_columns if col in df.columns]
    print(df[available_columns].head(5))


def main():
    csv_files = list(RAW_DATA_DIR.glob("*_raw.csv"))

    if not csv_files:
        print("No raw CSV files found in data/raw.")
        return

    for file_path in csv_files:
        inspect_file(file_path)


if __name__ == "__main__":
    main()
