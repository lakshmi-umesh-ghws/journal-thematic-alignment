import re
from pathlib import Path
from typing import Dict, List

import pandas as pd


def normalize_text(text):

    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def count_words(text):

    if pd.isna(text):
        return 0

    return len(str(text).split())


def load_raw_files(raw_data_dir: Path) -> pd.DataFrame:

    raw_files = list(raw_data_dir.glob("*_raw.csv"))

    if not raw_files:
        raise FileNotFoundError("No raw CSV files found in data/raw.")

    dataframes = []

    for file_path in raw_files:
        print(f"Loading {file_path.name}")
        df = pd.read_csv(file_path)
        dataframes.append(df)

    combined_df = pd.concat(dataframes, ignore_index=True)
    print(f"Loaded total rows: {len(combined_df)}")

    return combined_df


def filter_by_journal_match(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    df["venue_norm"] = df["venue"].apply(normalize_text)
    df["journal_api_norm"] = df["journal_name_from_api"].apply(normalize_text)
    df["journal_query_norm"] = df["journal_query"].apply(normalize_text)

    def is_match(row):
        query = row["journal_query_norm"]
        venue = row["venue_norm"]
        journal_api = row["journal_api_norm"]

        return query in venue or query in journal_api

    before = len(df)
    df = df[df.apply(is_match, axis=1)].copy()
    after = len(df)

    print(f"Journal filtering: kept {after} of {before} rows")

    return df


def clean_papers(df: pd.DataFrame, start_year: int, end_year: int) -> pd.DataFrame:


    df = df.copy()

    print("\nInitial rows:", len(df))

    # Keep only required years
    df = df[(df["year"] >= start_year) & (df["year"] <= end_year)]
    print("After year filtering:", len(df))


    df = df.dropna(subset=["title", "abstract"])
    print("After removing missing title/abstract:", len(df))


    df = df.drop_duplicates(subset=["paper_id"])
    print("After removing duplicate paper IDs:", len(df))

    # Add abstract word count
    df["abstract_word_count"] = df["abstract"].apply(count_words)

    df = df[df["abstract_word_count"] >= 40]
    print("After removing short abstracts:", len(df))

    # Clean text fields
    df["title_clean"] = df["title"].apply(normalize_text)
    df["abstract_clean"] = df["abstract"].apply(normalize_text)


    columns_to_keep = [
        "paper_id",
        "journal_query",
        "journal_short_name",
        "title",
        "title_clean",
        "abstract",
        "abstract_clean",
        "abstract_word_count",
        "year",
        "publication_date",
        "venue",
        "journal_name_from_api",
        "citation_count",
        "doi",
        "url",
        "fields_of_study",
    ]

    available_columns = [col for col in columns_to_keep if col in df.columns]
    df = df[available_columns]

    return df


def create_dataset_summary(df: pd.DataFrame) -> pd.DataFrame:


    summary = (
        df.groupby(["journal_short_name", "year"])
        .size()
        .reset_index(name="n_papers")
        .sort_values(["journal_short_name", "year"])
    )

    return summary


def save_processed_data(df: pd.DataFrame, output_path: Path) -> None:

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"\nSaved cleaned data to: {output_path}")
    print(f"Final rows: {len(df)}")