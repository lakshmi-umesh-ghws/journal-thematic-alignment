import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from journal_alignment.data_collection import (
    SemanticScholarCollector,
    load_config,
    save_raw_data,
)


def main():
    config = load_config(PROJECT_ROOT / "config.yaml")

    start_year = config["project"]["start_year"]
    end_year = config["project"]["end_year"]
    fields = config["semantic_scholar"]["fields"]
    api_key = config["semantic_scholar"]["api_key"]

    collector = SemanticScholarCollector(api_key=api_key)

    all_dataframes = []

    for journal in config["journals"]:
        journal_name = journal["name"]
        short_name = journal["short_name"]

        print("\n" + "=" * 80)
        print(f"Starting collection for: {journal_name}")
        print("=" * 80)

        df = collector.collect_journal(
            journal_name=journal_name,
            short_name=short_name,
            start_year=start_year,
            end_year=end_year,
            fields=fields,
        )

        raw_output_path = PROJECT_ROOT / "data" / "raw" / f"{short_name}_raw.csv"
        save_raw_data(df, raw_output_path)

        all_dataframes.append(df)

    combined_df = pd.concat(all_dataframes, ignore_index=True)

    combined_output_path = PROJECT_ROOT / "data" / "raw" / "all_journals_raw.csv"
    save_raw_data(combined_df, combined_output_path)

    print("\nData collection finished.")
    print(f"Total collected rows: {len(combined_df)}")


if __name__ == "__main__":
    main()