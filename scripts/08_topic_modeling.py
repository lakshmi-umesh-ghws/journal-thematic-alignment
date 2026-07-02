import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from journal_alignment.topic_modeling import (
    fit_bertopic_model,
    create_topic_info,
    create_topic_alignment_summary,
    create_topic_year_summary,
    save_bertopic_model,
)


def main():
    results_tables_dir = PROJECT_ROOT / "results" / "tables"
    results_topics_dir = PROJECT_ROOT / "results" / "topics"

    alignment_path = results_tables_dir / "alignment_scores.csv"

    print("Loading alignment scores...")
    df = pd.read_csv(alignment_path)
    print(f"Loaded rows: {len(df)}")

    topic_model, df_with_topics = fit_bertopic_model(
        df,
        text_column="abstract",
        min_topic_size=50,
    )

    results_topics_dir.mkdir(parents=True, exist_ok=True)

    document_topics_path = results_topics_dir / "document_topics.csv"
    df_with_topics.to_csv(document_topics_path, index=False, encoding="utf-8")
    print(f"Saved document topics to: {document_topics_path}")

    topic_info = create_topic_info(topic_model)
    topic_info_path = results_topics_dir / "topic_info.csv"
    topic_info.to_csv(topic_info_path, index=False, encoding="utf-8")
    print(f"Saved topic info to: {topic_info_path}")

    topic_alignment = create_topic_alignment_summary(df_with_topics)
    topic_alignment_path = results_topics_dir / "topic_alignment_summary.csv"
    topic_alignment.to_csv(topic_alignment_path, index=False, encoding="utf-8")
    print(f"Saved topic alignment summary to: {topic_alignment_path}")

    topic_year_summary = create_topic_year_summary(df_with_topics)
    topic_year_path = results_topics_dir / "topic_year_summary.csv"
    topic_year_summary.to_csv(topic_year_path, index=False, encoding="utf-8")
    print(f"Saved topic year summary to: {topic_year_path}")

    save_bertopic_model(topic_model, results_topics_dir)

    print("\nTop 15 topics:")
    print(topic_info.head(15))

    print("\nMost aligned topics:")
    print(topic_alignment.head(10))

    print("\nLeast aligned topics:")
    print(topic_alignment.tail(10))

    print("\nBERTopic analysis completed.")


if __name__ == "__main__":
    main()