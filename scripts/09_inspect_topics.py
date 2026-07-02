from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOPICS_DIR = PROJECT_ROOT / "results" / "topics"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"


def main():
    topic_info_path = TOPICS_DIR / "topic_info.csv"
    topic_alignment_path = TOPICS_DIR / "topic_alignment_summary.csv"
    document_topics_path = TOPICS_DIR / "document_topics.csv"

    topic_info = pd.read_csv(topic_info_path)
    topic_alignment = pd.read_csv(topic_alignment_path)
    document_topics = pd.read_csv(document_topics_path)

    print("\nTopic info columns:")
    print(topic_info.columns.tolist())

    print("\nTop 20 BERTopic topics:")
    print(topic_info.head(20))

    # Merge topic size and alignment information
    topic_overview = topic_info.merge(
        topic_alignment,
        left_on="Topic",
        right_on="topic",
        how="left",
    )

    # Remove outlier topic -1 if present
    topic_overview = topic_overview[topic_overview["Topic"] != -1].copy()

    # Keep most useful columns
    useful_columns = []

    for col in [
        "Topic",
        "Count",
        "Name",
        "Representation",
        "n_papers",
        "mean_alignment",
        "median_alignment",
    ]:
        if col in topic_overview.columns:
            useful_columns.append(col)

    topic_overview = topic_overview[useful_columns]

    topic_overview_path = TOPICS_DIR / "topic_overview_for_interpretation.csv"
    topic_overview.to_csv(topic_overview_path, index=False, encoding="utf-8")

    print(f"\nSaved topic overview to: {topic_overview_path}")

    print("\nMost frequent topics:")
    print(topic_overview.sort_values("Count", ascending=False).head(15))

    print("\nMost aligned topics:")
    print(topic_overview.sort_values("mean_alignment", ascending=False).head(10))

    print("\nLeast aligned topics:")
    print(topic_overview.sort_values("mean_alignment", ascending=True).head(10))

    # Show example papers from top topics
    print("\nExample papers from main topics:")

    main_topics = topic_overview.sort_values("Count", ascending=False)["Topic"].head(5)

    for topic_id in main_topics:
        print("\n" + "=" * 80)
        print(f"Topic {topic_id}")
        print("=" * 80)

        topic_docs = document_topics[document_topics["topic"] == topic_id].head(5)

        for _, row in topic_docs.iterrows():
            print(f"- [{row['journal_short_name'].upper()} | {row['year']}] {row['title']}")


if __name__ == "__main__":
    main()