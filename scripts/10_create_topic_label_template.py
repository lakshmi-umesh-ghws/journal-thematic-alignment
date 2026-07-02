from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOPICS_DIR = PROJECT_ROOT / "results" / "topics"


def main():
    topic_overview_path = TOPICS_DIR / "topic_overview_for_interpretation.csv"
    document_topics_path = TOPICS_DIR / "document_topics.csv"

    topic_overview = pd.read_csv(topic_overview_path)
    document_topics = pd.read_csv(document_topics_path)

    rows = []

    for _, topic_row in topic_overview.iterrows():
        topic_id = topic_row["Topic"]

        examples = (
            document_topics[document_topics["topic"] == topic_id]
            .sort_values("alignment_score", ascending=False)
            .head(3)["title"]
            .tolist()
        )

        rows.append(
            {
                "topic_id": topic_id,
                "count": topic_row.get("Count"),
                "bertopic_name": topic_row.get("Name"),
                "keywords": topic_row.get("Representation"),
                "mean_alignment": topic_row.get("mean_alignment"),
                "example_title_1": examples[0] if len(examples) > 0 else "",
                "example_title_2": examples[1] if len(examples) > 1 else "",
                "example_title_3": examples[2] if len(examples) > 2 else "",
                "manual_topic_label": "",
                "interpretation_notes": "",
            }
        )

    label_df = pd.DataFrame(rows)

    output_path = TOPICS_DIR / "topic_labels_template.csv"
    label_df.to_csv(output_path, index=False, encoding="utf-8")

    print(f"Saved topic label template to: {output_path}")
    print("\nFirst 15 topics for labeling:")
    print(
        label_df[
            [
                "topic_id",
                "count",
                "bertopic_name",
                "mean_alignment",
                "example_title_1",
                "manual_topic_label",
            ]
        ].head(15)
    )


if __name__ == "__main__":
    main()