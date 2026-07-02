from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOPICS_DIR = PROJECT_ROOT / "results" / "topics"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"


def get_topic_label(row):

    manual_label = row.get("manual_topic_label", "")

    if isinstance(manual_label, str) and manual_label.strip():
        return manual_label.strip()

    return str(row.get("bertopic_name", row.get("Name", row.get("topic_id"))))


def plot_topic_counts_by_journal(document_topics, topic_labels):


    merged = document_topics.merge(
        topic_labels[["topic_id", "topic_label"]],
        left_on="topic",
        right_on="topic_id",
        how="left",
    )

    merged["topic_label"] = merged["topic_label"].fillna("Unknown topic")

    topic_counts = (
        merged[merged["topic"] != -1]
        .groupby(["topic_label", "journal_short_name"])
        .size()
        .reset_index(name="n_papers")
    )

    pivot = topic_counts.pivot(
        index="topic_label",
        columns="journal_short_name",
        values="n_papers",
    ).fillna(0)

    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=True).index]

    fig, ax = plt.subplots(figsize=(12, 8))
    pivot.plot(kind="barh", ax=ax)

    ax.set_xlabel("Number of papers")
    ax.set_ylabel("Topic")
    ax.set_title("Topic distribution by journal")

    plt.tight_layout()

    output_path = FIGURES_DIR / "topic_distribution_by_journal.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved figure: {output_path}")


def plot_topic_alignment(topic_alignment, topic_labels):


    merged = topic_alignment.merge(
        topic_labels[["topic_id", "topic_label"]],
        left_on="topic",
        right_on="topic_id",
        how="left",
    )

    merged["topic_label"] = merged["topic_label"].fillna("Unknown topic")
    merged = merged.sort_values("mean_alignment", ascending=True)

    fig, ax = plt.subplots(figsize=(12, 8))

    ax.barh(merged["topic_label"], merged["mean_alignment"])

    ax.set_xlabel("Mean alignment score")
    ax.set_ylabel("Topic")
    ax.set_title("Mean thematic alignment by topic")

    plt.tight_layout()

    output_path = FIGURES_DIR / "topic_alignment_by_topic.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved figure: {output_path}")


def plot_topic_trends(document_topics, topic_labels, top_n=5):


    merged = document_topics.merge(
        topic_labels[["topic_id", "topic_label"]],
        left_on="topic",
        right_on="topic_id",
        how="left",
    )

    merged = merged[merged["topic"] != -1].copy()
    merged["topic_label"] = merged["topic_label"].fillna("Unknown topic")

    top_topics = (
        merged["topic_label"]
        .value_counts()
        .head(top_n)
        .index
        .tolist()
    )

    trend_data = (
        merged[merged["topic_label"].isin(top_topics)]
        .groupby(["year", "topic_label"])
        .size()
        .reset_index(name="n_papers")
    )

    fig, ax = plt.subplots(figsize=(12, 7))

    for topic in top_topics:
        subset = trend_data[trend_data["topic_label"] == topic]
        ax.plot(
            subset["year"],
            subset["n_papers"],
            marker="o",
            label=topic,
        )

    ax.set_xlabel("Year")
    ax.set_ylabel("Number of papers")
    ax.set_title("Yearly evolution of top topics")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = FIGURES_DIR / "top_topic_trends_over_time.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved figure: {output_path}")


def create_final_topic_table(topic_labels, topic_alignment):


    final_table = topic_labels.copy()

    # Normalize column name if needed
    if "count" not in final_table.columns and "Count" in final_table.columns:
        final_table = final_table.rename(columns={"Count": "count"})

    # Add alignment columns only if  missing
    columns_to_add = []

    if "mean_alignment" not in final_table.columns:
        columns_to_add.append("mean_alignment")

    if "median_alignment" not in final_table.columns:
        columns_to_add.append("median_alignment")

    if columns_to_add:
        merge_columns = ["topic"] + columns_to_add

        final_table = final_table.merge(
            topic_alignment[merge_columns],
            left_on="topic_id",
            right_on="topic",
            how="left",
        )

    # Remove extra topic column after merging
    if "topic" in final_table.columns:
        final_table = final_table.drop(columns=["topic"])

    keep_columns = [
        "topic_id",
        "topic_label",
        "count",
        "mean_alignment",
        "median_alignment",
        "example_title_1",
        "manual_topic_label",
        "interpretation_notes",
    ]

    available_columns = [col for col in keep_columns if col in final_table.columns]
    final_table = final_table[available_columns]

    if "mean_alignment" in final_table.columns:
        final_table = final_table.sort_values("mean_alignment", ascending=False)

    output_path = TABLES_DIR / "final_topic_interpretation_table.csv"
    final_table.to_csv(output_path, index=False, encoding="utf-8")

    print(f"Saved table: {output_path}")


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    document_topics = pd.read_csv(TOPICS_DIR / "document_topics.csv")
    topic_alignment = pd.read_csv(TOPICS_DIR / "topic_alignment_summary.csv")
    topic_labels = pd.read_csv(TOPICS_DIR / "topic_labels_template.csv")

    topic_labels["topic_label"] = topic_labels.apply(get_topic_label, axis=1)

    plot_topic_counts_by_journal(document_topics, topic_labels)
    plot_topic_alignment(topic_alignment, topic_labels)
    plot_topic_trends(document_topics, topic_labels, top_n=5)
    create_final_topic_table(topic_labels, topic_alignment)

    print("\nTopic visualization step completed.")


if __name__ == "__main__":
    main()