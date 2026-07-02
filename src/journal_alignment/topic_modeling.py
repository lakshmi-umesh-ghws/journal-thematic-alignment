from pathlib import Path

import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer


def fit_bertopic_model(
    df: pd.DataFrame,
    text_column: str = "abstract",
    min_topic_size: int = 50,
):


    texts = df[text_column].fillna("").astype(str).tolist()

    print("Loading embedding model for BERTopic...")
    embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    print("Fitting BERTopic model...")
    vectorizer_model = CountVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=5,
    )

    topic_model = BERTopic(
        embedding_model=embedding_model,
        vectorizer_model=vectorizer_model,
        min_topic_size=min_topic_size,
        nr_topics="auto",
        verbose=True,
    )


    topics, probabilities = topic_model.fit_transform(texts)

    df_with_topics = df.copy()
    df_with_topics["topic"] = topics

    return topic_model, df_with_topics


def create_topic_info(topic_model: BERTopic) -> pd.DataFrame:


    return topic_model.get_topic_info()


def create_topic_alignment_summary(df: pd.DataFrame) -> pd.DataFrame:

    summary = (
        df[df["topic"] != -1]
        .groupby("topic")["alignment_score"]
        .agg(
            n_papers="count",
            mean_alignment="mean",
            median_alignment="median",
            min_alignment="min",
            max_alignment="max",
        )
        .reset_index()
        .sort_values("mean_alignment", ascending=False)
    )

    return summary


def create_topic_year_summary(df: pd.DataFrame) -> pd.DataFrame:


    summary = (
        df[df["topic"] != -1]
        .groupby(["journal_short_name", "year", "topic"])
        .size()
        .reset_index(name="n_papers")
        .sort_values(["journal_short_name", "year", "n_papers"], ascending=[True, True, False])
    )

    return summary


def save_bertopic_model(topic_model: BERTopic, output_dir: Path) -> None:


    output_dir.mkdir(parents=True, exist_ok=True)
    topic_model.save(str(output_dir / "bertopic_model"), serialization="pickle")
    print(f"Saved BERTopic model to: {output_dir / 'bertopic_model'}")