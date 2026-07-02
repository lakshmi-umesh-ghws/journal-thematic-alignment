from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


def load_scope_texts(scopes_dir: Path) -> Dict[str, str]:

    scope_texts = {}

    for file_path in scopes_dir.glob("*_scope.txt"):
        journal_short_name = file_path.stem.replace("_scope", "")
        text = file_path.read_text(encoding="utf-8").strip()
        scope_texts[journal_short_name] = text

    return scope_texts


def compute_alignment_scores(
    papers_df: pd.DataFrame,
    scope_texts: Dict[str, str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> pd.DataFrame:


    df = papers_df.copy()

    print(f"Loading Sentence-BERT model: {model_name}")
    model = SentenceTransformer(model_name)

    print("Encoding journal scope texts...")
    scope_embeddings = {}

    for journal_short_name, scope_text in scope_texts.items():
        embedding = model.encode(
            scope_text,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        scope_embeddings[journal_short_name] = embedding

    print("Encoding paper abstracts...")
    abstracts = df["abstract"].fillna("").astype(str).tolist()

    abstract_embeddings = model.encode(
        abstracts,
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    alignment_scores = []

    for index, row in df.iterrows():
        journal_short_name = row["journal_short_name"]

        if journal_short_name not in scope_embeddings:
            alignment_scores.append(np.nan)
            continue

        scope_embedding = scope_embeddings[journal_short_name]
        abstract_embedding = abstract_embeddings[index]

        # embeddings are normalized, dot product = cosine similarity
        score = float(np.dot(scope_embedding, abstract_embedding))
        alignment_scores.append(score)

    df["alignment_score"] = alignment_scores

    return df


def create_yearly_alignment_summary(df: pd.DataFrame) -> pd.DataFrame:


    summary = (
        df.groupby(["journal_short_name", "year"])["alignment_score"]
        .agg(
            mean_alignment="mean",
            median_alignment="median",
            std_alignment="std",
            min_alignment="min",
            max_alignment="max",
            n_papers="count",
        )
        .reset_index()
        .sort_values(["journal_short_name", "year"])
    )

    return summary


def create_journal_alignment_summary(df: pd.DataFrame) -> pd.DataFrame:


    summary = (
        df.groupby("journal_short_name")["alignment_score"]
        .agg(
            mean_alignment="mean",
            median_alignment="median",
            std_alignment="std",
            min_alignment="min",
            max_alignment="max",
            n_papers="count",
        )
        .reset_index()
        .sort_values("mean_alignment", ascending=False)
    )

    return summary