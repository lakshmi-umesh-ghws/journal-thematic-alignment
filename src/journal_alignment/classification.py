from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def train_journal_classifier(df: pd.DataFrame):


    data = df.dropna(subset=["abstract", "journal_short_name"]).copy()

    X = data["abstract"].astype(str)
    y = data["journal_short_name"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    classifier = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=5,
                    max_features=20000,
                ),
            ),
            (
                "logreg",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    print("Training TF-IDF + Logistic Regression classifier...")
    classifier.fit(X_train, y_train)

    print("Evaluating classifier...")
    y_pred = classifier.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    report_dict = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=classifier.classes_,
    )

    return {
        "classifier": classifier,
        "accuracy": accuracy,
        "classification_report": report_dict,
        "confusion_matrix": cm,
        "labels": classifier.classes_,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
    }


def save_classification_report(report_dict, output_path: Path):


    report_df = pd.DataFrame(report_dict).transpose()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(output_path, encoding="utf-8")

    print(f"Saved classification report to: {output_path}")


def save_confusion_matrix(cm, labels, output_csv_path: Path, output_fig_path: Path):

    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{label}" for label in labels],
        columns=[f"pred_{label}" for label in labels],
    )

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    cm_df.to_csv(output_csv_path, encoding="utf-8")

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[label.upper() for label in labels],
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    display.plot(ax=ax, values_format="d")
    ax.set_title("Journal classification confusion matrix")

    plt.tight_layout()
    output_fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_fig_path, dpi=300)
    plt.close()

    print(f"Saved confusion matrix CSV to: {output_csv_path}")
    print(f"Saved confusion matrix figure to: {output_fig_path}")


def extract_top_terms(classifier, output_path: Path, top_n: int = 30):


    tfidf = classifier.named_steps["tfidf"]
    logreg = classifier.named_steps["logreg"]

    feature_names = tfidf.get_feature_names_out()
    classes = logreg.classes_

    rows = []

    if len(classes) != 2:
        print("Top term extraction currently expects two classes.")
        return

    coefficients = logreg.coef_[0]

    negative_class = classes[0]
    positive_class = classes[1]

    top_positive_indices = coefficients.argsort()[-top_n:][::-1]
    top_negative_indices = coefficients.argsort()[:top_n]

    for idx in top_positive_indices:
        rows.append(
            {
                "journal": positive_class,
                "term": feature_names[idx],
                "coefficient": coefficients[idx],
            }
        )

    for idx in top_negative_indices:
        rows.append(
            {
                "journal": negative_class,
                "term": feature_names[idx],
                "coefficient": coefficients[idx],
            }
        )

    terms_df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    terms_df.to_csv(output_path, index=False, encoding="utf-8")

    print(f"Saved top classifier terms to: {output_path}")