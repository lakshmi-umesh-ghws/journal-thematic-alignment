# Measuring Thematic Alignment and Drift in AI Journals

This repository contains the code, data-processing pipeline, experimental results, and report materials for the NLP final project:

**Measuring Thematic Alignment and Drift in AI Journals: A Sentence-BERT and BERTopic Study of ESWA and KBS (2015–2025)**

The project studies whether articles published in two Artificial Intelligence journals remain thematically similar to the journals’ stated Aims & Scope over time.

## Project Overview

Scientific journals usually describe their academic focus through an Aims & Scope statement. However, the actual topics of published papers can change as a research field develops. This is especially relevant in Artificial Intelligence, where new methods and subfields appear quickly.

In this project, I compare the semantic meaning of journal Aims & Scope statements with the abstracts of published articles. The main idea is to measure how closely each article align the thematic direction of the journal in which it was published.

The study focuses mainly on two AI-related journals:

* *Expert Systems with Applications* (ESWA)
* *Knowledge-Based Systems* (KBS)

The analysis covers papers published between 2015 and 2025.

## Research Question

The main research question is:

**How has thematic alignment between published articles and journal Aims & Scope evolved in selected Artificial Intelligence journals between 2015 and 2025?**

The project also looks at three related questions:

1. Is there evidence of thematic drift over time?
2. Which topics are most and least aligned with the journal scopes?
3. Can ESWA and KBS be distinguished based on article abstracts?

## Methodology

The project follows a reproducible NLP pipeline:

1. Article metadata and abstracts were collected from Semantic Scholar.
2. The dataset was cleaned by removing missing abstracts, duplicates, and very short abstracts.
3. Journal Aims & Scope texts and article abstracts were encoded using Sentence-BERT.
4. Cosine similarity was used to calculate a thematic alignment score.
5. Yearly alignment scores were analyzed to study possible thematic drift.
6. BERTopic was used to identify major topics in the corpus.
7. A TF-IDF and Logistic Regression classifier was trained as a baseline experiment.
8. High-alignment and low-alignment papers were inspected qualitatively.

## Main Results

The final cleaned dataset contains **5,788 papers**.

| Journal                            | Number of papers |
| ---------------------------------- | ---------------: |
| *Expert Systems with Applications* |            3,465 |
| *Knowledge-Based Systems*          |            2,323 |

The main findings are:

* ESWA has a slightly higher mean alignment score than KBS.
* Both journals show a weak but statistically significant decrease in thematic alignment over time.
* Topics related to knowledge-based systems, decision support, fuzzy decision-making, and rule-based methods show higher alignment.
* Newer or more specialized AI topics, such as privacy-preserving learning, federated learning, segmentation, and optimization, show lower alignment.
* The classifier baseline achieved approximately **69.2% accuracy**, suggesting that the two journals are thematically distinguishable to some extent, although they still overlap strongly.

## Repository Structure

```text
journal-thematic-alignment/
├── config.yaml
├── README.md
├── requirements.txt
├── data/
│   ├── processed/
│       └── papers_clean.csv
│   └── scopes/
│       ├── eswa_scope.txt
│       └── kbs_scope.txt
├── report/
│   ├── final_report_draft.md
│   └── project_summary.md
├── results/
│   ├── figures/
│   ├── tables/
│   └── topics/
├── scripts/
└── src/
    └── journal_alignment/
        ├── __init__.py
        ├── alignment.py
        ├── classification.py
        ├── data_collection.py
        ├── preprocessing.py
        ├── statistics_analysis.py
        ├── topic_modeling.py
        └── visualization.py
```

## Installation

Create and activate a Python virtual environment.

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

Install the required packages:

```powershell
pip install -r requirements.txt
```

## How to Run the Project

The scripts are numbered according to the project pipeline.

### 1. Preprocess the data

```powershell
.\.venv\Scripts\python.exe scripts/03_preprocess_data.py
```

### 2. Check journal scope files

```powershell
.\.venv\Scripts\python.exe scripts/04_check_scopes.py
```

### 3. Compute alignment scores

```powershell
.\.venv\Scripts\python.exe scripts/05_compute_alignment.py
```

### 4. Create visualizations

```powershell
.\.venv\Scripts\python.exe scripts/06_create_visualizations.py
```

### 5. Run statistical analysis

```powershell
.\.venv\Scripts\python.exe scripts/07_statistical_analysis.py
```

### 6. Run topic modeling

```powershell
.\.venv\Scripts\python.exe scripts/08_topic_modeling.py
```

### 7. Inspect topics

```powershell
.\.venv\Scripts\python.exe scripts/09_inspect_topics.py
```

### 8. Create topic visualizations

```powershell
.\.venv\Scripts\python.exe scripts/11_topic_visualizations.py
```

### 9. Train classifier baseline

```powershell
.\.venv\Scripts\python.exe scripts/12_train_classifier.py
```

### 10. Prepare outlier review

```powershell
.\.venv\Scripts\python.exe scripts/13_prepare_outlier_review.py
```

### 11. Generate project summary

```powershell
.\.venv\Scripts\python.exe scripts/14_project_summary.py
```

## Important Output Files

Main result tables:

```text
results/tables/alignment_scores.csv
results/tables/yearly_alignment_summary.csv
results/tables/journal_alignment_summary.csv
results/tables/trend_statistics.csv
results/tables/journal_comparison_statistics.csv
results/tables/classifier_report.csv
results/tables/classifier_confusion_matrix.csv
results/tables/outlier_review_for_report.csv
```

Main figures:

```text
results/figures/alignment_distribution_by_journal.png
results/figures/yearly_alignment_trend.png
results/figures/topic_alignment_by_topic.png
results/figures/classifier_confusion_matrix.png
```

Topic modeling outputs:

```text
results/topics/topic_info.csv
results/topics/document_topics.csv
results/topics/topic_alignment_summary.csv
results/topics/topic_year_summary.csv
```

Report files:

```text
report/final_report_draft.md
report/project_summary.md
```

## Notes on Reproducibility

The project is organized into reusable Python modules inside `src/journal_alignment/`. The scripts in the `scripts/` folder call these modules and reproduce the main stages of the analysis.

The raw and processed data folders may be excluded from GitHub if the files are too large. In that case, the pipeline can be reproduced by running the data collection and preprocessing scripts again, depending on the availability of metadata from Semantic Scholar.

## AI Usage Disclaimer

Parts of this project were developed with the assistance of OpenAI’s ChatGPT(GPT-5.5). The AI was used to support workflow structuring, code explanation, debugging, drafting of descriptive text, and improving readability. All generated content and code were reviewed, edited, tested, and validated by me. I take full responsibility for the final project, including its methodology, results, interpretation, and academic integrity.

## Author

Lakshmi Umesh  
Natural Language Processing  
University of Milan  
2026
