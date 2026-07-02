import pandas as pd
from scipy.stats import spearmanr, linregress, mannwhitneyu


def compute_trend_statistics(df: pd.DataFrame) -> pd.DataFrame:


    rows = []

    for journal in sorted(df["journal_short_name"].unique()):
        subset = df[df["journal_short_name"] == journal].dropna(
            subset=["year", "alignment_score"]
        )

        spearman_corr, spearman_p = spearmanr(
            subset["year"],
            subset["alignment_score"],
        )

        regression = linregress(
            subset["year"],
            subset["alignment_score"],
        )

        rows.append(
            {
                "journal_short_name": journal,
                "n_papers": len(subset),
                "spearman_correlation": spearman_corr,
                "spearman_p_value": spearman_p,
                "linear_slope": regression.slope,
                "linear_intercept": regression.intercept,
                "linear_p_value": regression.pvalue,
                "r_squared": regression.rvalue ** 2,
            }
        )

    return pd.DataFrame(rows)


def compare_journals(df: pd.DataFrame) -> pd.DataFrame:

    journals = sorted(df["journal_short_name"].unique())

    if len(journals) != 2:
        raise ValueError("This comparison expects exactly two journals.")

    journal_a = journals[0]
    journal_b = journals[1]

    scores_a = df[df["journal_short_name"] == journal_a]["alignment_score"].dropna()
    scores_b = df[df["journal_short_name"] == journal_b]["alignment_score"].dropna()

    test_result = mannwhitneyu(
        scores_a,
        scores_b,
        alternative="two-sided",
    )

    return pd.DataFrame(
        [
            {
                "journal_a": journal_a,
                "journal_b": journal_b,
                "n_a": len(scores_a),
                "n_b": len(scores_b),
                "mean_a": scores_a.mean(),
                "mean_b": scores_b.mean(),
                "median_a": scores_a.median(),
                "median_b": scores_b.median(),
                "mannwhitney_u_statistic": test_result.statistic,
                "p_value": test_result.pvalue,
            }
        ]
    )