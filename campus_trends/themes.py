"""Group similar report descriptions and save charts in visualizations/."""

import argparse
from pathlib import Path
import re
import textwrap
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

from campus_trends.data import load_reports


def clean_description(text: str) -> str:
    text = re.sub(r"https?://\S+|www\.\S+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\b[\w.+-]+@[\w.-]+\.\w+\b", " ", text)
    text = re.sub(r"(?<!\w)\+?\d[\d\s().-]{7,}\d(?!\w)", " ", text)
    return text.lower()


def theme_label(theme: str, keywords: str) -> str:
    if theme in ("Mixed themes", "Unassigned"):
        return theme
    phrases: list[str] = []
    seen: set[str] = set()
    for phrase in keywords.split(", "):
        words = set(phrase.split())
        if not words & seen:
            phrases.append(phrase)
            seen.update(words)
        if len(phrases) == 3:
            break
    return f"{theme}: {', '.join(phrases)}"


def find_themes(
    reports: pd.DataFrame, topics: int = 6
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Assign a dominant theme, retaining mixed and unmatched descriptions."""
    if topics < 2:
        raise ValueError("Choose at least two themes.")
    if len(reports) < max(10, topics * 2):
        raise ValueError("Need at least 10 reports and twice as many reports as themes.")
    location_words: set[str] = set()
    for column in ("College/University", "City of Incident", "State of Incident"):
        for value in reports[column]:
            location_words.update(re.findall(r"[a-z]{2,}", value.lower()))
    stop_words = sorted(
        set(ENGLISH_STOP_WORDS)
        | location_words
        | {
            "student", "students", "campus", "university", "college", "reported",
            "incident", "took", "place", "included", "including", "held",
            "displayed", "messages", "message", "protesters", "protester",
            "individual", "individuals", "unidentified",
        }
    )
    vectorizer = TfidfVectorizer(
        preprocessor=clean_description,
        stop_words=stop_words,
        token_pattern=r"(?u)\b[a-z][a-z]+\b",
        min_df=2,
        max_df=0.85,
        max_features=5000,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(reports["Description"])
    if topics >= min(matrix.shape):
        raise ValueError("Too few distinct words or reports for this many themes.")
    model = NMF(n_components=topics, init="nndsvda", random_state=42, max_iter=1000)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        weights = model.fit_transform(matrix)
    names = vectorizer.get_feature_names_out()
    terms = [
        ", ".join(names[np.argsort(component)[::-1][:8]])
        for component in model.components_
    ]
    strengths = weights.sum(axis=1)
    assignments = np.where(strengths > 0, weights.argmax(axis=1), -1)
    weight_shares = np.divide(
        weights.max(axis=1), strengths, out=np.zeros(len(weights)), where=strengths > 0
    )
    theme_names = [f"Theme {index + 1}" for index in range(topics)] + [
        "Mixed themes", "Unassigned"
    ]
    assigned = pd.DataFrame(
        {
            "Date": reports["Date"].to_numpy(),
            "theme": [
                (
                    "Unassigned" if index < 0
                    else "Mixed themes" if weight_shares[position] < 0.5
                    else f"Theme {index + 1}"
                )
                for position, index in enumerate(assignments)
            ],
            # Model weights are not probabilities that a label is correct.
            "dominant_weight_share": weight_shares,
        }
    )
    months = pd.period_range(reports["Date"].min(), reports["Date"].max(), freq="M")
    monthly = pd.crosstab(assigned["Date"].dt.to_period("M"), assigned["theme"])
    monthly = monthly.reindex(index=months, columns=theme_names, fill_value=0)
    monthly.index = monthly.index.astype(str)
    monthly.index.name = "month"
    shares = monthly.div(monthly.sum(axis=1).replace(0, np.nan), axis=0)
    summary = pd.DataFrame(
        {
            "theme": theme_names,
            "keywords": terms + [
                "No single theme has at least half the model weight",
                "No retained words matched a theme",
            ],
            "reports": assigned["theme"].value_counts().reindex(
                theme_names, fill_value=0
            ).to_numpy(),
        }
    )
    return summary, monthly, shares, assigned


def plot_themes(
    summary: pd.DataFrame, monthly: pd.DataFrame, shares: pd.DataFrame, output: Path
) -> None:
    palette = plt.get_cmap("tab10")
    colors = {theme: palette(index % 10) for index, theme in enumerate(monthly.columns)}
    labels = {}
    for row in summary.itertuples(index=False):
        labels[row.theme] = theme_label(row.theme, row.keywords)
    fig, ax = plt.subplots(figsize=(13, max(6, len(summary) * 0.8)))
    y = np.arange(len(summary))
    ax.barh(y, summary["reports"], color=[colors[theme] for theme in summary["theme"]])
    ax.set_yticks(y, [
        textwrap.fill(labels[row.theme], width=42)
        for row in summary.itertuples(index=False)
    ], fontsize=10)
    ax.invert_yaxis()
    ax.bar_label(ax.containers[0], padding=5)
    ax.set(
        title="What recurring language appears in the descriptions?",
        xlabel="Reports with this as their dominant theme",
        xlim=(0, max(1, summary["reports"].max()) * 1.15),
    )
    fig.subplots_adjust(left=0.34, right=0.96, bottom=0.16, top=0.91)
    fig.text(
        0.03, 0.035,
        "Suggested word groups, not verified incident types. Method: TF-IDF + NMF.\n"
        "A theme needs at least half a report's model weight; otherwise it is mixed. "
        "This rule is not a measured accuracy threshold.",
        fontsize=9,
    )
    fig.savefig(output / "description_themes.png", dpi=150)
    plt.close(fig)

    month_labels = [
        f"{month}\n({int(total)} reports)"
        for month, total in monthly.sum(axis=1).items()
    ]
    for data, filename, title, ylabel in (
        (
            monthly, "theme_counts_by_month.png",
            "How many reports match each theme each month?", "Reports",
        ),
        (
            shares * 100, "theme_share_by_month.png",
            "How does the mix of themes change each month?", "Percent of monthly reports",
        ),
    ):
        fig, ax = plt.subplots(figsize=(13, 8))
        for theme in monthly.columns:
            if monthly[theme].sum():
                ax.plot(
                    np.arange(len(data)), data[theme], marker="o",
                    label=labels[theme], color=colors[theme],
                )
        ax.set_xticks(np.arange(len(data)), month_labels, rotation=35, ha="right", fontsize=9)
        ax.set(title=title, ylabel=ylabel)
        ax.grid(alpha=0.2)
        ax.legend(loc="upper left", bbox_to_anchor=(0, -0.23), ncol=2, fontsize=9)
        fig.subplots_adjust(left=0.08, right=0.97, top=0.91, bottom=0.35)
        fig.text(
            0.04, 0.025,
            "Groups describe wording, not motives or causes. Small months have unstable "
            "percentages; boundary months may be incomplete.\n"
            "Groups can reflect reporting templates. All descriptions were used to "
            "discover themes; these are not future predictions.",
            fontsize=9,
        )
        fig.savefig(output / filename, dpi=150)
        plt.close(fig)


def run_themes(input_path: Path, output: Path, topics: int = 6) -> None:
    reports = load_reports(input_path)
    summary, monthly, shares, _ = find_themes(reports, topics)
    output.mkdir(parents=True, exist_ok=True)
    plot_themes(summary, monthly, shares, output)
    print(f"Saved three theme charts to {output}. No HTML or raw descriptions written.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/campus_reports.csv"))
    parser.add_argument("--output", type=Path, default=Path("visualizations"))
    parser.add_argument("--topics", type=int, default=6)
    args = parser.parse_args()
    run_themes(args.input, args.output, args.topics)


if __name__ == "__main__":
    main()
