#!/usr/bin/env python3
"""Plot a saved large-improvement query as a ranking comparison."""

import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_PATH = ROOT / "artifacts/examples/semantic_query_examples.csv"
OUTPUT_PATH = ROOT / "docs/images/semantic_ranking_example.png"

QUERY = (
    "Since I have a desktop any laptop for work "
    "with decent specifications would do"
)

PRIOR_COLOR = "#64748B"
SEMANTIC_COLOR = "#0F766E"
TEXT_COLOR = "#334155"
MUTED_TEXT = "#475569"
TITLE_COLOR = "#0F172A"
RULE_COLOR = "#E2E8F0"

LABEL_NAMES = {
    "E": "Exact",
    "S": "Substitute",
    "C": "Complement",
    "I": "Irrelevant",
}
POSITIVE_LABELS = {"E", "S"}


def shorten_title(title: str, width: int = 62, max_lines: int = 2) -> str:
    cleaned = " ".join(str(title).split())
    lines = textwrap.wrap(
        cleaned,
        width=width,
        break_long_words=True,
        break_on_hyphens=True,
    )
    if not lines:
        return ""
    if len(lines) <= max_lines:
        return "\n".join(lines)

    kept = lines[:max_lines]
    last = kept[-1].rstrip(" .,:;")
    limit = max(width - 1, 1)
    if len(last) > limit:
        last = last[:limit].rstrip()
    kept[-1] = last + "…"
    return "\n".join(kept)


def load_query_candidates(path: Path, query: str) -> pd.DataFrame:
    examples = pd.read_csv(path)
    rows = examples[examples["query"] == query].copy()
    if rows.empty:
        raise SystemExit(f"Query not found in {path}: {query!r}")
    return rows


def draw_panel(ax, panel_title: str, rows: pd.DataFrame, rank_col: str, accent: str) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_facecolor("white")

    ax.text(
        0.0,
        1.05,
        panel_title,
        fontsize=13,
        fontweight="bold",
        color=accent,
        ha="left",
        va="bottom",
    )

    n_rows = len(rows)
    row_h = 1 / n_rows

    for i, item in enumerate(rows.itertuples(index=False)):
        y_top = 1 - i * row_h
        y_mid = y_top - row_h / 2
        rank = int(getattr(item, rank_col))
        label = item.esci_label
        label_color = SEMANTIC_COLOR if label in POSITIVE_LABELS else PRIOR_COLOR

        if i > 0:
            ax.plot(
                [0.0, 1.0],
                [y_top, y_top],
                color=RULE_COLOR,
                linewidth=0.8,
                solid_capstyle="butt",
            )

        ax.text(
            0.04,
            y_mid,
            str(rank),
            fontsize=15,
            fontweight="bold",
            color=accent,
            ha="center",
            va="center",
        )
        ax.text(
            0.10,
            y_mid + 0.028,
            shorten_title(item.product_title),
            fontsize=10.5,
            color=TITLE_COLOR,
            ha="left",
            va="center",
            linespacing=1.28,
        )
        ax.text(
            0.10,
            y_mid - 0.055,
            f"{label} · {LABEL_NAMES[label]}",
            fontsize=9,
            fontweight="bold" if label in POSITIVE_LABELS else "normal",
            color=label_color,
            ha="left",
            va="center",
        )


def main() -> None:
    candidates = load_query_candidates(EXAMPLES_PATH, QUERY)
    delta = float(candidates["delta_ndcg_at_10"].iloc[0])

    reference = candidates.sort_values("reference_rank").head(5)
    semantic = candidates.sort_values("semantic_rank").head(5)
    if len(reference) != 5 or len(semantic) != 5:
        raise SystemExit("Expected five candidates in each ranking panel.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(13, 7), dpi=180)
    fig.patch.set_facecolor("white")

    fig.text(
        0.04,
        0.955,
        "What Changed in the Ranking?",
        fontsize=16,
        fontweight="bold",
        color=TITLE_COLOR,
        ha="left",
        va="top",
    )
    fig.text(
        0.04,
        0.905,
        "Example validation query with a large semantic-ranking improvement",
        fontsize=11,
        color=MUTED_TEXT,
        ha="left",
        va="top",
    )

    fig.text(
        0.04,
        0.845,
        f'"{QUERY}"',
        fontsize=12,
        color=TEXT_COLOR,
        ha="left",
        va="top",
        style="italic",
    )
    fig.text(
        0.04,
        0.790,
        f"ΔNDCG@10  {delta:+.4f}",
        fontsize=13,
        fontweight="bold",
        color=SEMANTIC_COLOR,
        ha="left",
        va="top",
    )

    ax_left = fig.add_axes([0.04, 0.11, 0.44, 0.57])
    ax_right = fig.add_axes([0.52, 0.11, 0.44, 0.57])

    draw_panel(ax_left, "Reference XGBRanker", reference, "reference_rank", PRIOR_COLOR)
    draw_panel(ax_right, "Semantic XGBRanker", semantic, "semantic_rank", SEMANTIC_COLOR)

    fig.text(
        0.04,
        0.04,
        "Same candidate set · frozen validation models · ESCI relevance judgments",
        fontsize=9,
        color=MUTED_TEXT,
        ha="left",
        va="center",
    )

    fig.savefig(OUTPUT_PATH, dpi=180, facecolor="white")
    plt.close(fig)
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
