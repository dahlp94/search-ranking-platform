#!/usr/bin/env python3
"""Plot semantic-minus-reference ΔNDCG@10 by validation-query segment."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SEGMENTS_PATH = ROOT / "artifacts/metrics/semantic_segments.csv"
OUTPUT_PATH = ROOT / "docs/images/semantic_segment_gains.png"

PRIOR_COLOR = "#64748B"
SEMANTIC_COLOR = "#0F766E"
TEXT_COLOR = "#334155"
MUTED_TEXT = "#475569"
TITLE_COLOR = "#0F172A"

OVERLAP_ORDER = ["low", "medium", "high"]
OVERLAP_DISPLAY = {
    "low": "Low",
    "medium": "Medium",
    "high": "High",
}
OVERLAP_HIGHLIGHT = "low"

AGREEMENT_ORDER = ["high disagreement", "mixed", "high agreement"]
AGREEMENT_DISPLAY = {
    "high disagreement": "High disagreement",
    "mixed": "Mixed",
    "high agreement": "High agreement",
}
AGREEMENT_HIGHLIGHT = "high disagreement"


def load_named_segments(
    table: pd.DataFrame,
    names: list[str],
) -> pd.DataFrame:
    rows = []
    for name in names:
        match = table[table["segment"] == name]
        if match.empty:
            raise ValueError(f"Expected segment {name!r} not found in {SEGMENTS_PATH}")
        rows.append(match.iloc[0])
    return pd.DataFrame(rows)


def draw_panel(ax, rows: pd.DataFrame, display: dict[str, str], highlight: str, title: str) -> None:
    y_pos = list(range(len(rows)))
    values = rows["delta_ndcg_at_10"].to_numpy()
    colors = [
        SEMANTIC_COLOR if segment == highlight else PRIOR_COLOR
        for segment in rows["segment"]
    ]
    labels = [
        f"{display[segment]} (n={int(n):,})"
        for segment, n in zip(rows["segment"], rows["n_queries"])
    ]

    ax.barh(y_pos, values, color=colors, height=0.58, edgecolor="none")
    ax.axvline(0, color="#CBD5E1", linewidth=1.0, zorder=0)

    for y, value, color in zip(y_pos, values, colors):
        is_highlight = color == SEMANTIC_COLOR
        ax.text(
            value + 0.0012,
            y,
            f"{value:+.4f}",
            va="center",
            ha="left",
            fontsize=11,
            fontweight="bold" if is_highlight else "normal",
            color=SEMANTIC_COLOR if is_highlight else TEXT_COLOR,
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=11)
    for label, color in zip(ax.get_yticklabels(), colors):
        if color == SEMANTIC_COLOR:
            label.set_fontweight("bold")
            label.set_color(SEMANTIC_COLOR)

    ax.set_title(title, fontsize=12, color=TITLE_COLOR, pad=10, loc="left")
    ax.set_xlabel("Mean ΔNDCG@10", fontsize=11, labelpad=8)
    ax.set_ylim(2.55, -0.55)
    ax.tick_params(axis="x", labelsize=10, colors=MUTED_TEXT)
    ax.tick_params(axis="y", length=0, pad=6)

    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#CBD5E1")
    ax.xaxis.grid(True, color="#E2E8F0", linewidth=0.8)
    ax.set_axisbelow(True)


def main() -> None:
    table = pd.read_csv(SEGMENTS_PATH)
    overlap = load_named_segments(table, OVERLAP_ORDER)
    agreement = load_named_segments(table, AGREEMENT_ORDER)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=180)
    fig.patch.set_facecolor("white")

    draw_panel(
        axes[0],
        overlap,
        OVERLAP_DISPLAY,
        OVERLAP_HIGHLIGHT,
        "By lexical overlap",
    )
    draw_panel(
        axes[1],
        agreement,
        AGREEMENT_DISPLAY,
        AGREEMENT_HIGHLIGHT,
        "By semantic / TF-IDF agreement",
    )

    x_max = max(
        overlap["delta_ndcg_at_10"].max(),
        agreement["delta_ndcg_at_10"].max(),
    )
    for ax in axes:
        ax.set_xlim(0, x_max + 0.012)
        ax.set_xticks([0.00, 0.01, 0.02, 0.03, 0.04, 0.05])

    fig.suptitle(
        "Where Semantic Relevance Helps Most",
        fontsize=16,
        fontweight="bold",
        color=TITLE_COLOR,
        x=0.02,
        ha="left",
    )
    fig.text(
        0.02,
        0.90,
        "Mean semantic-minus-reference ΔNDCG@10 by validation-query segment",
        fontsize=11,
        color=MUTED_TEXT,
        ha="left",
    )

    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(OUTPUT_PATH, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
