#!/usr/bin/env python3
"""Plot approved validation NDCG@10 progression across ranking models."""

from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "docs/images/model_progression.png"

MODELS = [
    "Random",
    "BM25",
    "TF-IDF",
    "XGBRanker",
    "Semantic XGBRanker",
]
NDCG_AT_10 = [0.7611, 0.8124, 0.8173, 0.8236, 0.8441]

PRIOR_COLOR = "#64748B"
SEMANTIC_COLOR = "#0F766E"
COLORS = [PRIOR_COLOR] * 4 + [SEMANTIC_COLOR]


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5), dpi=180)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    y_pos = list(range(len(MODELS)))
    x_min, x_max = 0.75, 0.86

    ax.hlines(y_pos, x_min, NDCG_AT_10, colors=COLORS, linewidth=1.4)
    ax.scatter(
        NDCG_AT_10,
        y_pos,
        s=[90] * 4 + [130],
        c=COLORS,
        zorder=3,
        linewidths=0,
    )

    for y, value, color in zip(y_pos, NDCG_AT_10, COLORS):
        is_semantic = color == SEMANTIC_COLOR
        ax.text(
            value + 0.0025,
            y,
            f"{value:.4f}",
            va="center",
            ha="left",
            fontsize=11,
            fontweight="bold" if is_semantic else "normal",
            color=SEMANTIC_COLOR if is_semantic else "#334155",
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(MODELS, fontsize=12)
    labels = ax.get_yticklabels()
    labels[-1].set_fontweight("bold")
    labels[-1].set_color(SEMANTIC_COLOR)

    ax.set_xlabel("NDCG@10", fontsize=12, labelpad=8)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-0.55, 4.55)
    ax.set_xticks([0.75, 0.78, 0.81, 0.84, 0.86])
    ax.tick_params(axis="x", labelsize=10, colors="#475569")
    ax.tick_params(axis="y", length=0, pad=6)

    ax.text(
        0,
        1.16,
        "Validation Ranking Quality",
        transform=ax.transAxes,
        fontsize=16,
        fontweight="bold",
        color="#0F172A",
        ha="left",
        va="bottom",
    )
    ax.text(
        0,
        1.07,
        "Amazon ESCI candidate re-ranking · NDCG@10",
        transform=ax.transAxes,
        fontsize=11,
        color="#475569",
        ha="left",
        va="bottom",
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#CBD5E1")
    ax.xaxis.grid(True, color="#E2E8F0", linewidth=0.8)
    ax.set_axisbelow(True)

    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(OUTPUT_PATH, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
