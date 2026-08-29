#!/usr/bin/env python3
"""Evaluate lexical re-ranking baselines on project validation queries.

Does not load or iterate on the official ESCI test holdout.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import DEFAULT_K
from src.data.load_data import PROCESSED_DIR
from src.evaluation.analysis import select_inspection_queries
from src.evaluation.evaluate import compare_models, per_query_metrics, summarize_query_metrics, write_json
from src.retrieval.baselines import rank_random, rank_tfidf
from src.retrieval.bm25 import rank_candidates
from src.retrieval.ranking import rerank_dataset

ARTIFACTS = PROJECT_ROOT / "artifacts"
METRICS_DIR = ARTIFACTS / "metrics"
EXAMPLES_DIR = ARTIFACTS / "examples"

MODELS = {
    "random": rank_random,
    "tfidf": rank_tfidf,
    "bm25": rank_candidates,
}


def _markdown_table(df: pd.DataFrame, max_rows: int = 10) -> str:
    view = df.head(max_rows)
    if view.empty:
        return "_no rows_"
    cols = [str(col) for col in view.columns]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = []
    for record in view.to_dict(orient="records"):
        cells = [str(record[col]).replace("|", "\\|") for col in view.columns]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep, *rows])


def _load_validation() -> pd.DataFrame:
    path = PROCESSED_DIR / "project_validation.parquet"
    if not path.is_file():
        raise FileNotFoundError(
            f"Processed validation table not found: {path}\n"
            "Run `python scripts/prepare_data.py` after placing the official "
            "ESCI parquet files in data/raw/."
        )
    df = pd.read_parquet(path)
    print(f"[eval] loaded project validation: {df.shape[0]:,} rows, {df['query_id'].nunique():,} queries")
    return df


def _wide_metrics(per_query_by_model: dict[str, pd.DataFrame], k: int) -> pd.DataFrame:
    wide = None
    for name, table in per_query_by_model.items():
        piece = table.rename(
            columns={
                f"ndcg@{k}": f"ndcg@{k}_{name}",
                f"recall@{k}": f"recall@{k}_{name}",
                "mrr": f"mrr_{name}",
            }
        )
        cols = ["query_id", "query", "n_candidates", f"ndcg@{k}_{name}", f"recall@{k}_{name}", f"mrr_{name}"]
        piece = piece[cols]
        wide = piece if wide is None else wide.merge(
            piece.drop(columns=["query", "n_candidates"]),
            on="query_id",
            how="inner",
            validate="one_to_one",
        )
    return wide


def _write_examples(
    selected: list[dict],
    ranked_by_model: dict[str, pd.DataFrame],
) -> None:
    EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    write_json(EXAMPLES_DIR / "selected_queries.json", selected)

    lines = [
        "# Validation query inspection",
        "",
        "These examples come from **project validation** only. The official test holdout was not inspected.",
        "",
    ]
    preview_cols = [
        "predicted_rank",
        "product_id",
        "product_title",
        "esci_label",
        "relevance_gain",
        "bm25_score",
        "tfidf_score",
        "random_score",
    ]
    for item in selected:
        query_id = item["query_id"]
        lines.extend(
            [
                f"## {item['kind']}: {item['query']}",
                "",
                f"- query_id: `{query_id}`",
                f"- why selected: {item['reason']}",
                f"- NDCG@10 BM25={item['ndcg@10_bm25']:.4f}, TF-IDF={item['ndcg@10_tfidf']:.4f}, "
                f"Random={item['ndcg@10_random']:.4f}",
                "",
            ]
        )
        merged = None
        for name, ranked in ranked_by_model.items():
            piece = ranked.loc[ranked["query_id"] == query_id].copy()
            score_col = {"random": "random_score", "tfidf": "tfidf_score", "bm25": "bm25_score"}[name]
            keep = [
                col
                for col in [
                    "query_id",
                    "query",
                    "product_id",
                    "product_title",
                    "esci_label",
                    "relevance_gain",
                    score_col,
                    "predicted_rank",
                ]
                if col in piece.columns
            ]
            piece = piece[keep].rename(columns={"predicted_rank": f"rank_{name}"})
            merged = piece if merged is None else merged.merge(
                piece[["product_id", score_col, f"rank_{name}"]],
                on="product_id",
                how="outer",
            )
        if merged is None:
            continue
        if "bm25_score" in merged.columns:
            merged = merged.sort_values("bm25_score", ascending=False, kind="mergesort")
        csv_name = f"query_{query_id}_{item['kind']}.csv"
        merged.to_csv(EXAMPLES_DIR / csv_name, index=False)
        show = [col for col in preview_cols if col in merged.columns]
        # Rank columns were renamed; show rank_bm25 if present.
        extra = [col for col in ["rank_bm25", "rank_tfidf", "rank_random"] if col in merged.columns]
        preview = merged[show + extra].head(10) if show or extra else merged.head(10)
        lines.append(_markdown_table(preview))
        lines.append("")
        lines.append(f"Full ranked list: `{csv_name}`")
        lines.append("")

    (EXAMPLES_DIR / "error_analysis.md").write_text("\n".join(lines))
    print(f"[eval] wrote inspection files under {EXAMPLES_DIR}")


def main() -> None:
    k = DEFAULT_K
    validation = _load_validation()

    ranked_by_model: dict[str, pd.DataFrame] = {}
    per_query_by_model: dict[str, pd.DataFrame] = {}
    summaries: dict[str, dict] = {}

    for name, rank_fn in MODELS.items():
        print(f"[eval] ranking with {name} ...")
        ranked = rerank_dataset(validation, rank_fn, k=None)
        ranked_by_model[name] = ranked
        per_query = per_query_metrics(ranked, k=k, model_name=name)
        per_query_by_model[name] = per_query
        summaries[name] = summarize_query_metrics(per_query, k=k)
        mean_ndcg = summaries[name][f"ndcg@{k}"]["mean"]
        mean_recall = summaries[name][f"recall@{k}"]["mean"]
        mean_mrr = summaries[name]["mrr"]["mean"]
        print(
            f"[eval] {name}: NDCG@{k}={mean_ndcg:.4f}  "
            f"Recall@{k}={mean_recall:.4f}  MRR={mean_mrr:.4f}"
        )

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    wide = _wide_metrics(per_query_by_model, k)
    wide_path = METRICS_DIR / "baseline_per_query.csv"
    wide.to_csv(wide_path, index=False)

    comparison = {
        "bm25_vs_tfidf": compare_models(per_query_by_model, metric=f"ndcg@{k}", left="bm25", right="tfidf"),
        "bm25_vs_random": compare_models(per_query_by_model, metric=f"ndcg@{k}", left="bm25", right="random"),
        "tfidf_vs_random": compare_models(per_query_by_model, metric=f"ndcg@{k}", left="tfidf", right="random"),
    }
    payload = {
        "k": k,
        "partition": "project_validation",
        "n_queries": int(validation["query_id"].nunique()),
        "n_rows": int(len(validation)),
        "models": summaries,
        "comparison": comparison,
        "mean_table": {
            name: {
                f"ndcg@{k}": summaries[name][f"ndcg@{k}"]["mean"],
                f"recall@{k}": summaries[name][f"recall@{k}"]["mean"],
                "mrr": summaries[name]["mrr"]["mean"],
            }
            for name in MODELS
        },
    }
    summary_path = METRICS_DIR / "baseline_summary.json"
    write_json(summary_path, payload)
    print(f"[eval] wrote {wide_path}")
    print(f"[eval] wrote {summary_path}")

    selected = select_inspection_queries(validation, wide, ranked_by_model["bm25"], k=k)
    _write_examples(selected, ranked_by_model)

    print("[eval] mean metrics (project validation):")
    print(f"{'Model':<10} {'NDCG@10':>10} {'Recall@10':>10} {'MRR':>10}")
    for name in MODELS:
        row = payload["mean_table"][name]
        print(
            f"{name:<10} {row[f'ndcg@{k}']:10.4f} {row[f'recall@{k}']:10.4f} {row['mrr']:10.4f}"
        )


if __name__ == "__main__":
    main()
