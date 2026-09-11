# Product Search & Ranking Platform

A **lexical candidate re-ranking** baseline and query-level evaluation pipeline on Amazon Science's Shopping Queries (ESCI) dataset.

The current implementation focuses on candidate re-ranking: Amazon ESCI provides a judged candidate set for each query, and the system evaluates how effectively different ranking methods order those candidates. Full-catalog retrieval is a future extension and is not part of the current implementation.

This repository does **not** currently include learning-to-rank, semantic or vector retrieval, personalization, recommendation systems, online experiments, or production serving.

---

## Problem

A search system has to put useful products near the top of a result list. Predicting whether one query-product pair is relevant is a classification problem. Ranking is different: the model is judged by **order**. A relevant product at position 15 is a worse outcome than the same product at position 2, even if both predictions are "relevant."

Ordinary row-level accuracy hides that. It treats every query-product row as an independent yes/no decision and ignores position.

---

## Task formulation

This project evaluates **lexical re-ranking over query-specific candidate sets supplied by the ESCI dataset**.

This stage is **candidate re-ranking, not full-catalog retrieval**.

The ESCI Task 1 data already provides, for each query, a list of up to about 40 judged products. We ask:

> Given the candidate products available for a query, can a lexical relevance model put the more relevant products near the top?

BM25 in this project does **not** search the Amazon product catalog. It re-orders the candidates that ESCI already attached to the query.

---

## Dataset

We use the [Shopping Queries Dataset / ESCI](https://github.com/amazon-science/esci-data) (Reddy et al., 2022). This repository does **not** own or redistribute the raw Amazon files.

Dataset filters:

- `small_version == 1` (Task 1 / reduced set)
- `product_locale == "us"`
- merge examples and products on **both** `product_id` and `product_locale`

### How to obtain the official files

The products parquet is large (on the order of 1 GB). The code will **not** download it for you.

1. Read the official page: https://github.com/amazon-science/esci-data
2. Install Git LFS and clone:

```bash
git lfs install
git clone https://github.com/amazon-science/esci-data.git
```

3. Copy these files into `data/raw/`:

```text
data/raw/shopping_queries_dataset_examples.parquet
data/raw/shopping_queries_dataset_products.parquet
```

If a file is missing, `python scripts/prepare_data.py` fails with the expected paths and these instructions.

---

## Data integrity

Ranking metrics are only as trustworthy as the candidate lists.

- **Product merge key.** `(product_id, product_locale)` must be unique in the product table. Duplicate keys would make a many-to-one merge ambiguous and silently multiply candidates. The preparation step reports duplicates and **stops** rather than silently dropping them.
- **Many-to-one merge.** Many query-product example rows may point at one product record, but each product-locale key should match one product record (`validate="many_to_one"`). We record rows before merge, rows after merge, matched rows, and unmatched rows. Unmatched rows are **kept and reported**, not deleted.
- **Duplicate candidates.** `(query_id, product_id)` pairs are checked explicitly. Conflicting ESCI labels fail the pipeline. Non-conflicting duplicates are kept and documented rather than silently dropped.

These checks matter because NDCG/Recall/MRR assume we know the candidate set and the relevance of each item. Silent deduplication or a Cartesian join would make the baseline look better or worse for bookkeeping reasons, not ranking reasons.

---

## Experimental discipline

The official dataset already has `split ∈ {train, test}`.

- Official `split == "test"` is written to `data/processed/official_test_holdout.parquet` and **left untouched**. This baseline does not iterate on it.
- Official training **query IDs** are split with seed `1234` into:
  - 85% project train
  - 15% project validation
- Split unit is `query_id`, not the query-product row.
- Project train, project validation, and official test query IDs are disjoint.
- All candidate rows for a query stay in the same partition.

Row-level splitting would leak: some candidates for a query could sit in train while others sit in validation. A later model could overfit that query's wording and look stronger than it would on new queries. **The unit being held out is the query.**

Lexical baselines are evaluated on **project validation only**.

---

## Relevance conventions

ESCI labels are Exact, Substitute, Complement, and Irrelevant.

This project's graded mapping, used by NDCG, is:

| Label | Meaning      | Gain |
| ----- | ------------ | ---: |
| E     | Exact        |    3 |
| S     | Substitute   |    2 |
| C     | Complement   |    1 |
| I     | Irrelevant   |    0 |

Exact matches receive the highest gain, substitutes an intermediate gain, complements a lower positive gain, and irrelevant results zero gain.

This is an **internal project convention**. It is not claimed to reproduce Amazon's official benchmark gain scheme.

For binary metrics (Recall@K and MRR):

- **Relevant:** E or S
- **Not relevant:** C or I

Complement is a real shopping relationship, but for "did we return a product the shopper could buy instead of browsing away?" we treat only Exact and Substitute as hits.

---

## Baselines

All three methods re-rank each query's existing candidate set. None of them retrieve from the full catalog.

| Model | What it does |
| ----- | ------------ |
| Random | Shuffle candidates with a fixed seed. Calibrates the metrics. Not competitive. |
| TF-IDF | Cosine similarity between the query and `product_title`, with IDF estimated **inside that query's candidate set**. |
| BM25 | Okapi BM25 on the same title text. |

**BM25 scope (important):** BM25 statistics are computed within each query-specific candidate set for this candidate re-ranking baseline. This is not a full product-catalog search engine and not a production retrieval index.

Tied scores break ties by `product_id` ascending (then `example_id` if needed). Inherited DataFrame order is never the tie-breaker. Repeated runs on the same data produce the same ranking and the same metrics.

Lexical document: `product_title`, lowercased, whitespace-normalized. Model numbers and alphanumeric tokens are kept. No stemming.

---

## Evaluation

Metrics are computed **per query**, then aggregated (mean, median, std, 25th/75th percentile, number of queries). We do not treat query-product rows as i.i.d. observations.

**NDCG@10.** Graded relevance. A relevant product counts more at rank 1 than at rank 10. The score is divided by the DCG of the ideal ordering, so 1.0 is a perfect ranking of that query's candidates.

Formulation used here:

```text
DCG@k = Σ_{i=1..k} gain_i / log2(i + 1)
NDCG@k = DCG@k / IDCG@k
```

If IDCG is 0 (no positive-gain items), NDCG is 0, not NaN.

**Recall@10.** Of the products we consider relevant (E or S) for a query, how many appeared in the top 10? If a query has no E/S items, recall is 0.

**MRR.** How quickly does the ranking return the first relevant (E or S) item? Reciprocal rank is `1/rank` of that first hit, or 0 if none exist.

Accuracy is the wrong lens: swapping two products can leave accuracy unchanged and still ruin the page the user sees.

---

## Results

Populate this section only from `python scripts/run_baseline.py`. Values must come from executed code on project validation, never from placeholders.

After a successful run, summary numbers live in:

```text
artifacts/metrics/baseline_summary.json
artifacts/metrics/baseline_per_query.csv
```

Inspection examples live in:

```text
artifacts/examples/
```

If those files are missing, the official dataset has not been evaluated in this environment yet.

---

## Example behavior

Representative **project validation** queries are exported by the baseline script after a real-data run. They are chosen to show:

1. BM25 doing well
2. BM25 doing poorly
3. a brand / model-number query
4. a broad/generic query
5. lexical overlap that is misleading
6. a query where TF-IDF and BM25 disagree

The official test set is not inspected.

---

## Limitations

- Candidate-set evaluation rather than full-catalog retrieval
- Query-specific BM25 / TF-IDF corpus statistics, not a global index
- Lexical matching only: synonyms, semantic intent, and vocabulary mismatch can fail
- No personalization or recommendation system
- No machine-learned ranking
- No semantic or vector retrieval
- Offline ESCI labels rather than online customer outcomes (clicks, purchases, revenue)

Those lexical failures are known limits of this baseline, not something it tries to paper over with embeddings.

---

## Why these baselines exist

A later learned ranker is only interesting if it beats something honest. Random calibrates the metric scale. TF-IDF is a simple lexical similarity control. BM25 is the standard sparse re-ranking baseline. The current strongest lexical result is the floor any future model must beat on validation.

---

## Reproduce the lexical baseline

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# After the official parquet files are in data/raw/:
python scripts/prepare_data.py
python -m pytest
python scripts/run_baseline.py
jupyter notebook notebooks/01_data_and_baseline_analysis.ipynb
```

---

## Repository layout

Reusable logic lives in `src/`. The notebook is presentation, not the source of truth.

```text
src/data/          load, validate, merge, query-level split
src/retrieval/     text normalization, random, TF-IDF, BM25
src/evaluation/    NDCG / Recall / MRR, query-level aggregation
scripts/           prepare_data.py, run_baseline.py
tests/             synthetic fixtures only; no official ESCI in CI
```

---

## Citation

```text
Reddy, C. K., Màrquez, L., Valero, F., Rao, N., Zaragoza, H.,
Bandyopadhyay, S., Biswas, A., Xing, A., and Subbian, K. (2022).
Shopping Queries Dataset: A Large-Scale ESCI Benchmark for Improving Product Search.
```
