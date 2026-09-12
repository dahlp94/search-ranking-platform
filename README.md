# Product Search & Ranking Platform

A product-search **candidate re-ranking** project built on Amazon Science's Shopping Queries (ESCI) dataset.

The project progresses from lexical baselines to supervised learning-to-rank and emphasizes:

* trustworthy ranking evaluation;
* query-level train/validation separation;
* BM25 and TF-IDF baselines;
* candidate-level ranking features;
* query-grouped XGBRanker training;
* paired query-level model comparison;
* feature ablation and failure analysis.

The current implementation focuses on **re-ranking candidate products already supplied by ESCI**. It does not yet perform full-catalog retrieval.


## Problem

Product search is an ordering problem.

For a query, the goal is not simply to classify products as relevant or irrelevant. More relevant products should appear earlier in the result list.

This project asks:

> Given a query and an already-provided set of candidate products, can we rank the most relevant products near the top?

Amazon ESCI provides judged query-product candidate sets, making this a **candidate re-ranking** problem rather than a full-catalog retrieval problem.


## Dataset

The project uses the [Amazon Science Shopping Queries Dataset (ESCI)](https://github.com/amazon-science/esci-data).

Current filters:

* `small_version == 1`
* U.S. locale
* examples and products merged on `(product_id, product_locale)`

The raw Amazon files are not redistributed in this repository.

Expected files:

```text
data/raw/shopping_queries_dataset_examples.parquet
data/raw/shopping_queries_dataset_products.parquet
```


## Data integrity

Ranking metrics are only useful if the candidate sets are trustworthy.

The preparation pipeline checks:

* uniqueness of `(product_id, product_locale)` product keys;
* many-to-one example-to-product merges;
* row counts before and after joins;
* matched and unmatched product rows;
* duplicate `(query_id, product_id)` candidates;
* conflicting relevance labels.

Potential integrity failures stop the pipeline rather than being silently corrected.


## Experimental design

The official ESCI test partition is isolated and left untouched during model development.

Official training queries are split by `query_id` into:

* **85% project train**
* **15% project validation**

Using the project seed:

```text
1234
```

All candidate rows belonging to the same query remain in the same partition.

Current project split:

```text
Train rows:          356,615
Train queries:        17,754

Validation rows:      63,038
Validation queries:    3,134
```

Train and validation query IDs are required to be disjoint.


## Relevance labels

ESCI relevance labels are mapped to graded gains:

| Label | Meaning    | Gain |
| ----- | ---------- | ---: |
| E     | Exact      |    3 |
| S     | Substitute |    2 |
| C     | Complement |    1 |
| I     | Irrelevant |    0 |

For binary Recall@10 and MRR calculations:

```text
Relevant:     E, S
Not relevant: C, I
```


## Evaluation

Metrics are calculated **per query** and then aggregated across queries.

Primary metric:

### NDCG@10

NDCG rewards highly relevant products appearing near the top while accounting for graded relevance.

```text
DCG@k = Σ gain_i / log2(i + 1)

NDCG@k = DCG@k / IDCG@k
```

Additional metrics:

### Recall@10

Measures how many relevant products appear within the first 10 results.

### MRR

Measures how early the first relevant product appears.

The query, rather than the individual query-product row, is the primary evaluation unit.


# Lexical Baselines

Three initial candidate-ranking methods establish the benchmark.

| Model  | Description                                                      |
| ------ | ---------------------------------------------------------------- |
| Random | Deterministic random ordering used to calibrate the metric scale |
| BM25   | Candidate-local BM25 using product titles                        |
| TF-IDF | Candidate-local TF-IDF cosine similarity using product titles    |

These methods reorder the candidate products already associated with each ESCI query.

They do **not** search the full Amazon product catalog.

### Validation results

| Model  |    NDCG@10 |  Recall@10 |        MRR |
| ------ | ---------: | ---------: | ---------: |
| Random |     0.7611 |     0.5684 |     0.8757 |
| BM25   |     0.8124 |     0.5906 |     0.8988 |
| TF-IDF | **0.8173** | **0.5922** | **0.9060** |

TF-IDF became the strongest lexical benchmark.


# Learning-to-Rank

A supervised XGBRanker was trained using candidate-level signals from several feature families.

### Lexical scores

```text
bm25_score
tfidf_score
```

### Query-product overlap

```text
shared_token_count
query_token_coverage
title_token_coverage
exact_query_in_title
query_bullet_coverage
```

### Structured and length signals

```text
brand_match
title_token_count
token_length_difference
```

The ranker uses:

```text
XGBRanker
objective = rank:ndcg
trees = 200
```

Training preserves query groups so products are learned and evaluated relative to other candidates for the same query.


## Learning-to-rank results

| Model         |    NDCG@10 |  Recall@10 |        MRR |
| ------------- | ---------: | ---------: | ---------: |
| Random        |     0.7611 |     0.5684 |     0.8757 |
| BM25          |     0.8124 |     0.5906 |     0.8988 |
| TF-IDF        |     0.8173 |     0.5922 |     0.9060 |
| **XGBRanker** | **0.8236** | **0.5929** | **0.9089** |

Compared with TF-IDF:

```text
Mean ΔNDCG@10:    +0.006329
Median ΔNDCG@10:  +0.000854

Wins:              1,580
Losses:            1,226
Ties:                328
```

A paired query-level bootstrap with 2,000 replicates produced:

```text
95% CI for mean ΔNDCG@10:
[0.003667, 0.009108]
```

The result supports a **small but consistent validation improvement** over the strongest lexical baseline.


# What created the improvement?

Feature importance alone does not tell us whether a feature provides unique ranking information, so the project also performs controlled feature ablations.

### Gain importance

The strongest fitted-tree signals were:

| Rank | Feature               | Normalized gain |
| ---: | --------------------- | --------------: |
|    1 | TF-IDF score          |           0.336 |
|    2 | Query token coverage  |           0.216 |
|    3 | Title token coverage  |           0.111 |
|    4 | Query bullet coverage |           0.078 |
|    5 | Brand match           |           0.051 |

TF-IDF was heavily used by the fitted trees, but ablation showed substantial redundancy among lexical signals.


## Ablation results

| Variant                   |     NDCG@10 | Δ vs full |
| ------------------------- | ----------: | --------: |
| Full model                | **0.82362** |         — |
| BM25 + TF-IDF only        |     0.81573 |  -0.00789 |
| Without TF-IDF            |     0.82338 |  -0.00025 |
| Without BM25              |     0.82378 |  +0.00016 |
| Without overlap/coverage  |     0.82128 |  -0.00235 |
| Without structured/length |     0.82253 |  -0.00109 |

The most important result is that the learning-to-rank improvement did **not** come from simply learning a combination of BM25 and TF-IDF.

The lexical-score-only ranker performed slightly below TF-IDF itself.

Explicit query-product overlap and coverage features provided meaningful additional ranking information.

The ablation analysis also illustrates why tree feature importance should not be interpreted as unique or causal importance.


# Failure analysis

Aggregate metrics do not explain why a ranking model succeeds or fails.

The project therefore preserves per-query comparisons and candidate-level examples for queries where XGBRanker strongly improves or regresses relative to TF-IDF.

Observed limitations include:

* lexical overlap rewarding the wrong product intent;
* vocabulary mismatch;
* typo sensitivity;
* limited handling of negation;
* difficulty representing semantic similarity when query and product wording differ.

These failures motivate the next modeling question:

> Can semantic relevance provide information beyond the current lexical learning-to-rank model?


# Current pipeline

```text
Amazon ESCI candidate sets
            ↓
Data validation and query-level split
            ↓
Random / BM25 / TF-IDF baselines
            ↓
Candidate-level ranking features
            ↓
Query-grouped XGBRanker
            ↓
NDCG@10 / Recall@10 / MRR
            ↓
Paired query-level bootstrap
            ↓
Feature importance
            ↓
Ablation and failure analysis
```


# Reproduce

Create the environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

After placing the official ESCI parquet files in `data/raw/`:

```bash
python -m scripts.prepare_data
python -m scripts.run_baseline
python -m scripts.build_features
python -m scripts.train_ranker
python -m scripts.evaluate_ranker
python -m scripts.analyze_ranker
```

Run tests:

```bash
python -m pytest -q
```

Current test suite:

```text
59 passed
```

Analysis notebooks:

```text
notebooks/01_data_and_baseline_analysis.ipynb
notebooks/02_learning_to_rank_analysis.ipynb
```


# Repository structure

```text
src/
├── data/          data loading, validation, merging, query splitting
├── retrieval/     text processing, Random, TF-IDF, BM25
├── features/      candidate-level ranking features
├── ranking/       query grouping and XGBRanker logic
└── evaluation/    ranking metrics, comparison, bootstrap, analysis

scripts/
├── prepare_data.py
├── run_baseline.py
├── build_features.py
├── train_ranker.py
├── evaluate_ranker.py
└── analyze_ranker.py

notebooks/
├── 01_data_and_baseline_analysis.ipynb
└── 02_learning_to_rank_analysis.ipynb

artifacts/
├── features/
├── models/
├── metrics/
└── examples/

tests/
```

Reusable logic lives in `src/`. Scripts orchestrate reproducible workflows, while notebooks present and interpret saved results.


# Current limitations

The project currently demonstrates **candidate re-ranking**, not a complete search engine.

It does not yet include:

* full-catalog retrieval;
* dense or vector retrieval;
* semantic embedding features;
* personalization;
* recommendation systems;
* online experiments;
* production serving.

The current results are offline results from the project-validation partition.

The official ESCI test holdout remains untouched during model development.


# Next direction

The current XGBRanker establishes a stronger benchmark than the lexical baselines.

The next question is:

> **Does semantic relevance add incremental ranking value beyond the lexical learning-to-rank model?**

The existing XGBRanker validation NDCG@10 of **0.8236** becomes the benchmark that future semantic modeling must justify itself against.


## Citation

```text
Reddy, C. K., Màrquez, L., Valero, F., Rao, N., Zaragoza, H.,
Bandyopadhyay, S., Biswas, A., Xing, A., and Subbian, K. (2022).
Shopping Queries Dataset: A Large-Scale ESCI Benchmark for Improving Product Search.
```
