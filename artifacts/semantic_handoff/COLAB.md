# Colab semantic encode

Google Colab is used only for the GPU-intensive query/title embedding step.

The semantic inputs are prepared locally. Colab reads those frozen development inputs, generates `semantic_similarity`, and returns the resulting artifacts to the local repository.

Do not upload:

* `data/raw/`
* official ESCI test files
* `official_test_holdout.parquet`
* existing ranker artifacts

## 1. Start a GPU runtime

Open a new Google Colab notebook.

Select:

```text
Runtime → Change runtime type → GPU
```

Any CUDA-capable GPU is sufficient.

## 2. Upload the handoff archive

Upload:

```text
search_ranking_semantic_handoff.zip
```

Then unzip it:

```python
!unzip -q search_ranking_semantic_handoff.zip -d search-ranking-platform
```

## 3. Install semantic dependencies

```python
!pip install -q "sentence-transformers>=3.0" pandas pyarrow numpy
```

## 4. Confirm GPU access

```python
!python -c "import torch; print('cuda', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

Expected output should show:

```text
cuda True
```

followed by the available GPU name.

## 5. Locate the project directory

Because notebook shell commands such as:

```python
!cd search-ranking-platform
```

do not persist between Colab cells, locate the semantic build script first:

```python
!find /content -name "build_semantic_features.py"
```

For the current handoff archive, the working directory is:

```python
%cd /content/search-ranking-platform/search_ranking_semantic_handoff/
```

Verify the expected project files:

```python
from pathlib import Path

ROOT = Path.cwd()
print(ROOT)

print((ROOT / "scripts" / "build_semantic_features.py").exists())
print((ROOT / "src").exists())
```

Both checks should return:

```text
True
```

## 6. Generate semantic features

Run:

```python
!python -m scripts.build_semantic_features
```

The script:

* loads the frozen train and validation semantic inputs;
* encodes unique query texts;
* encodes unique product-title texts;
* uses normalized MiniLM embeddings;
* computes query-title cosine similarity;
* maps `semantic_similarity` back to each candidate row.

The official ESCI test is not used.

## 7. Confirm generated artifacts

Check recently created files:

```python
!find artifacts -type f -mmin -10 -print
```

The required outputs are:

```text
artifacts/semantic/train_semantic.parquet
artifacts/semantic/validation_semantic.parquet
artifacts/semantic/semantic_summary.json
```

## 8. Inspect artifact sizes

```python
from pathlib import Path

paths = [
    Path("artifacts/semantic/train_semantic.parquet"),
    Path("artifacts/semantic/validation_semantic.parquet"),
    Path("artifacts/semantic/semantic_summary.json"),
]

for path in paths:
    print(f"{path}: {path.stat().st_size / 1024 / 1024:.2f} MB")
```

## 9. Perform a quick integrity check

```python
import pandas as pd
import json

train_sem = pd.read_parquet(
    "artifacts/semantic/train_semantic.parquet"
)

val_sem = pd.read_parquet(
    "artifacts/semantic/validation_semantic.parquet"
)

print("Train shape:", train_sem.shape)
print("Validation shape:", val_sem.shape)

print("\nColumns:")
print(train_sem.columns.tolist())

display(train_sem.head())
```

Expected row counts:

```text
Train:       356,615
Validation:   63,038
```

Expected columns:

```text
example_id
query_id
product_id
semantic_similarity
```

Check missing values:

```python
print("Train missing:")
print(train_sem.isna().sum())

print("\nValidation missing:")
print(val_sem.isna().sum())
```

There should be no missing semantic similarity values.

## 10. Package only the required outputs

Create a small flat archive containing only the generated semantic artifacts:

```python
!zip -j semantic_artifacts.zip \
    artifacts/semantic/train_semantic.parquet \
    artifacts/semantic/validation_semantic.parquet \
    artifacts/semantic/semantic_summary.json
```

Raw embedding matrices are not required.

## 11. Download the artifacts

```python
from google.colab import files

files.download("semantic_artifacts.zip")
```

Extract the downloaded files locally into:

```text
artifacts/semantic/
```

The local repository should then contain:

```text
artifacts/semantic/
├── train_semantic.parquet
├── validation_semantic.parquet
└── semantic_summary.json
```

The generated parquet files are local artifacts and do not need to be committed to Git.

`semantic_summary.json` may be committed as a lightweight record of the semantic encoding run.

## Safety boundary

The Colab job uses only the prepared train and validation semantic inputs.

Do not upload, inspect, encode, or score the official ESCI test or raw ESCI dataset in this workflow.
