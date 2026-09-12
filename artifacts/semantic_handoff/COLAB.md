# Colab semantic encode

This archive contains only the source needed to encode query/title semantic similarity for the frozen development partitions.

Do **not** upload:

* `data/raw/`
* official ESCI test files
* `official_test_holdout.parquet`
* Week 1 ranker artifacts

## Steps

1. In Google Colab, open a new notebook.
2. Runtime → Change runtime type → Hardware accelerator → **GPU**. Any CUDA GPU is fine; a Tesla T4 is not required.
3. Upload `search_ranking_semantic_handoff.zip`.
4. Unzip it and enter the project directory:

```bash
unzip -q search_ranking_semantic_handoff.zip -d search-ranking-platform
cd search-ranking-platform
```

5. Install the packages required for encoding. Colab already provides PyTorch.

```bash
pip install -q "sentence-transformers>=3.0" pandas pyarrow numpy
```

6. Confirm CUDA if possible:

```bash
python -c "import torch; print('cuda', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

7. Encode unique query and product-title texts:

```bash
python -m scripts.build_semantic_features
```

8. Confirm these files exist:

```text
artifacts/semantic/train_semantic.parquet
artifacts/semantic/validation_semantic.parquet
artifacts/semantic/semantic_summary.json
```

9. Zip the semantic artifacts:

```bash
zip -r semantic_artifacts.zip artifacts/semantic
```

10. Download `semantic_artifacts.zip` and extract it to the same path in the local repository:

```text
artifacts/semantic/
```

Raw embedding caches under `artifacts/semantic/cache/` are optional and do not need to be copied back.

Do not upload or encode official-test or raw ESCI files.
