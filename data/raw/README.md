# Raw ESCI data (not committed)

This repository does **not** redistribute Amazon's Shopping Queries / ESCI dataset.

Place the official parquet files here:

```text
data/raw/shopping_queries_dataset_examples.parquet
data/raw/shopping_queries_dataset_products.parquet
```

## How to obtain the files

1. Read the official dataset page: https://github.com/amazon-science/esci-data
2. Install Git LFS, then clone the official repository:

```bash
git lfs install
git clone https://github.com/amazon-science/esci-data.git
```

3. Copy the parquet files from `esci-data/shopping_queries_dataset/` into this directory.

The products file is large (on the order of 1 GB). This project will not download it automatically.

Citation: Reddy et al., "Shopping Queries Dataset: A Large-Scale ESCI Benchmark for Improving Product Search," 2022.
