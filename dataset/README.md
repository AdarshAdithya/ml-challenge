# Dataset Directory

## Overview
This repository contains sample subsets of the Amazon ML Challenge 2026 dataset in [`dataset/sample/`](sample/) to allow repository cloning, schema inspection, unit testing, and code development without exceeding GitHub's 100MB file limit.

## Full Dataset Placement
When running training or generating final submissions, place the full extracted `.tsv` files in their respective folders:

### Training Set (`dataset/train/`)
- `dataset/train/train_source1.tsv` (~210 MB)
- `dataset/train/train_source2.tsv` (~489 MB)
- `dataset/train/train_source3.tsv` (~503 MB)
- `dataset/train/train_ground_truth.tsv` (~127 MB)

### Test Set (`dataset/test/`)
- `dataset/test/test_source1.tsv` (~175 MB)
- `dataset/test/test_source2.tsv` (~509 MB)
- `dataset/test/test_source3.tsv` (~506 MB)

> Note: All full `.tsv` files in `dataset/train/` and `dataset/test/` are ignored by `.gitignore` so they won't accidentally be pushed to GitHub.
