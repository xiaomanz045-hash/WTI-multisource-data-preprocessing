# Multi-source data and feature preprocessing for WTI interval forecasting

This repository contains the data and preprocessing materials used to construct the multi-source feature matrix in the associated manuscript.

## Public scope

The repository supports reproduction of the **data-construction and feature-preprocessing stage**. It contains:

- source data files used in the study;
- a processed, time-aligned feature matrix;
- a Python script and Jupyter notebook for temporal alignment, Spearman screening, PCA, PLS, and daily aggregation of BERT-derived sentiment scores; and
- variable definitions, provenance notes, and file checksums.

The repository does **not** contain the forecasting-model implementation, CEEMDAN decomposition, graph-node construction, ELM/GRU/KAN-LSTM training, combination forecasting, or forecasting-result files. The forecasting design, data split, hyperparameters, repeated-run settings, and evaluation measures are described in the manuscript.

## Repository structure

```text
data/raw/                         source data used by preprocessing
data/processed/                   final eight-column model-input matrix
notebooks/feature_preprocessing.ipynb
preprocessing/feature_preprocessing.py
data_dictionary.md
MANIFEST.sha256
requirements.txt
```

## Processed matrix

`data/processed/WTI_interval_feature_matrix.xlsx` contains 1,469 daily observations from 2 January 2020 to 15 September 2025 and the following columns:

`Date`, `UB`, `LB`, `BI1`, `BI2`, `NH`, `SD1`, and `SD2`.

It contains no similar-node variables, decomposed components, model predictions, or evaluation results.

## Reproduce the preprocessing stage

Python 3.10 is recommended. From the repository root, run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python preprocessing\feature_preprocessing.py
```

On macOS or Linux, replace the activation command with:

```bash
source .venv/bin/activate
```

The generated files are written to `preprocessing_outputs/`. The key output is `09_final_multisource_feature_matrix.xlsx`.

Alternatively, open `notebooks/feature_preprocessing.ipynb` from the repository root and run all cells in order.

## Reproducibility check

The public preprocessing workflow was tested against the deposited feature matrix. It selected the same 11 Baidu Index variables and produced a matrix of 1,469 rows and 8 columns. The reproduced values are identical for `UB`, `LB`, `NH`, `SD1`, and `SD2`; the maximum absolute numerical differences for `BI1` and `BI2` are below `3.1e-11`, reflecting floating-point precision only.

## Data provenance and reuse

The original providers, source websites, retrieval periods, and collection procedures are reported in the manuscript. The files in `data/raw/` retain the observations used in the study. `news_text_with_sentiment_scores.xlsx` contains cleaned news text and the BERT-derived sentiment score used in the aggregation step; the repository does not retrain BERT.

No separate redistribution licence for third-party source data is asserted by this repository. Users should consult the original data providers' terms before reuse.

## Data and code availability wording

The source-data files, preprocessing code, and processed time-aligned feature matrix used in this study are available in this repository. The materials document temporal alignment, Spearman correlation analysis, feature screening, PCA, PLS-based dimension reduction, and daily aggregation of BERT-derived sentiment scores.

