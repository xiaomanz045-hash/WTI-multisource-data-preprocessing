# Multi-source feature construction for WTI interval forecasting

This repository contains the data and preprocessing materials used to construct the complete feature inputs for the associated manuscript.

## Public scope

The repository supports reproduction of:

1. temporal alignment of the source data;
2. Spearman correlation screening;
3. PCA of the selected Baidu Index variables;
4. PLS-based reduction of the structured variables;
5. article-level financial-news sentiment classification with the cited Chinese BERT model and daily score aggregation;
6. CEEMDAN decomposition of all seven input/target variables;
7. sample-entropy calculation and low/medium/high-frequency reconstruction; and
8. time-ordered top-2 interval-node construction using the two-view GCN procedure.

The repository does **not** contain the ELM, GRU, KAN-LSTM, or combination-forecasting implementations, trained forecasting models, prediction files, or forecasting evaluation results.

## Repository structure

```text
data/raw/                              source data
data/processed/                        aligned eight-column feature matrix
artifacts/frozen_ceemdan.xlsx          historical CEEMDAN output
artifacts/frozen_similarity/           historical 2V-GCN node outputs
notebooks/feature_preprocessing.ipynb  data preprocessing notebook
notebooks/bert_sentiment_analysis.ipynb
preprocessing/bert_sentiment_analysis.py
preprocessing/feature_preprocessing.py
preprocessing/ceemdan_reconstruction.py
preprocessing/two_view_gcn_similarity.py
preprocessing/run_feature_pipeline.py
data_dictionary.md
MANIFEST.sha256
requirements.txt
```

The BERT model weights are not redistributed in this repository. The code loads the official model [`hw2942/bert-base-chinese-finetuning-financial-news-sentiment`](https://huggingface.co/hw2942/bert-base-chinese-finetuning-financial-news-sentiment) at pinned revision `596188a9c884118e13984140a8b568a2252e01c2`, or accepts a user-supplied local model directory. Model provenance and the class-to-score mapping are recorded in `model_metadata.json`.

## Processed matrix

`data/processed/WTI_interval_feature_matrix.xlsx` contains 1,469 daily observations from 2 January 2020 to 15 September 2025. Its columns are `Date`, `UB`, `LB`, `BI1`, `BI2`, `NH`, `SD1`, and `SD2`.The data is chronologically divided into a training set, a validation set, and a testing set with a ratio of 7:1:2.

## Installation

Python 3.10 is recommended. From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On macOS or Linux, use `source .venv/bin/activate`.

## Run the feature pipeline

To reproduce article-level sentiment scores from cleaned news text, run:

```powershell
python preprocessing\bert_sentiment_analysis.py
```

The model outputs `Negative`, `Neutral`, and `Positive`, which are mapped to `-1`, `0`, and `1`, respectively. Text is truncated to 512 tokens. Use `--model-path <local-model-directory>` to run from a local copy instead of downloading the pinned Hugging Face revision. The output is written to `sentiment_outputs/news_text_scored_by_bert.xlsx`.

To rebuild the multi-source feature matrix from the raw files:

```powershell
python preprocessing\feature_preprocessing.py
```

The final output of this stage is `preprocessing_outputs/09_final_multisource_feature_matrix.xlsx`.

To reproduce the exact historical decomposition, reconstructed frequency inputs, and top-2 similarity-node inputs using the frozen stochastic outputs:

```powershell
python preprocessing\run_feature_pipeline.py --mode frozen
```

To rerun CEEMDAN and 2V-GCN from the processed feature matrix:

```powershell
python preprocessing\run_feature_pipeline.py --mode recompute
```

To connect a newly generated preprocessing matrix directly to the recomputation stage:

```powershell
python preprocessing\run_feature_pipeline.py --mode recompute --feature-matrix preprocessing_outputs\09_final_multisource_feature_matrix.xlsx
```

All generated frequency and similar-node inputs are written to `feature_outputs/`. The pipeline stops before forecasting-model training.

## Stochastic-stage clarification

The original CEEMDAN and 2V-GCN executions did not save random seeds. Therefore, `--mode frozen` is the route for reproducing the exact historical feature inputs used in the manuscript. `--mode recompute` preserves the original algorithms and hyperparameters but may produce numerically different stochastic realizations. An optional `--seed` argument is provided for new deterministic runs; it was not imposed on the historical experiment.

The retained settings include CEEMDAN `Nstd=0.2`, `NR=100`, and `MaxIter=10`; sample entropy with `m=2` and `r=0.2×SD`; two historical interval nodes; a visibility threshold of `0.5`; and the original three-layer GCN training settings.

## Reproducibility checks

The source-data preprocessing workflow was run successfully. It selected the same 11 Baidu Index variables and produced a 1,469-row, 8-column matrix. `UB`, `LB`, `NH`, `SD1`, and `SD2` matched exactly; the maximum absolute differences for `BI1` and `BI2` were below `3.1e-11`, reflecting floating-point precision only. The frozen feature pipeline was also executed to verify the three reconstructed frequency files and the attached top-2 historical interval-node variables.

## Data provenance and reuse

Original providers, source websites, retrieval periods, and collection procedures are reported in the manuscript. `news_text_with_sentiment_scores.xlsx` contains cleaned news text and the article-level BERT-derived sentiment scores used for daily aggregation. The repository provides inference code for regenerating these scores but does not fine-tune or retrain the third-party BERT model.

No separate redistribution licence for third-party source data is asserted by this repository. Users should consult the original providers' terms before reuse.
