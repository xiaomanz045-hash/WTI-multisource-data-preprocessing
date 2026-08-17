# Multi-source data preprocessing for WTI interval forecasting

This repository provides the source data, processed feature matrix, and preprocessing code used to construct the multi-source inputs for the associated manuscript.

## Public scope

The repository supports reproduction of:

1. temporal alignment of the source data;
2. Spearman correlation analysis and screening;
3. unstandardized two-component PCA of the selected Baidu Index variables;
4. two-component PLS reduction of the structured variables;
5. article-level financial-news sentiment classification with the cited Chinese BERT model;
6. daily sentiment aggregation, missing-value treatment, and feature integration; and
7. construction of the final model-ready feature matrix.

The repository covers only the multi-source data-preparation stages listed above. Subsequent feature-extraction and forecasting stages are outside its scope.

## Repository structure

```text
data/raw/                              source data
data/processed/                        aligned model-ready feature matrix
notebooks/feature_preprocessing.ipynb  feature preprocessing notebook
notebooks/bert_sentiment_analysis.ipynb
preprocessing/feature_preprocessing.py
preprocessing/bert_sentiment_analysis.py
data_dictionary.md
model_metadata.json
MANIFEST.sha256
requirements.txt
```

The BERT model weights are not redistributed. The code loads [`hw2942/bert-base-chinese-finetuning-financial-news-sentiment`](https://huggingface.co/hw2942/bert-base-chinese-finetuning-financial-news-sentiment) at pinned revision `596188a9c884118e13984140a8b568a2252e01c2`, or accepts a user-supplied local model directory. Model provenance and the class-to-score mapping are recorded in `model_metadata.json`.

## Processed matrix

`data/processed/WTI_interval_feature_matrix.xlsx` contains 1,469 daily observations from 2 January 2020 to 15 September 2025 and the variables `Date`, `UB`, `LB`, `BI1`, `BI2`, `NH`, `SD1`, and `SD2`. The observations are divided chronologically into training, validation, and test sets in a 7:1:2 ratio.

- `BI1` and `BI2` are the two PCA components extracted from the selected Baidu Index variables.
- `NH` is the daily BERT-derived news-sentiment score.
- `SD1` and `SD2` are the two PLS components extracted from the structured variables.

Researchers may use this matrix directly or reconstruct it from the supplied source files and preprocessing code.

## Installation

Python 3.10 is recommended. From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On macOS or Linux, use `source .venv/bin/activate`.

## Run the preprocessing

To reproduce article-level sentiment scores from the cleaned news text:

```powershell
python preprocessing\bert_sentiment_analysis.py
```

The model outputs `Negative`, `Neutral`, and `Positive`, which are mapped to `-1`, `0`, and `1`, respectively. Inputs are truncated to 512 tokens. Use `--model-path <local-model-directory>` to load a local model copy. The output is written to `sentiment_outputs/news_text_scored_by_bert.xlsx`.

To reconstruct the final multi-source feature matrix from the source files:

```powershell
python preprocessing\feature_preprocessing.py
```

The output is written to `preprocessing_outputs/09_final_multisource_feature_matrix.xlsx`. The same run also exports the correlation tables, selected Baidu Index variables, PCA scores, daily sentiment series, aligned structured variables, and PLS scores.

## Reproducibility check

The preprocessing workflow was executed successfully. It selected the same 11 Baidu Index variables and reproduced the feature matrix used in the manuscript. `UB`, `LB`, `NH`, `SD1`, and `SD2` matched exactly; the maximum absolute differences for `BI1` and `BI2` were below `3.1e-11`, reflecting floating-point precision only.

## Data provenance and reuse

Original providers, source websites, retrieval periods, and collection procedures are reported in the manuscript. `news_text_with_sentiment_scores.xlsx` contains the cleaned news text and article-level BERT-derived sentiment scores used for daily aggregation. The repository provides inference code for regenerating these scores but does not fine-tune or retrain the third-party BERT model.

No separate redistribution licence for third-party source data is asserted by this repository. Users should consult the original providers' terms before reuse.
