# Data dictionary

## Processed feature matrix

File: `data/processed/WTI_interval_feature_matrix.xlsx`

| Variable | Definition | Construction | Unit/scale |
|---|---|---|---|
| `Date` | Trading date | Common date retained after temporal alignment | `YYYY-MM-DD` |
| `UB` | Upper bound of the daily WTI price interval | Daily high WTI crude-oil futures price | US dollars per barrel |
| `LB` | Lower bound of the daily WTI price interval | Daily low WTI crude-oil futures price | US dollars per barrel |
| `BI1` | First Baidu Index component | First component from unstandardized two-component PCA after Spearman screening | PCA score |
| `BI2` | Second Baidu Index component | Second component from unstandardized two-component PCA after Spearman screening | PCA score |
| `NH` | Daily news sentiment feature | Daily mean of article-level BERT sentiment scores; missing trading dates are forward-filled | Sentiment score |
| `SD1` | First structured-data component | First two-component PLS score from GPR, natural-gas futures, gold futures, and the US Dollar Index, using `UB` as the response and `scale=True` | PLS score |
| `SD2` | Second structured-data component | Second PLS score constructed under the same specification | PLS score |

The matrix contains 1,469 observations from 2020-01-02 to 2025-09-15 and has no missing values.

## Source files

| File | Content | Rows | Coverage |
|---|---|---:|---|
| `wti_raw.xlsx` | Daily WTI futures prices and trading information | 1,500 | 2020-01-01 to 2025-09-16 |
| `baidu_index_raw_data.xlsx` | Daily search indices for 37 candidate keywords | 2,085 | 2020-01-01 to 2025-09-15 |
| `news_text_with_sentiment_scores.xlsx` | Cleaned news text and article-level BERT sentiment scores (`Negative=-1`, `Neutral=0`, `Positive=1`) | 10,216 | 2020-01-02 to 2025-09-11 |
| `geopolitical_risk_index.xlsx` | Geopolitical-risk series used in structured features | 1,499 | 2020-01-01 to 2025-09-15 |
| `natural_gas_futures.xlsx` | Daily natural-gas futures series | 1,501 | 2020-01-01 to 2025-09-16 |
| `gold_futures.xlsx` | Daily gold futures series | 1,474 | 2020-01-02 to 2025-09-17 |
| `us_dollar_index.xlsx` | Daily US Dollar Index series | 1,484 | 2020-01-01 to 2025-09-16 |

The manuscript provides the original provider names, source websites, and collection procedures. Column names in the raw workbooks are retained to preserve provenance.

## BERT sentiment model

Article-level news sentiment was generated with [`hw2942/bert-base-chinese-finetuning-financial-news-sentiment`](https://huggingface.co/hw2942/bert-base-chinese-finetuning-financial-news-sentiment), pinned to revision `596188a9c884118e13984140a8b568a2252e01c2`. The model configuration defines class 0 as `Negative`, class 1 as `Neutral`, and class 2 as `Positive`; the manuscript pipeline maps these classes to -1, 0, and 1. Inputs are padded and truncated to a maximum length of 512 tokens. `preprocessing/bert_sentiment_analysis.py` performs inference, while `feature_preprocessing.py` averages article-level scores by day and forward-fills missing trading dates when constructing `NH`.

## Frozen intermediate feature data

### CEEMDAN decomposition

`artifacts/frozen_ceemdan.xlsx` stores the historical decomposition actually used in the manuscript. Its worksheets map to variables as follows:

| Worksheet | Variable |
|---|---|
| `Upper bound` | `UB` |
| `Lower bound` | `LB` |
| `BIPC1` | `BI1` |
| `BIPC2` | `BI2` |
| `Sentiment Score` | `NH` |
| `SDPLS1` | `SD1` |
| `SDPLS2` | `SD2` |

Each sheet contains the original signal, its IMF components, and the associated sample-entropy information. The exact IMF allocation used for low-, medium-, and high-frequency reconstruction is declared in `preprocessing/ceemdan_reconstruction.py`.

### Time-ordered 2V-GCN nodes

The three workbooks in `artifacts/frozen_similarity/` contain the historical similarity outputs for the low-, medium-, and high-frequency target intervals. The `Similar_Nodes` worksheet contains:

| Variable | Definition |
|---|---|
| `TimePoint` | Zero-based index of the current observation |
| `Sim_1_UB`, `Sim_1_LB` | Upper and lower bounds of the most similar historical interval |
| `Index_1`, `Similarity_1` | Historical index and similarity score of the first node |
| `Sim_2_UB`, `Sim_2_LB` | Upper and lower bounds of the second-most-similar historical interval |
| `Index_2`, `Similarity_2` | Historical index and similarity score of the second node |

The `Similarity_Matrix` worksheet stores the full time-ordered interval similarity matrix used to select these nodes.
