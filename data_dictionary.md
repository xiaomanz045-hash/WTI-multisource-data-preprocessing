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
| `news_text_with_sentiment_scores.xlsx` | Cleaned news text and article-level BERT sentiment scores | 10,216 | 2020-01-02 to 2025-09-11 |
| `geopolitical_risk_index.xlsx` | Geopolitical-risk series used in structured features | 1,499 | 2020-01-01 to 2025-09-15 |
| `natural_gas_futures.xlsx` | Daily natural-gas futures series | 1,501 | 2020-01-01 to 2025-09-16 |
| `gold_futures.xlsx` | Daily gold futures series | 1,474 | 2020-01-02 to 2025-09-17 |
| `us_dollar_index.xlsx` | Daily US Dollar Index series | 1,484 | 2020-01-01 to 2025-09-16 |

The manuscript provides the original provider names, source websites, and collection procedures. Column names in the raw workbooks are retained to preserve provenance.

