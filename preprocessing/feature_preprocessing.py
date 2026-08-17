# -*- coding: utf-8 -*-
"""Reproduce the feature-selection and dimension-reduction stage of the paper.

Expected raw files (Chinese or English aliases are supported):
1. WTI原油期货历史数据.xlsx / wti.xlsx
2. 百度指数原始数据.xlsx / baidu_raw.xlsx
3. 新闻文本.xlsx / news_text.xlsx
4. 处理后GPR.xlsx / gpr.xlsx
5. 天然气期货历史数据.xlsx / gas.xlsx
6. 黄金期货历史数据.xlsx / gold.xlsx
7. 美元指数历史数据.xlsx / dollar.xlsx

The script reproduces the original experiment as follows:
- Baidu Index: Spearman screening (|rho| >= 0.4 for either bound), followed by
  unstandardized two-component PCA.
- Structured variables: Spearman correlations, followed by two-component PLS
  using the upper bound as the response and scale=True.
- News sentiment: daily mean aggregation and forward filling on WTI dates.

Outputs are written to ``preprocessing_outputs``.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA


# %% 1. Configuration
SCRIPT_DIR = (
    Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd().resolve()
)
DEFAULT_BASE_DIR = SCRIPT_DIR.parent if SCRIPT_DIR.name == "preprocessing" else SCRIPT_DIR
BASE_DIR = Path(os.environ.get("FEATURE_DATA_DIR", DEFAULT_BASE_DIR)).resolve()
OUTPUT_DIR = Path(
    os.environ.get("FEATURE_OUTPUT_DIR", BASE_DIR / "preprocessing_outputs")
).resolve()

CORRELATION_THRESHOLD = 0.40
BAIDU_PCA_COMPONENTS = 2
STRUCTURED_PLS_COMPONENTS = 2

FILE_ALIASES = {
    "wti": ["WTI原油期货历史数据.xlsx", "wti.xlsx", "wti_raw.xlsx"],
    "baidu": ["百度指数原始数据.xlsx", "baidu_raw.xlsx", "baidu_index_raw_data.xlsx"],
    "news": ["新闻文本.xlsx", "news_text.xlsx", "news_text_with_sentiment_scores.xlsx"],
    "gpr": ["处理后GPR.xlsx", "gpr.xlsx", "geopolitical_risk_index.xlsx"],
    "gas": ["天然气期货历史数据.xlsx", "gas.xlsx", "natural_gas_futures.xlsx"],
    "gold": ["黄金期货历史数据.xlsx", "gold.xlsx", "gold_futures.xlsx"],
    "dollar": ["美元指数历史数据.xlsx", "dollar.xlsx", "us_dollar_index.xlsx"],
}


# %% 2. General utilities
def find_input_file(base_dir: Path, aliases: Sequence[str]) -> Path:
    """Find one input file, preferring a file placed beside this script."""
    for name in aliases:
        direct = base_dir / name
        if direct.exists():
            return direct

    matches: list[Path] = []
    for name in aliases:
        matches.extend(base_dir.rglob(name))
    matches = sorted(set(matches), key=lambda p: (len(p.parts), str(p)))

    if not matches:
        raise FileNotFoundError(
            f"Missing input file. Expected one of: {', '.join(aliases)}"
        )
    if len(matches) > 1:
        joined = "\n  - ".join(str(p) for p in matches)
        raise RuntimeError(
            "Multiple matching input files were found. Place the intended file "
            f"beside this script or remove duplicates:\n  - {joined}"
        )
    return matches[0]


def read_excel(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def find_column(
    df: pd.DataFrame,
    aliases: Iterable[str],
    *,
    fallback_index: int | None = None,
) -> str:
    normalized = {str(c).strip().lower(): str(c) for c in df.columns}
    for alias in aliases:
        hit = normalized.get(alias.strip().lower())
        if hit is not None:
            return hit
    if fallback_index is not None:
        return str(df.columns[fallback_index])
    raise KeyError(
        f"None of the expected columns {list(aliases)} were found. "
        f"Actual columns: {list(df.columns)}"
    )


def to_daily_datetime(series: pd.Series) -> pd.Series:
    """Parse Excel dates, YYYYMMDD integers, or normal date strings."""
    text = series.astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
    ymd_mask = text.str.fullmatch(r"\d{8}")
    parsed = pd.to_datetime(series, errors="coerce")
    if ymd_mask.any():
        parsed.loc[ymd_mask] = pd.to_datetime(
            text.loc[ymd_mask], format="%Y%m%d", errors="coerce"
        )
    if parsed.isna().any():
        bad = series.loc[parsed.isna()].head(5).tolist()
        raise ValueError(f"Unparseable dates were found, e.g. {bad}")
    return parsed.dt.normalize()


def ensure_numeric(df: pd.DataFrame, columns: Sequence[str], label: str) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    missing = out[list(columns)].isna().sum()
    if int(missing.sum()) > 0:
        details = missing[missing > 0].to_dict()
        raise ValueError(f"{label} contains missing/non-numeric values: {details}")
    return out


def spearman_table(
    df: pd.DataFrame,
    feature_columns: Sequence[str],
    upper_column: str = "UB",
    lower_column: str = "LB",
) -> pd.DataFrame:
    rows = []
    for feature in feature_columns:
        work = df[[feature, upper_column, lower_column]].dropna()
        rho_ub, p_ub = spearmanr(work[feature], work[upper_column])
        rho_lb, p_lb = spearmanr(work[feature], work[lower_column])
        rows.append(
            {
                "Feature": feature,
                "Upper_bound_correlation": float(rho_ub),
                "Upper_bound_p_value": float(p_ub),
                "Lower_bound_correlation": float(rho_lb),
                "Lower_bound_p_value": float(p_lb),
                "Max_absolute_correlation": float(max(abs(rho_ub), abs(rho_lb))),
                "Observations": int(len(work)),
            }
        )
    return pd.DataFrame(rows)


# %% 3. Load the WTI interval target
def load_wti(path: Path) -> pd.DataFrame:
    raw = read_excel(path)
    date_col = find_column(raw, ["时间", "日期", "date"], fallback_index=0)
    upper_col = find_column(raw, ["高", "ub", "upper bound", "upper_bound"])
    lower_col = find_column(raw, ["低", "lb", "lower bound", "lower_bound"])

    out = raw[[date_col, upper_col, lower_col]].copy()
    out.columns = ["Date", "UB", "LB"]
    out["Date"] = to_daily_datetime(out["Date"])
    out = ensure_numeric(out, ["UB", "LB"], "WTI interval data")
    out = out.drop_duplicates("Date", keep="last").sort_values("Date").reset_index(drop=True)
    return out


# %% 4. Baidu Index: alignment, Spearman screening, and PCA
def load_baidu(path: Path) -> pd.DataFrame:
    raw = read_excel(path)

    # Long format: one row per date-keyword pair.
    long_date = next((c for c in ["日期", "时间", "Date"] if c in raw.columns), None)
    keyword_col = next((c for c in ["关键词", "Keyword", "keyword"] if c in raw.columns), None)
    value_col = next(
        (c for c in ["移动端", "指数", "Index", "value"] if c in raw.columns),
        None,
    )
    if long_date and keyword_col and value_col:
        raw[long_date] = to_daily_datetime(raw[long_date])
        out = raw.pivot_table(
            index=long_date,
            columns=keyword_col,
            values=value_col,
            aggfunc="sum",
            fill_value=0,
        ).reset_index()
        out.columns.name = None
        out = out.rename(columns={long_date: "Date"})
    else:
        # Wide format: one date column followed by one column per keyword.
        date_col = find_column(raw, ["时间", "日期", "date"], fallback_index=0)
        out = raw.rename(columns={date_col: "Date"}).copy()
        out["Date"] = to_daily_datetime(out["Date"])

    feature_columns = [c for c in out.columns if c != "Date"]
    out = ensure_numeric(out, feature_columns, "Baidu Index data")
    out = out.drop_duplicates("Date", keep="last").sort_values("Date").reset_index(drop=True)
    return out


def process_baidu(wti: pd.DataFrame, baidu: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    aligned = wti.merge(baidu, on="Date", how="inner", validate="one_to_one")
    candidate_features = [c for c in aligned.columns if c not in {"Date", "UB", "LB"}]

    correlation = spearman_table(aligned, candidate_features)
    selected_set = set(
        correlation.loc[
            correlation["Max_absolute_correlation"] >= CORRELATION_THRESHOLD,
            "Feature",
        ]
    )
    # Preserve the original column order to reproduce the published experiment.
    selected_features = [c for c in candidate_features if c in selected_set]
    if len(selected_features) < BAIDU_PCA_COMPONENTS:
        raise ValueError(
            f"Only {len(selected_features)} Baidu features passed the threshold; "
            f"at least {BAIDU_PCA_COMPONENTS} are required."
        )

    selected = aligned[["Date", "UB", "LB", *selected_features]].copy()

    # No standardization is applied here because this reproduces the original
    # BI1 and BI2 values exactly.
    pca_full = PCA(svd_solver="full")
    pca_full.fit(selected[selected_features])
    pca = PCA(n_components=BAIDU_PCA_COMPONENTS, svd_solver="full")
    scores = pca.fit_transform(selected[selected_features])

    reduced = selected[["Date", "UB", "LB"]].copy()
    for i in range(BAIDU_PCA_COMPONENTS):
        reduced[f"BI{i + 1}"] = scores[:, i]

    metadata = {
        "correlation_threshold": CORRELATION_THRESHOLD,
        "candidate_feature_count": len(candidate_features),
        "selected_feature_count": len(selected_features),
        "selected_features": selected_features,
        "pca_components_retained": BAIDU_PCA_COMPONENTS,
        "pca_explained_variance_ratio_retained": pca.explained_variance_ratio_.tolist(),
        "pca_cumulative_variance_retained": float(pca.explained_variance_ratio_.sum()),
        "pca_full_cumulative_variance": np.cumsum(pca_full.explained_variance_ratio_).tolist(),
        "standardized_before_pca": False,
    }
    return correlation, selected, reduced, metadata


# %% 5. News sentiment: daily aggregation and alignment
def process_sentiment(wti: pd.DataFrame, path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = read_excel(path)
    date_col = find_column(raw, ["时间", "日期", "date"], fallback_index=0)
    score_col = find_column(raw, ["Sentiment_Score", "sentiment score", "情感分数"])
    text_col = next(
        (c for c in ["Cleaned_Text", "cleaned_text", "新闻标题", "文本"] if c in raw.columns),
        None,
    )

    raw[date_col] = to_daily_datetime(raw[date_col])
    raw[score_col] = pd.to_numeric(raw[score_col], errors="coerce")
    if raw[score_col].isna().any():
        raise ValueError("News data contain missing/non-numeric sentiment scores.")

    aggregation = {score_col: "mean"}
    if text_col is not None:
        aggregation[text_col] = lambda values: " ".join(values.dropna().astype(str))
    daily = raw.groupby(date_col, as_index=False).agg(aggregation)
    rename = {date_col: "Date", score_col: "NH"}
    if text_col is not None:
        rename[text_col] = "Combined_Text"
    daily = daily.rename(columns=rename).sort_values("Date").reset_index(drop=True)

    aligned = wti[["Date", "UB", "LB"]].merge(
        daily[["Date", "NH"]], on="Date", how="left", validate="one_to_one"
    )
    aligned["NH"] = aligned["NH"].ffill()
    return daily, aligned


# %% 6. Structured variables: alignment, Spearman correlations, and PLS
def load_single_feature(
    path: Path,
    output_name: str,
    feature_aliases: Sequence[str],
) -> pd.DataFrame:
    raw = read_excel(path)
    date_col = find_column(raw, ["时间", "日期", "date"], fallback_index=0)
    feature_col = find_column(raw, feature_aliases)
    out = raw[[date_col, feature_col]].copy()
    out.columns = ["Date", output_name]
    out["Date"] = to_daily_datetime(out["Date"])
    out = ensure_numeric(out, [output_name], f"{output_name} data")
    return out.drop_duplicates("Date", keep="last").sort_values("Date").reset_index(drop=True)


def process_structured(
    wti: pd.DataFrame,
    gpr_path: Path,
    gas_path: Path,
    gold_path: Path,
    dollar_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    gpr = load_single_feature(
        gpr_path,
        "GPR",
        ["GPRD_MA30", "GPRD", "GPR", "geopolitical risk"],
    )
    gas = load_single_feature(
        gas_path,
        "Natural_Gas",
        ["天然气期货", "natural gas", "natural_gas"],
    )
    gold = load_single_feature(
        gold_path,
        "Gold",
        ["黄金期货", "gold futures", "gold"],
    )
    dollar = load_single_feature(
        dollar_path,
        "Dollar_Index",
        ["美元指数", "dollar index", "dollar_index"],
    )

    aligned = wti.copy()
    for feature_df in [gpr, gas, gold, dollar]:
        aligned = aligned.merge(feature_df, on="Date", how="inner", validate="one_to_one")

    feature_columns = ["GPR", "Natural_Gas", "Gold", "Dollar_Index"]
    correlation = spearman_table(aligned, feature_columns)

    # The original SD1 and SD2 values were produced by PLSRegression with the
    # upper interval bound as y and scale=True. This is intentionally retained
    # so the cleaned code reproduces the original feature matrix exactly.
    pls = PLSRegression(n_components=STRUCTURED_PLS_COMPONENTS, scale=True)
    scores = pls.fit(aligned[feature_columns], aligned["UB"]).transform(
        aligned[feature_columns]
    )

    reduced = aligned[["Date", "UB", "LB"]].copy()
    for i in range(STRUCTURED_PLS_COMPONENTS):
        reduced[f"SD{i + 1}"] = scores[:, i]

    metadata = {
        "structured_features": feature_columns,
        "dimension_reduction_method": "PLSRegression",
        "pls_components": STRUCTURED_PLS_COMPONENTS,
        "pls_response": "WTI upper bound (UB)",
        "pls_scale": True,
    }
    return correlation, aligned, reduced, metadata


# %% 7. Combine the final feature matrix
def combine_features(
    structured_reduced: pd.DataFrame,
    baidu_reduced: pd.DataFrame,
    sentiment_aligned: pd.DataFrame,
) -> pd.DataFrame:
    final = structured_reduced.merge(
        baidu_reduced[["Date", "BI1", "BI2"]],
        on="Date",
        how="inner",
        validate="one_to_one",
    )
    final = final.merge(
        sentiment_aligned[["Date", "NH"]],
        on="Date",
        how="inner",
        validate="one_to_one",
    )
    final = final[["Date", "UB", "LB", "BI1", "BI2", "NH", "SD1", "SD2"]]
    if final.isna().any().any():
        missing = final.isna().sum()
        raise ValueError(f"The final matrix contains missing values: {missing[missing > 0].to_dict()}")
    return final.sort_values("Date").reset_index(drop=True)


def save_outputs(
    baidu_correlation: pd.DataFrame,
    baidu_selected: pd.DataFrame,
    baidu_reduced: pd.DataFrame,
    sentiment_daily: pd.DataFrame,
    sentiment_aligned: pd.DataFrame,
    structured_correlation: pd.DataFrame,
    structured_aligned: pd.DataFrame,
    structured_reduced: pd.DataFrame,
    final_matrix: pd.DataFrame,
    metadata: dict,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    baidu_correlation.to_excel(OUTPUT_DIR / "01_baidu_spearman_correlations.xlsx", index=False)
    baidu_selected.to_excel(OUTPUT_DIR / "02_baidu_selected_features.xlsx", index=False)
    baidu_reduced.to_excel(OUTPUT_DIR / "03_baidu_pca_2d.xlsx", index=False)
    sentiment_daily.to_excel(OUTPUT_DIR / "04_sentiment_daily.xlsx", index=False)
    sentiment_aligned.to_excel(OUTPUT_DIR / "05_sentiment_aligned.xlsx", index=False)
    structured_correlation.to_excel(
        OUTPUT_DIR / "06_structured_spearman_correlations.xlsx", index=False
    )
    structured_aligned.to_excel(OUTPUT_DIR / "07_structured_aligned.xlsx", index=False)
    structured_reduced.to_excel(OUTPUT_DIR / "08_structured_pls_2d.xlsx", index=False)
    final_matrix.to_excel(OUTPUT_DIR / "09_final_multisource_feature_matrix.xlsx", index=False)

    with (OUTPUT_DIR / "preprocessing_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


# %% 8. Main program
def main() -> pd.DataFrame:
    paths = {key: find_input_file(BASE_DIR, aliases) for key, aliases in FILE_ALIASES.items()}

    print("Input files:")
    for key, path in paths.items():
        print(f"  {key:>7}: {path}")

    wti = load_wti(paths["wti"])
    baidu = load_baidu(paths["baidu"])
    baidu_corr, baidu_selected, baidu_reduced, baidu_meta = process_baidu(wti, baidu)

    sentiment_daily, sentiment_aligned = process_sentiment(wti, paths["news"])

    structured_corr, structured_aligned, structured_reduced, structured_meta = process_structured(
        wti,
        paths["gpr"],
        paths["gas"],
        paths["gold"],
        paths["dollar"],
    )

    final_matrix = combine_features(structured_reduced, baidu_reduced, sentiment_aligned)
    metadata = {
        "input_files": {key: str(path) for key, path in paths.items()},
        "wti_observations": len(wti),
        "final_observations": len(final_matrix),
        "baidu": baidu_meta,
        "structured": structured_meta,
    }
    save_outputs(
        baidu_corr,
        baidu_selected,
        baidu_reduced,
        sentiment_daily,
        sentiment_aligned,
        structured_corr,
        structured_aligned,
        structured_reduced,
        final_matrix,
        metadata,
    )

    print("\nCompleted successfully.")
    print(f"Selected Baidu features ({len(baidu_meta['selected_features'])}):")
    for feature in baidu_meta["selected_features"]:
        print(f"  - {feature}")
    print(
        "Cumulative variance explained by BI1 and BI2: "
        f"{baidu_meta['pca_cumulative_variance_retained']:.6%}"
    )
    print(f"Final feature matrix: {final_matrix.shape[0]} rows x {final_matrix.shape[1]} columns")
    print(f"Outputs: {OUTPUT_DIR}")
    return final_matrix


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise
