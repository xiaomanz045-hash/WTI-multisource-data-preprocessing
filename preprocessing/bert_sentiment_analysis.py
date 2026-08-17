"""Generate article-level financial-news sentiment scores with Chinese BERT.

Model source:
https://huggingface.co/hw2942/bert-base-chinese-finetuning-financial-news-sentiment

The model predicts Negative, Neutral, and Positive. Following the manuscript
experiment, these classes are mapped to -1, 0, and 1, respectively.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


MODEL_ID = "hw2942/bert-base-chinese-finetuning-financial-news-sentiment"
MODEL_REVISION = "596188a9c884118e13984140a8b568a2252e01c2"
LABEL_TO_SCORE = {0: -1, 1: 0, 2: 1}
LABEL_TO_NAME = {0: "Negative", 1: "Neutral", 2: "Positive"}


def find_column(frame: pd.DataFrame, aliases: Sequence[str]) -> str:
    normalized = {str(column).strip().lower(): str(column) for column in frame.columns}
    for alias in aliases:
        match = normalized.get(alias.strip().lower())
        if match is not None:
            return match
    raise KeyError(f"Expected one of {list(aliases)}; received {list(frame.columns)}")


def load_model(model_path: str | Path | None = None, device: str | None = None):
    """Load the pinned Hugging Face revision or a user-supplied local copy."""
    source = str(model_path) if model_path is not None else MODEL_ID
    revision = None if model_path is not None else MODEL_REVISION
    tokenizer = AutoTokenizer.from_pretrained(source, revision=revision)
    model = AutoModelForSequenceClassification.from_pretrained(source, revision=revision)
    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(selected_device)
    model.eval()
    return tokenizer, model, selected_device


def predict_sentiment(
    texts: Sequence[str],
    tokenizer,
    model,
    device: str,
    batch_size: int = 32,
    max_length: int = 512,
) -> tuple[list[int], list[str], list[int]]:
    class_ids: list[int] = []
    for start in range(0, len(texts), batch_size):
        batch = [str(value) for value in texts[start : start + batch_size]]
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        encoded = {name: tensor.to(device) for name, tensor in encoded.items()}
        with torch.no_grad():
            logits = model(**encoded).logits
        class_ids.extend(torch.argmax(logits, dim=1).cpu().tolist())

    labels = [LABEL_TO_NAME[index] for index in class_ids]
    scores = [LABEL_TO_SCORE[index] for index in class_ids]
    return class_ids, labels, scores


def score_news_file(
    input_file: Path,
    output_file: Path,
    model_path: str | Path | None = None,
    batch_size: int = 32,
    device: str | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    frame = pd.read_excel(input_file)
    date_column = find_column(frame, ["Date", "日期", "时间"])
    text_column = find_column(
        frame,
        ["Cleaned_Text", "cleaned text", "新闻文本", "新闻标题", "title", "text"],
    )
    selected = frame[[date_column, text_column]].copy()
    selected.columns = ["Date", "Cleaned_Text"]
    selected["Date"] = pd.to_datetime(selected["Date"])
    selected["Cleaned_Text"] = selected["Cleaned_Text"].fillna("").astype(str)
    if limit is not None:
        selected = selected.iloc[:limit].copy()

    tokenizer, model, selected_device = load_model(model_path, device)
    class_ids, labels, scores = predict_sentiment(
        selected["Cleaned_Text"].tolist(),
        tokenizer,
        model,
        selected_device,
        batch_size=batch_size,
    )
    selected["Sentiment_Class_ID"] = class_ids
    selected["Sentiment_Label"] = labels
    selected["Sentiment_Score"] = scores
    output_file.parent.mkdir(parents=True, exist_ok=True)
    selected.to_excel(output_file, index=False)
    return selected


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=root / "data" / "raw" / "news_text_with_sentiment_scores.xlsx",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "sentiment_outputs" / "news_text_scored_by_bert.xlsx",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Optional local model directory. If omitted, the pinned Hugging Face revision is downloaded.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for a smoke test.")
    args = parser.parse_args()

    result = score_news_file(
        args.input,
        args.output,
        model_path=args.model_path,
        batch_size=args.batch_size,
        device=args.device,
        limit=args.limit,
    )
    print(f"Saved {len(result)} scored news records to {args.output}")


if __name__ == "__main__":
    main()

