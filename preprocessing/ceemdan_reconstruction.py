"""CEEMDAN decomposition, sample entropy, and three-frequency reconstruction.

The numerical settings and historical IMF allocation are kept identical to the
executed manuscript experiment. The original CEEMDAN run did not save a random
seed, so a fresh decomposition follows the same method but need not reproduce
the frozen historical components bit for bit.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


VARIABLE_TO_SHEET = {
    "UB": "Upper bound",
    "LB": "Lower bound",
    "BI1": "BIPC1",
    "BI2": "BIPC2",
    "NH": "Sentiment Score",
    "SD1": "SDPLS1",
    "SD2": "SDPLS2",
}

# One-based IMF indices used in the original experiment.
IMF_GROUPS = {
    "UB": {"high": [1, 2, 3], "medium": [4, 5, 6], "low": [7, 8]},
    "LB": {"high": [1, 2, 3], "medium": [4, 5, 6], "low": [7, 8, 9]},
    "BI1": {"high": [1, 2, 3], "medium": [4, 5, 6, 7], "low": [8, 9]},
    "BI2": {"high": [1, 2, 3], "medium": [4, 5, 6, 7], "low": [8, 9]},
    "NH": {"high": [1, 2, 3], "medium": [4, 5, 6, 7], "low": [8, 9]},
    "SD1": {"high": [1, 2, 3], "medium": [4, 5, 6], "low": [7, 8]},
    "SD2": {"high": [1, 2, 3], "medium": [4, 5, 6], "low": [7]},
}


def _maximum_absolute_difference(a: Iterable, b: Iterable) -> float:
    x = pd.to_numeric(pd.Series(a), errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(pd.Series(b), errors="coerce").to_numpy(dtype=float)
    return float(np.nanmax(np.abs(x - y)))


def load_feature_matrix(path: Path) -> pd.DataFrame:
    frame = pd.read_excel(path)
    required = ["Date", "UB", "LB", "BI1", "BI2", "NH", "SD1", "SD2"]
    if list(frame.columns) != required:
        raise ValueError(f"Expected columns {required}; received {list(frame.columns)}")
    frame["Date"] = pd.to_datetime(frame["Date"])
    if frame.isna().any().any():
        raise ValueError(f"Input contains missing values: {frame.isna().sum().to_dict()}")
    if (frame["UB"] < frame["LB"]).any():
        raise ValueError("At least one observation has UB < LB.")
    return frame


def sample_entropy(series: np.ndarray, m: int = 2, r_ratio: float = 0.2) -> float:
    """Sample entropy with embedding dimension 2 and r=0.2*SD."""
    from scipy.spatial import cKDTree

    values = np.asarray(series, dtype=float)
    radius = r_ratio * np.std(values)

    def pair_count(dimension: int) -> int:
        windows = np.lib.stride_tricks.sliding_window_view(values, dimension)
        return len(cKDTree(windows).query_pairs(radius, p=np.inf))

    count_m = pair_count(m)
    count_m1 = pair_count(m + 1)
    return float(-np.log(count_m1 / count_m)) if count_m and count_m1 else float("inf")


def decompose_feature_matrix(
    feature_matrix: Path,
    output_workbook: Path,
    seed: int | None = None,
) -> Path:
    """Recompute CEEMDAN using Nstd=0.2, NR=100, and MaxIter=10."""
    from PyEMD import CEEMDAN

    raw = load_feature_matrix(feature_matrix)
    output_workbook.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_workbook, engine="openpyxl") as writer:
        for variable, sheet_name in VARIABLE_TO_SHEET.items():
            signal = raw[variable].to_numpy(dtype=float)
            ceemdan = CEEMDAN(Nstd=0.2, NR=100, MaxIter=10)
            if seed is not None:
                ceemdan.noise_seed(seed)
            imfs = np.asarray(ceemdan(signal))

            decomposition = pd.DataFrame({"Original_Signal": signal})
            entropy_rows = []
            for index, imf in enumerate(imfs, start=1):
                decomposition[f"IMF_{index}"] = imf
                entropy_rows.append(
                    {"IMF": f"IMF_{index}", "Sample_Entropy": sample_entropy(imf)}
                )

            decomposition.to_excel(writer, sheet_name=sheet_name, index=False)
            pd.DataFrame(entropy_rows).to_excel(
                writer,
                sheet_name=sheet_name,
                startrow=len(decomposition) + 2,
                index=False,
            )
    return output_workbook


def reconstruct_frequencies(
    feature_matrix: Path,
    decomposition_workbook: Path,
    output_directory: Path,
) -> dict[str, Path]:
    """Sum the historical IMF groups into low, medium, and high frequencies."""
    raw = load_feature_matrix(feature_matrix)
    output_directory.mkdir(parents=True, exist_ok=True)
    components: dict[str, np.ndarray] = {}

    for variable, sheet_name in VARIABLE_TO_SHEET.items():
        sheet = pd.read_excel(decomposition_workbook, sheet_name=sheet_name, nrows=len(raw))
        if _maximum_absolute_difference(sheet.iloc[:, 0], raw[variable]) > 1e-10:
            raise ValueError(f"Decomposition source signal does not match {variable}.")
        largest_index = max(max(group) for group in IMF_GROUPS[variable].values())
        components[variable] = sheet.iloc[:, 1 : largest_index + 1].to_numpy(dtype=float).T

    outputs: dict[str, Path] = {}
    for frequency in ("low", "medium", "high"):
        values: dict[str, object] = {"Date": raw["Date"]}
        for variable in VARIABLE_TO_SHEET:
            indices = np.asarray(IMF_GROUPS[variable][frequency], dtype=int) - 1
            values[variable] = components[variable][indices].sum(axis=0)
        output = output_directory / f"{frequency}_frequency.xlsx"
        pd.DataFrame(values).to_excel(output, index=False)
        outputs[frequency] = output
    return outputs


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=root / "data" / "processed" / "WTI_interval_feature_matrix.xlsx",
    )
    parser.add_argument(
        "--decomposition",
        type=Path,
        default=root / "artifacts" / "frozen_ceemdan.xlsx",
    )
    parser.add_argument("--output-dir", type=Path, default=root / "feature_outputs" / "reconstructed")
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.recompute:
        decompose_feature_matrix(args.input, args.decomposition, seed=args.seed)
    outputs = reconstruct_frequencies(args.input, args.decomposition, args.output_dir)
    for frequency, path in outputs.items():
        print(f"{frequency}: {path}")


if __name__ == "__main__":
    main()

