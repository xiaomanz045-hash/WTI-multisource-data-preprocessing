"""Run feature construction through CEEMDAN and top-2 2V-GCN nodes.

This script deliberately stops before ELM, GRU, KAN-LSTM, and combination
forecasting. Use ``--mode frozen`` for the exact historical intermediate data,
or ``--mode recompute`` to rerun the stochastic CEEMDAN and 2V-GCN stages.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ceemdan_reconstruction import decompose_feature_matrix, reconstruct_frequencies
from two_view_gcn_similarity import calculate_similarity_nodes


def attach_nodes(frequency_file: Path, node_file: Path, output_file: Path) -> Path:
    frequency = pd.read_excel(frequency_file)
    nodes = pd.read_excel(node_file, sheet_name="Similar_Nodes")
    if len(frequency) != len(nodes):
        raise ValueError(f"Row mismatch: {frequency_file} and {node_file}")
    for column in ["Sim_1_UB", "Sim_1_LB", "Sim_2_UB", "Sim_2_LB"]:
        frequency[column] = nodes[column].to_numpy()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    frequency.to_excel(output_file, index=False)
    return output_file


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["frozen", "recompute"], default="frozen")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--feature-matrix",
        type=Path,
        default=root / "data" / "processed" / "WTI_interval_feature_matrix.xlsx",
    )
    parser.add_argument("--output-dir", type=Path, default=root / "feature_outputs")
    args = parser.parse_args()

    feature_matrix = args.feature_matrix
    if args.mode == "frozen":
        decomposition = root / "artifacts" / "frozen_ceemdan.xlsx"
        node_directory = root / "artifacts" / "frozen_similarity"
    else:
        decomposition = args.output_dir / "ceemdan_decomposition_results.xlsx"
        decompose_feature_matrix(feature_matrix, decomposition, seed=args.seed)
        node_directory = args.output_dir / "similarity"

    reconstructed = reconstruct_frequencies(
        feature_matrix,
        decomposition,
        args.output_dir / "reconstructed",
    )

    output_files = []
    for frequency in ("low", "medium", "high"):
        if args.mode == "recompute":
            node_file = node_directory / f"{frequency}_frequency_similar_nodes.xlsx"
            calculate_similarity_nodes(reconstructed[frequency], node_file, seed=args.seed)
        else:
            node_file = node_directory / f"{frequency}_frequency_similar_nodes.xlsx"
        output_files.append(
            attach_nodes(
                reconstructed[frequency],
                node_file,
                args.output_dir / "model_inputs" / f"{frequency}_frequency_with_similar_nodes.xlsx",
            )
        )

    print("Feature pipeline completed before forecasting-model training.")
    for output in output_files:
        print(output)


if __name__ == "__main__":
    main()
