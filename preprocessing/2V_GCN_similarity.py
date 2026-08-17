"""Compute time-ordered top-2 interval nodes using the original 2V-GCN design."""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from tqdm import tqdm


class ImprovedGCN(torch.nn.Module):
    def __init__(self, in_channels: int = 1, hidden: int = 32, out: int = 16, dropout: float = 0.3):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.conv3 = GCNConv(hidden, out)
        self.dropout = dropout

    def forward(self, features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        features = F.relu(self.conv1(features, edge_index))
        features = F.dropout(features, p=self.dropout, training=self.training)
        features = F.relu(self.conv2(features, edge_index))
        features = F.dropout(features, p=self.dropout, training=self.training)
        return self.conv3(features, edge_index)


def sparse_directed_visibility_graph(series: np.ndarray, visibility_threshold: float = 0.5) -> np.ndarray:
    """O(n^2) evaluation of the visibility rule used in the original notebook."""
    values = np.asarray(series, dtype=float)
    adjacency = np.zeros((len(values), len(values)), dtype=np.float32)
    for start in range(len(values)):
        maximum_adjusted_slope = -np.inf
        for end in range(start + 1, len(values)):
            slope = (values[end] - values[start]) / (end - start)
            if slope >= maximum_adjusted_slope:
                adjacency[start, end] = 1.0
            adjusted = (values[end] - values[start] - visibility_threshold) / (end - start)
            maximum_adjusted_slope = max(maximum_adjusted_slope, adjusted)
    return adjacency


def graph_data(series: np.ndarray, adjacency: np.ndarray) -> Data:
    features = torch.tensor(series, dtype=torch.float32).view(-1, 1)
    edge_index = torch.tensor(np.asarray(np.nonzero(adjacency)), dtype=torch.long)
    return Data(x=features, edge_index=edge_index)


def train_embeddings(data: Data) -> np.ndarray:
    """Original settings: 200 epochs, hidden=32, out=16, lr=0.001."""
    model = ImprovedGCN(dropout=0.3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    node_count = data.x.shape[0]
    edge_index = data.edge_index
    adjacency = torch.zeros((node_count, node_count))
    adjacency[edge_index[0], edge_index[1]] = 1
    negative_count = int(edge_index.shape[1] * 2)

    best_loss = float("inf")
    best_embeddings: np.ndarray | None = None
    patience_counter = 0

    for epoch in tqdm(range(200), desc="Training 2V-GCN"):
        model.train()
        optimizer.zero_grad()
        embeddings = model(data.x, edge_index)
        predicted = torch.sigmoid(embeddings @ embeddings.T)
        positive_loss = F.binary_cross_entropy(
            predicted[edge_index[0], edge_index[1]],
            adjacency[edge_index[0], edge_index[1]],
        )
        negative_indices = torch.randint(0, node_count, (2, negative_count))
        negative_loss = F.binary_cross_entropy(
            predicted[negative_indices[0], negative_indices[1]],
            torch.zeros(negative_count),
        )
        contrastive_loss = F.mse_loss(embeddings @ embeddings.T, torch.eye(node_count))
        total_loss = positive_loss + 0.5 * negative_loss + 0.1 * contrastive_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            best_embeddings = embeddings.detach().numpy()
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= 20:
            break

    if best_embeddings is None:
        raise RuntimeError("2V-GCN training did not produce embeddings.")
    return best_embeddings


def time_ordered_similarity(embeddings: np.ndarray) -> np.ndarray:
    full = cosine_similarity(embeddings)
    result = np.zeros_like(full)
    upper_rows, upper_cols = np.triu_indices(len(full), k=1)
    result[upper_rows, upper_cols] = full[upper_rows, upper_cols]
    return result


def top_historical_nodes(similarity: np.ndarray, interval: np.ndarray, top_n: int = 2) -> pd.DataFrame:
    rows = []
    for current in range(len(interval)):
        if current == 0:
            indices = np.full(top_n, -1, dtype=int)
            scores = np.zeros(top_n)
            values = np.zeros((top_n, 2))
        else:
            historical = similarity[:current, current]
            indices = np.argsort(-historical)[:top_n]
            scores = historical[indices]
            values = interval[indices]
            if len(indices) < top_n:
                missing = top_n - len(indices)
                indices = np.pad(indices, (0, missing), constant_values=-1)
                scores = np.pad(scores, (0, missing), constant_values=0)
                values = np.vstack([values, np.zeros((missing, 2))])

        row: dict[str, float | int] = {"TimePoint": current}
        for rank in range(top_n):
            row[f"Sim_{rank + 1}_UB"] = float(values[rank, 0])
            row[f"Sim_{rank + 1}_LB"] = float(values[rank, 1])
            row[f"Index_{rank + 1}"] = int(indices[rank])
            row[f"Similarity_{rank + 1}"] = float(scores[rank])
        rows.append(row)
    return pd.DataFrame(rows)


def calculate_similarity_nodes(input_file: Path, output_file: Path, seed: int | None = None) -> Path:
    frame = pd.read_excel(input_file)
    required = ["Date", "UB", "LB"]
    if not all(column in frame.columns for column in required):
        raise ValueError(f"{input_file} must contain {required}.")
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    interval = frame[["UB", "LB"]].to_numpy(dtype=float)
    upper = StandardScaler().fit_transform(interval[:, 0].reshape(-1, 1)).ravel()
    lower = StandardScaler().fit_transform(interval[:, 1].reshape(-1, 1)).ravel()
    upper_graph = graph_data(upper, sparse_directed_visibility_graph(upper, 0.5))
    lower_graph = graph_data(lower, sparse_directed_visibility_graph(lower, 0.5))
    upper_embeddings = train_embeddings(upper_graph)
    lower_embeddings = train_embeddings(lower_graph)
    combined = (
        time_ordered_similarity(upper_embeddings)
        + time_ordered_similarity(lower_embeddings)
    ) / 2.0
    nodes = top_historical_nodes(combined, interval, top_n=2)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        nodes.to_excel(writer, sheet_name="Similar_Nodes", index=False)
        pd.DataFrame(combined).to_excel(writer, sheet_name="Similarity_Matrix", index=False)
    return output_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    print(calculate_similarity_nodes(args.input, args.output, args.seed))


if __name__ == "__main__":
    main()

