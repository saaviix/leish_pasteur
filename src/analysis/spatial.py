"""
spatial.py
==========
Analyse spatiale des hotspots de presence de P. sergenti.
Utilise un graphe d'adjacence pour detecter les clusters spatiaux
et calcule une approximation du I de Moran local.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "data_prep"))
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from data_prep import config

logger = logging.getLogger(__name__)
sns.set_style("whitegrid")


def _load_province_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Province table not found: {path}")
    return pd.read_csv(path)


def _load_posterior(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Posterior CSV not found: {path}")
    return pd.read_csv(path)


def _load_adjacency(path: Path) -> np.ndarray:
    if not path.exists():
        logger.warning("Adjacency file not found: %s. Building fully connected graph instead.", path)
        return None
    return np.load(path)


def _build_graph(n_provinces: int, adj_edges: Optional[np.ndarray]) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(range(n_provinces))
    if adj_edges is not None and adj_edges.size > 0:
        for edge in adj_edges:
            if len(edge) >= 2:
                g.add_edge(int(edge[0]), int(edge[1]))
    else:
        for i in range(n_provinces):
            for j in range(i + 1, n_provinces):
                g.add_edge(i, j)
    return g


def _local_morans_i(psi_values: np.ndarray, graph: nx.Graph) -> np.ndarray:
    n = len(psi_values)
    mean_psi = np.mean(psi_values)
    centered = psi_values - mean_psi

    lisa = np.zeros(n)
    for i in range(n):
        neighbors = list(graph.neighbors(i))
        if not neighbors:
            lisa[i] = 0.0
            continue
        lag = np.mean(centered[neighbors])
        lisa[i] = len(neighbors) * centered[i] * lag

    return lisa


def identify_clusters(psi_values: np.ndarray, graph: nx.Graph) -> pd.DataFrame:
    lisa = _local_morans_i(psi_values, graph)
    threshold = np.std(lisa) if np.std(lisa) > 0 else 1.0
    clusters = np.where(lisa > threshold, "hot spot",
                        np.where(lisa < -threshold, "cold spot", "not significant"))
    return pd.DataFrame({"local_morans_i": lisa, "cluster": clusters})


def plot_spatial_clusters(province_df: pd.DataFrame, clusters_df: pd.DataFrame,
                          output_path: Path) -> None:
    if "psi_mean" not in province_df.columns:
        logger.warning("psi_mean missing in province table; skipping spatial cluster plot.")
        return

    merged = province_df.reset_index(drop=True).join(clusters_df.reset_index(drop=True))

    fig, ax = plt.subplots(figsize=(10, 7))
    palette = {"hot spot": "red", "cold spot": "blue", "not significant": "lightgray"}
    sns.scatterplot(data=merged, x="longitude", y="latitude", hue="cluster",
                    palette=palette, size="psi_mean", sizes=(20, 200), alpha=0.8, ax=ax)
    ax.set_title("Spatial Clusters of P. sergenti Occupancy", fontsize=14)
    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)
    ax.legend(title="Cluster type", loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info("Saved spatial cluster map to %s", output_path)


def plot_lisa_distribution(clusters_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(data=clusters_df, x="local_morans_i", hue="cluster",
                 element="step", stat="density", common_norm=False, ax=ax)
    ax.set_title("Local Moran's I Distribution", fontsize=14)
    ax.set_xlabel("Local Moran's I", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info("Saved LISA distribution plot to %s", output_path)


def main() -> None:
    config.ensure_dirs()
    logger.info("Running spatial analysis...")

    try:
        province_df = _load_province_table(config.PROVINCE_TABLE)
    except FileNotFoundError:
        logger.error("Province table not available: %s", config.PROVINCE_TABLE)
        return
    try:
        posterior_df = _load_posterior(config.POSTERIOR_CSV)
    except FileNotFoundError:
        logger.error("Posterior CSV not available: %s", config.POSTERIOR_CSV)
        return

    adj_edges = _load_adjacency(config.ADJ_EDGES)
    graph = _build_graph(len(province_df), adj_edges)

    psi_col = "psi_mean" if "psi_mean" in posterior_df.columns else posterior_df.columns[-1]
    psi_values = posterior_df[psi_col].fillna(0).values

    clusters_df = identify_clusters(psi_values, graph)
    clusters_df.to_csv(config.PROCESSED / "spatial_clusters.csv", index=False)
    logger.info("Saved spatial clusters CSV to %s", config.PROCESSED / "spatial_clusters.csv")

    plot_spatial_clusters(province_df, clusters_df, config.FIGURES / "spatial_cluster_map.png")
    plot_lisa_distribution(clusters_df, config.FIGURES / "spatial_lisa_distribution.png")

    logger.info("Spatial analysis completed.")


if __name__ == "__main__":
    main()
