from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


def load_and_concat(partitions: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Concatenate partitions from a ``PartitionedDataSet`` into a single DataFrame.

    Accepts CSV/Parquet partitions with at least 20 numeric columns. Non-numeric
    auxiliary columns are allowed and will be dropped during preprocessing.
    """
    if not partitions:
        logger.warning("No files found in incoming dataset. Skipping.")
        return pd.DataFrame()

    frames = []
    for name, df in partitions.items():
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)
        frames.append(df)
        logger.info(f"Loaded partition: {name} shape={df.shape}")
    data = pd.concat(frames, ignore_index=True)
    logger.info(f"Concatenated shape={data.shape}")
    return data


def preprocess(df: pd.DataFrame, standardize: bool = True) -> pd.DataFrame:
    if df.empty:
        return df
    # Select numeric columns only
    num_df = df.select_dtypes(include=[np.number]).copy()
    # Keep first 20 numeric dims (require >=20)
    if num_df.shape[1] >= 20:
        num_df = num_df.iloc[:, :20]
    else:
        raise ValueError(f"Expect >=20 numeric cols, got {num_df.shape[1]}")

    if standardize:
        scaler = StandardScaler()
        num_df.loc[:, :] = scaler.fit_transform(num_df.values)
    return num_df


def run_tsne(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    if df.empty:
        return df
    logger.info("Running t-SNE...")
    tsne = TSNE(
        n_components=int(params.get("n_components", 2)),
        perplexity=float(params.get("perplexity", 30)),
        learning_rate=params.get("learning_rate", "auto"),
        init=params.get("init", "pca"),
        random_state=int(params.get("random_state", 42)),
        n_iter=int(params.get("max_iter", 1000)),
        verbose=1,
    )
    emb = tsne.fit_transform(df.values)
    out = pd.DataFrame(emb, columns=["x", "y"])  # 2D
    return out


def plot_tsne(embedding: pd.DataFrame):
    import matplotlib.pyplot as plt

    if embedding.empty:
        fig = plt.figure()
        return fig, fig
    fig, ax = plt.subplots(figsize=(6, 5), dpi=120)
    ax.scatter(embedding["x"], embedding["y"], s=8, alpha=0.8)
    ax.set_title("t-SNE (20D → 2D)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, linestyle=":", linewidth=0.5)
    return fig, fig
