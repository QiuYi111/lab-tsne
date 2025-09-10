from __future__ import annotations

from typing import Dict
import os
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


# src/lab_tsne/pipelines/tsne/nodes.py 里的 load_and_concat

def load_and_concat(partitions: dict) -> pd.DataFrame:
    if not partitions:
        logger.warning("No files in /data/incoming. Skipping.")
        return pd.DataFrame()

    frames = []
    for name, part in partitions.items():
        # 1) 懒加载：callable -> 真正的数据
        if callable(part):
            part = part()

        # 2) 路径字符串 -> 手动读（兜底，避免奇葩情况）
        if isinstance(part, (str, bytes, os.PathLike)):
            if str(part).lower().endswith(".csv"):
                df = pd.read_csv(part)
            elif str(part).lower().endswith(".parquet"):
                df = pd.read_parquet(part)
            else:
                raise ValueError(f"Unknown file type for partition {name}: {part}")
        else:
            df = part

        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)

        logger.info(f"Loaded partition: {name} shape={df.shape}")
        frames.append(df)

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
        max_iter=int(params.get("max_iter", 1000)),
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
