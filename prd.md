# 目标

在实验室公用服务器上，研究同学把 20 维数据文件丢进共享目录 `/data/incoming`，容器内的 watcher 立刻触发 `kedro run`：

1. 读取新数据 → 2) 预处理 → 3) t‑SNE 降维 → 4) 生成可视化 PNG/SVG → 5) 产出 parquet/csv 结果到 `/data/processed`、图表到 `/data/reports`。

---

## 目录结构

```
lab-tsne/
├─ docker/
│  ├─ Dockerfile
│  ├─ entrypoint.sh
│  └─ requirements.txt
├─ docker-compose.yml
├─ pyproject.toml
├─ README.md
├─ conf/
│  └─ base/
│     ├─ catalog.yml
│     ├─ parameters.yml
│     └─ logging.yml
└─ src/
   └─ lab_tsne/
      ├─ __init__.py
      ├─ pipeline.py
      ├─ nodes.py
      └─ watch_and_run.py
```

---

## Dockerfile（`docker/Dockerfile`）

```dockerfile
FROM python:3.11-slim

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 基础依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    inotify-tools \
    && rm -rf /var/lib/apt/lists/*

# 工作目录
WORKDIR /app

# 依赖
COPY docker/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# 代码与配置
COPY pyproject.toml ./
COPY conf ./conf
COPY src ./src

# 环境变量
ENV KEDRO_ENV=base \
    PYTHONPATH=/app/src

# 入口脚本
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

CMD ["/entrypoint.sh"]
```

---

## requirements（`docker/requirements.txt`）

```text
kedro>=0.19.8
pandas>=2.2.2
pyarrow>=17.0.0
scikit-learn>=1.5.0
matplotlib>=3.9.0
watchdog>=4.0.1
loguru>=0.7.2
```

---

## docker-compose（`docker-compose.yml`）

```yaml
version: "3.9"
services:
  lab-tsne:
    build:
      context: .
      dockerfile: docker/Dockerfile
    image: lab-tsne:latest
    environment:
      - KEDRO_ENV=base
    volumes:
      - ./conf:/app/conf:ro
      - ./src:/app/src:ro
      - /data/incoming:/data/incoming
      - /data/processed:/data/processed
      - /data/reports:/data/reports
    restart: unless-stopped
```

> 注：将宿主机的三个目录预先创建并赋予写权限：`/data/incoming`、`/data/processed`、`/data/reports`。

---

## 入口脚本（`docker/entrypoint.sh`）

```bash
#!/usr/bin/env bash
set -euo pipefail

# 初次启动时先跑一遍，处理已有文件（可选）
python -m lab_tsne.watch_and_run --once

# 常驻监听 /data/incoming 目录
python -m lab_tsne.watch_and_run
```

---

## Kedro 配置

### catalog（`conf/base/catalog.yml`）

```yaml
# 原始 20 维数据：支持 CSV 或 Parquet，两种都给出；实际用哪种由文件扩展名决定
raw_partitioned:
  type: PartitionedDataSet
  path: /data/incoming
  dataset:
    type: pandas.CSVDataset
    load_args:
      dtype: float64
      header: infer
      encoding: utf-8

# 预处理后的合并数据
preprocessed_data:
  type: pandas.ParquetDataset
  filepath: /data/processed/preprocessed.parquet

# t-SNE 二维结果
embedding_2d:
  type: pandas.ParquetDataset
  filepath: /data/processed/embedding_2d.parquet

# 可视化图
tsne_plot_png:
  type: matplotlib.MatplotlibWriter
  filepath: /data/reports/tsne.png
  save_args:
    format: png

tsne_plot_svg:
  type: matplotlib.MatplotlibWriter
  filepath: /data/reports/tsne.svg
  save_args:
    format: svg
```

### parameters（`conf/base/parameters.yml`）

```yaml
n_components: 2
perplexity: 30
learning_rate: 'auto'
init: 'pca'
random_state: 42
max_iter: 1000
standardize: true   # 是否标准化
min_rows: 10        # 少于该行数则跳过运行
```

### logging（`conf/base/logging.yml`）

最小化版本即可，或沿用 Kedro 默认。

---

## Pipeline 与 Nodes

`src/lab_tsne/nodes.py`

```python
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Iterable
from loguru import logger
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


def load_and_concat(partitions: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """将 PartitionedDataSet 的分片字典合并为单表。
    要求每个 CSV/Parquet 至少包含 20 维数值列。可兼容带有非数值辅助列的情况。"""
    if not partitions:
        logger.warning("No files in /data/incoming. Skipping.")
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
    # 选取数值列
    num_df = df.select_dtypes(include=[np.number]).copy()
    # 仅保留前 20 维（多于 20 维时）
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
    out = pd.DataFrame(emb, columns=["x", "y"])  # 2 维
    return out


def plot_tsne(embedding: pd.DataFrame):
    import matplotlib.pyplot as plt
    if embedding.empty:
        fig = plt.figure()
        return fig
    fig, ax = plt.subplots(figsize=(6, 5), dpi=120)
    ax.scatter(embedding["x"], embedding["y"], s=8, alpha=0.8)
    ax.set_title("t-SNE (20D → 2D)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, linestyle=":", linewidth=0.5)
    return fig
```

`src/lab_tsne/pipeline.py`

```python
from kedro.pipeline import Pipeline, node
from .nodes import load_and_concat, preprocess, run_tsne, plot_tsne


def create_pipeline(**kwargs):
    return Pipeline([
        node(
            func=load_and_concat,
            inputs={"partitions": "raw_partitioned"},
            outputs="preprocessed_raw",
            name="load_concat",
        ),
        node(
            func=preprocess,
            inputs=dict(df="preprocessed_raw", standardize="params:standardize"),
            outputs="preprocessed_data",
            name="preprocess",
        ),
        node(
            func=run_tsne,
            inputs=dict(df="preprocessed_data", params="parameters"),
            outputs="embedding_2d",
            name="tsne",
        ),
        node(
            func=plot_tsne,
            inputs="embedding_2d",
            outputs=["tsne_plot_png", "tsne_plot_svg"],
            name="plot",
        ),
    ])
```

`src/lab_tsne/__init__.py`

```python
from kedro.framework.project import configure_project

configure_project("lab_tsne")

__all__ = ["configure_project"]
```

> 若使用 Kedro 0.19+，无需 `settings.py` 也可；如果项目模板要求，可添加 `settings.py` 指向 `create_pipeline`。

---

## Watcher（`src/lab_tsne/watch_and_run.py`）

一个简单的 watchdog 守护：

```python
from __future__ import annotations
import argparse
import subprocess
from pathlib import Path
from loguru import logger
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

INCOMING = Path("/data/incoming")


def run_kedro():
    logger.info("Trigger kedro run...")
    try:
        subprocess.run(["kedro", "run"], check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"kedro run failed: {e}")


class Handler(FileSystemEventHandler):
    def on_any_event(self, event):
        if event.is_directory:
            return
        p = Path(event.src_path)
        if p.suffix.lower() in {".csv", ".parquet"}:
            logger.info(f"Detected change: {p}")
            run_kedro()


def main(once: bool = False):
    INCOMING.mkdir(parents=True, exist_ok=True)
    if once:
        run_kedro()
        return
    event_handler = Handler()
    observer = Observer()
    observer.schedule(event_handler, str(INCOMING), recursive=False)
    observer.start()
    logger.info(f"Watching {INCOMING} for new CSV/Parquet files...")
    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    main(once=args.once)
```

---

## pyproject（`pyproject.toml`）

```toml
[build-system]
requires = ["setuptools", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "lab-tsne"
version = "0.1.0"
dependencies = []

[tool.kedro]
package_name = "lab_tsne"
project_name = "lab-tsne"
pipeline = "lab_tsne.pipeline:create_pipeline"
```

---

## 使用方法

1. 在服务器上准备目录，并给写权限：

   ```bash
   sudo mkdir -p /data/incoming /data/processed /data/reports
   sudo chown -R $USER:$USER /data
   ```
2. 构建与启动：

   ```bash
   docker compose build
   docker compose up -d
   ```
3. 实验同学将数据放入 `/data/incoming`：

   * 支持多个 CSV/Parquet，数值列不少于 20 列，包含表头更友好。
   * 例如：`/data/incoming/batch_2025_0910.csv`。
4. 容器自动触发：几秒内完成合并、预处理、t‑SNE、绘图与落盘。
5. 结果查看：

   * 表格：`/data/processed/preprocessed.parquet`、`/data/processed/embedding_2d.parquet`
   * 图片：`/data/reports/tsne.png`、`tsne.svg`

---

## 可选增强

* 用 `PartitionedDataSet` 的自定义保存策略，做增量去重（根据文件名或哈希）。
* 增加 `umap-learn`，并在参数中切换算法。
* 为每个批次产出独立文件：在 `watch_and_run.py` 里把新增文件名传给 `kedro run --params batch_file:xxx`，并在 `catalog.yml` 做参数化路径。
* 把 `perplexity` 自动设为 `min(30, n_samples/3)`，小样本更稳。
* 引入 `kedro-viz` 容器，暴露 `:4141` 端口给内网看流程与数据血缘。

---

## 样例数据生成（可选，给同学演示）

在宿主机执行：

```bash
python - <<'PY'
import numpy as np, pandas as pd
from pathlib import Path
Path('/data/incoming').mkdir(parents=True, exist_ok=True)

# 三个高斯簇，每簇中心在 20 维空间不同坐标
centers = [np.zeros(20), np.ones(20)*5, np.concatenate([np.ones(10)*-4, np.ones(10)*3])]
rows = []
for c in centers:
    x = np.random.randn(300, 20) * 0.8 + c
    rows.append(x)
X = np.vstack(rows)
cols = [f"f{i:02d}" for i in range(20)]
pd.DataFrame(X, columns=cols).to_csv('/data/incoming/demo_batch.csv', index=False)
print('Wrote /data/incoming/demo_batch.csv')
PY
```

放完文件后，容器会自动跑完并在 `/data/reports/tsne.png` 产出图。

