#!/usr/bin/env bash
set -euo pipefail

# Process existing files once (optional)
python -m lab_tsne.watch_and_run --once || true

# Watch /data/incoming and trigger kedro run
python -m lab_tsne.watch_and_run

