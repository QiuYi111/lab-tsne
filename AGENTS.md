# Repository Guidelines

## Project Structure & Module Organization
- `src/lab_tsne/`: Python package, entrypoint (`__main__.py`), hooks, and pipeline registry.
- `tests/`: Pytest suite; mirror `src/` structure (files named `test_*.py`).
- `conf/`: Kedro configuration (`conf/base` for shared, `conf/local` for developer‑specific; keep secrets in `conf/local`).
- `data/`: Layered data folders (`01_raw` → `08_reporting`); do not commit data.
- `docs/`: Sphinx documentation source (`docs/source`).
- `notebooks/`: Prototyping; keep outputs stripped from Git.

## Build, Test, and Development Commands
- Install deps: `pip install -r requirements.txt`
- Dev extras (lint/tests): `pip install -e .[dev]`
- Run pipeline: `kedro run` (or `python -m lab_tsne`)
- Run tests: `pytest` (coverage configured in `pyproject.toml`)
- Lint and fix: `ruff check . --fix`
- Format: `ruff format .`
- Build docs: `sphinx-build -b html docs/source docs/build/html`

## Coding Style & Naming Conventions
- Language: Python ≥ 3.9 with type hints where practical.
- Formatter/Lint: Ruff (line length 88; E501 ignored, import order via isort rules).
- Indentation: 4 spaces. Strings: prefer double quotes.
- Naming: `snake_case` for functions/vars, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants, modules in `snake_case`.
- Public APIs: docstrings (Google/Numpy style accepted) and clear typing.

## Testing Guidelines
- Framework: Pytest with coverage on `src/lab_tsne`.
- Location/Names: place under `tests/` mirroring `src/`; files `test_*.py`, tests `test_*`.
- Running: `pytest -q` (CI-friendly) or `pytest -k <pattern>` to filter.
- Kedro tests: create a `KedroSession` to run pipelines (see `tests/test_run.py`). Avoid reliance on external state or local config.

## Commit & Pull Request Guidelines
- Commits: concise, imperative; recommended Conventional Commits (`feat:`, `fix:`, `chore:`, `docs:`, `test:`).
- PRs: include purpose, scope, and linked issues; summarize pipeline/config changes; attach screenshots for docs/UI outputs when relevant.
- Require: passing CI, lint clean, tests updated/added, no data/secrets in diffs.

## Security & Configuration Tips
- Never commit credentials or `.env`. Keep secrets in `conf/local/` and exclude from VCS.
- Use `conf/base/` for shared, non‑secret defaults; override locally in `conf/local/`.
- Validate inputs from `data/01_raw` and prefer catalog‑managed IO for reproducibility.
