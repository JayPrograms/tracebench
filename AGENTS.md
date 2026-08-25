# AGENTS.md

## Before editing

- Read `AGENTS.md`, `README.md`, and `pyproject.toml` before making changes. If a listed file does not exist yet, note that and continue with the available guidance.

## Project conventions

- Use Python 3.13.
- Use a `src` layout for Python packages.
- Keep TraceBench local-first and functional without paid services.
- Use Typer for the CLI, Pydantic for data validation, the standard-library `sqlite3` module for persistence, pytest for tests, Ruff for linting and formatting, and mypy for static type checking.
- Add tests for every behavior change.
- Prefer straightforward implementations and avoid unnecessary abstractions.

## Current milestone: intelligent evaluation

### Stage A

- Add a shared provider interface while preserving the existing fixture provider.
- Add an Ollama-compatible local provider.
- Add rubric-based LLM judging with versioned judge prompts.
- Validate and persist structured judge results, including judge confidence.
- Add result caching and review escalation.

### Stage B

- Add local TF-IDF trace vectorization with optional Truncated SVD.
- Add KMeans clustering and persistent named slices.
- Add representative dataset sampling.
- Add slice-level aggregation and regression gates.

## Scope constraints

- Keep fixture providers as the default for tests and CI.
- Do not require hosted or paid services.
- Do not add authentication, distributed workers, cloud infrastructure, vector databases, a dashboard, or unrelated refactors.
- Never commit secrets, `.env` files, `.venv`, SQLite databases, caches, model files, or generated reports.
- Narrow reviewed-fixture exception: `demos/customer-support/hosted/northstar-fail-detail.json` is the one allowed versioned hosted-demo snapshot. It contains only synthetic Northstar data and exists so the hosted read-only dashboard can start from a stable, auditable `ExperimentDetail` without running models or evaluation jobs. Do not add local reports, arbitrary exports, production trace exports, or generated dashboard state under this exception.

## Before finishing

- Run Ruff, mypy, and pytest before finishing implementation tasks.
