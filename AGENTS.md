# AGENTS.md

## Before editing

- Read `AGENTS.md`, `README.md`, and `pyproject.toml` before making changes. If a listed file does not exist yet, note that and continue with the available guidance.

## Project conventions

- Use Python 3.12.
- Use a `src` layout for Python packages.
- Keep TraceBench local-first and functional without paid services.
- Use Typer for the CLI, Pydantic for data validation, the standard-library `sqlite3` module for persistence, pytest for tests, Ruff for linting and formatting, and mypy for static type checking.
- Add tests for every behavior change.
- Prefer straightforward implementations and avoid unnecessary abstractions.

## Scope constraints

- Do not add external APIs, cloud infrastructure, dashboards, embeddings, LLM integrations, ORMs, or Docker unless explicitly requested.
- Never commit secrets, `.env` files, `.venv`, SQLite databases, caches, model files, or generated reports.

## Before finishing

- Run Ruff, mypy, and pytest before finishing implementation tasks.
