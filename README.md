# TraceBench

TraceBench is a local-first regression-testing tool for LLM applications. It turns application traces into versioned evaluation datasets, compares baseline and candidate model or prompt configurations, scores their outputs, and blocks releases when quality regresses.

## The problem

Changes to models, prompts, and application logic can silently reduce output quality. Reproducing real interactions, comparing configurations consistently, and deciding whether a release is safe often requires custom tooling or paid hosted services. TraceBench aims to make that workflow repeatable, version-controlled, and runnable on a developer's machine.

## Planned V0.1 workflow

1. Import application traces into a local, versioned evaluation dataset.
2. Define baseline and candidate model or prompt configurations.
3. Run both configurations against the same dataset.
4. Score and compare their outputs.
5. Produce a regression report and return a failing exit code when configured quality thresholds are not met.

TraceBench is local-first and designed to work without paid services.

## Development status

TraceBench currently supports importing application traces into a local SQLite
database and listing the stored traces.

## Trace ingestion

Import newline-delimited JSON traces from the repository's sample dataset:

```powershell
tracebench ingest datasets/traces.sample.jsonl
```

Each input line must contain one JSON trace object. Invalid records are reported
with their line numbers and skipped, while valid records continue to be stored.
Duplicate `trace_id` values are also skipped. The command prints:

```text
Records read: N
Records accepted: N
Invalid records: N
Duplicates skipped: N
Records stored: N
```

List stored traces in newest-first order:

```powershell
tracebench traces list
```

By default, TraceBench stores data in `.tracebench/tracebench.sqlite3` relative
to the current directory. Override the location for tests or local workflows
with `TRACEBENCH_DB_PATH`:

```powershell
$env:TRACEBENCH_DB_PATH = "C:\path\to\tracebench.sqlite3"
tracebench ingest datasets/traces.sample.jsonl
tracebench traces list
```

## Local installation

TraceBench requires Python 3.13. From the repository root, activate the existing
virtual environment and install the package with its development dependencies:

```powershell
.\.venv\Scripts\Activate.ps1
python --version
python -m pip install -e ".[dev]"
```

The Python version should be `3.13.x`. Verify the installed CLI:

```powershell
tracebench version
```

## Validation

Run formatting, linting, type checking, and tests from the repository root:

```powershell
ruff format --check .
ruff check .
mypy src
pytest
```
