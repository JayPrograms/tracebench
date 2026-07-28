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
database, listing stored traces, and promoting selected traces into explicitly
versioned evaluation datasets.

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

## Versioned evaluation datasets

Create an independent dataset version:

```powershell
tracebench dataset create `
  --name support-eval `
  --version 0.1 `
  --description "Customer support regression benchmark"
```

List datasets and their case counts:

```powershell
tracebench dataset list
```

Promote a stored trace. Dataset references use `name:version`. Reference mode
requires either an explicit `--reference-answer` or the
`--use-source-response` flag:

```powershell
tracebench dataset add-trace `
  --dataset support-eval:0.1 `
  --trace-id sample-001 `
  --mode reference `
  --use-source-response
```

Deterministic mode requires one or more scorer configurations. TraceBench stores
these configurations but does not execute them:

```powershell
tracebench dataset add-trace `
  --dataset support-eval:0.1 `
  --trace-id sample-002 `
  --mode deterministic `
  --scorer-file datasets/scorer.exact-match.json `
  --priority high
```

Scorer files avoid native-command JSON quoting differences across Windows
PowerShell versions. Each file contains one scorer JSON object; repeat
`--scorer-file` for additional configurations. The inline `--scorer` option
remains available for callers that already have a correctly tokenized JSON
argument. Scorer files may be UTF-8 with or without the byte-order mark emitted
by Windows PowerShell.

Rubric mode accepts one or more criteria through repeated options:

```powershell
tracebench dataset add-trace `
  --dataset support-eval:0.1 `
  --trace-id sample-003 `
  --mode rubric `
  --rubric "The answer is factually correct." `
  --rubric "The answer directly addresses the request."
```

New cases default to `medium` priority and `draft` review status. The available
priorities are `low`, `medium`, `high`, and `critical`; review statuses are
`draft`, `approved`, and `rejected`.

Inspect or export the resulting dataset:

```powershell
tracebench dataset show support-eval:0.1
tracebench dataset export `
  support-eval:0.1 `
  --output datasets/support-eval-v0.1.jsonl
```

Exports use JSONL envelopes. The first line is always a `dataset` metadata
record, including for an empty dataset. Each following line is an `eval_case`
record. Cases contain a snapshot of the source trace ID, timestamp, task type,
metadata, prompt, and context, so the export remains auditable without the
SQLite database. Reference cases also retain the source response when one was
available; deterministic and rubric cases omit it to prevent answer leakage.
An existing export is protected unless `--overwrite` is explicitly supplied.

Dataset and evaluation-case IDs are stable UUID5 values. Dataset IDs are
derived from the trimmed dataset name and version under a fixed namespace;
case IDs are derived from that stable dataset ID and the source trace ID under
a second fixed namespace. Rebuilding the same logical dataset therefore
reproduces the same IDs without relying on Python's process-randomized
`hash()`.

Databases containing only traces are initialized with the evaluation schema
automatically. Development databases created by the earlier, unmerged
evaluation-dataset implementation must be recreated because they contain
random IDs and lack the provenance columns and final integrity constraints.
TraceBench reports this incompatibility when opening such a database; reingest
the original trace JSONL and recreate its datasets in a new database.

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
