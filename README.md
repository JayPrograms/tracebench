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

TraceBench supports importing application traces, promoting selected traces into
versioned evaluation datasets, and running persisted baseline-versus-candidate
experiments with deterministic regression gates, rubric judging, immutable judge
result caching, and human-review escalation metadata.

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

## Reproducible trace clustering and named slices

Create an immutable, version-named clustering run over the current trace snapshot:

```powershell
tracebench traces cluster --name support-slices-v1 --clusters 6
```

The default document is the stored prompt exactly as ingested. Add
`--include-context` to use a fixed prompt/context format containing canonical
context JSON, or `--svd-components N` to enable the optional dimensionality
reduction step. TraceBench requires the configured SVD dimension to satisfy
`1 <= N < min(trace count, TF-IDF feature count)`.

List the resulting numeric slices and assign a human-readable label:

```powershell
tracebench slices list support-slices-v1
tracebench slices rename support-slices-v1 0 refunds
```

Run names, configurations, source provenance, and trace assignments are immutable.
Labels are editable metadata and are unique within a run after Unicode NFKC
normalization and case folding. Clustering uses deterministic trace ordering and
fixed random seeds. The configuration identity records exact TraceBench,
scikit-learn, NumPy, and SciPy versions; assignment equivalence is not promised
across dependency versions, platforms, or numerical runtimes.

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

Deterministic mode requires one or more scorer configurations. The core experiment
loop validates and executes the built-in configurations:

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

## Core evaluation loop

The checked-in sample demonstrates the complete workflow. Start with a fresh local
database, ingest the traces, and create the versioned dataset used by the sample
experiment:

```powershell
$env:TRACEBENCH_DB_PATH = ".tracebench/core-loop.sqlite3"
tracebench ingest datasets/traces.sample.jsonl
tracebench dataset create `
  --name core-eval `
  --version 0.1 `
  --description "Core evaluation-loop sample"
```

Promote the three traces with all five deterministic scorer types:

```powershell
tracebench dataset add-trace `
  --dataset core-eval:0.1 `
  --trace-id sample-001 `
  --mode deterministic `
  --scorer-file datasets/scorer.exact-match.json `
  --scorer-file datasets/scorer.contains.json

tracebench dataset add-trace `
  --dataset core-eval:0.1 `
  --trace-id sample-002 `
  --mode deterministic `
  --scorer-file datasets/scorer.regex.json

tracebench dataset add-trace `
  --dataset core-eval:0.1 `
  --trace-id sample-003 `
  --mode deterministic `
  --scorer-file datasets/scorer.json-validity.json `
  --scorer-file datasets/scorer.required-keys.json
```

Run the baseline and candidate fixtures, score both, compare them globally and by
evaluation mode, and apply the configured regression gate:

```powershell
tracebench experiment run datasets/core-evaluation.sample.yaml
```

The sample baseline fails the `required_keys` scorer for one case, while the
candidate fixes it without introducing a failure, so the command prints `PASS` and
exits `0`. Use `--json` for a stable machine-readable result:

```powershell
tracebench experiment run datasets/core-evaluation.sample.yaml --json
$LASTEXITCODE
```

Each fixture is UTF-8 JSONL with exactly one strict record per evaluation case:

```json
{"eval_id":"eval_...","output":"provider output, which may be empty"}
```

Duplicate, missing, or unknown evaluation IDs are rejected before an experiment
attempt is created. Fixture paths in YAML are relative to the configuration file.
The supported scorer configurations are:

- `exact_match`: `expected` and optional `case_sensitive` (default `true`)
- `contains`: nonblank `substring` and optional `case_sensitive`
- `regex`: compilable `pattern` and optional `case_sensitive`; matching uses search
  semantics
- `json_validity`: an empty configuration and strict standard-JSON parsing
- `required_keys`: a nonempty, unique list of required top-level JSON object `keys`

Reference-mode cases use implicit case-sensitive exact matching against their
reference answer. Rubric cases are evaluated by the experiment's configured judge
provider, described below.

The YAML gate defaults to zero allowed global score drop and zero newly failed
cases. Optional `by_mode` entries may override either limit and inherit the other
global value. An override must name a supported mode present in the dataset.
Absent modes are omitted from results rather than represented with zero or `NaN`.

Every successful preflight creates a new experiment attempt. Names are reusable
labels, so rerunning the same name never overwrites or resumes an earlier attempt.
Every attempt receives a new ID; behaviorally identical dataset definitions,
fixtures, scorers, and thresholds receive the same deterministic configuration
hash.

Experiment status and verdict are separate. A completed gate decision has status
`completed` and verdict `PASS` or `FAIL`. An operationally failed attempt has status
`failed` and a null verdict. Exit codes are:

| Exit | Meaning |
| ---: | --- |
| `0` | Completed and passed the regression gate |
| `1` | Completed and failed the regression gate |
| `2` | Usage or preflight validation error; no attempt was persisted |
| `3` | Operational failure; an accepted attempt is marked failed when possible |

Baseline results, candidate results, and comparison/gate results use separate
transactions. A failed stage rolls back its partial scored rows; a follow-up
transaction records the operational failure. Raw judge attempts use independent
audit transactions so malformed responses remain inspectable after a failed run.
Judge cache lookups and inserts also use short transactions that never remain open
while waiting for a provider. Machine-readable reports are reconstructed from
normalized SQLite rows.

## Rubric judging

An experiment containing rubric cases requires one judge configuration shared by
the baseline and candidate runs. This local fixture example is suitable for tests
and CI:

```yaml
schema_version: 1
name: rubric-check
dataset: support-eval:0.1
baseline: {provider: fixture, path: support-baseline.jsonl}
candidate: {provider: fixture, path: support-candidate.jsonl}
judge:
  provider: fixture
  path: support-judge.jsonl
  prompt_version: rubric-v1
  prompt_file: ../prompts/judges/rubric-v1.txt
  retry_prompt_file: ../prompts/judges/rubric-retry-v1.txt
  confidence_threshold: 0.7
gate:
  max_score_drop: 0
  max_new_failures: 0
```

The checked-in primary prompt is `prompts/judges/rubric-v1.txt`. Prompt paths are
resolved relative to the experiment YAML and preflighted like answer-provider
prompts. Prompt version, content hashes, judge settings, and the finite
`confidence_threshold` in the inclusive range `[0,1]` contribute to the
configuration hash. A result below the threshold requires human review but does
not alter scores, gates, retries, verdicts, or exit codes.

Judge fixtures contain one strict JSONL record for each rubric case and role. A
second raw output is optional and is consumed only when the first response is
malformed:

```json
{"eval_id":"eval_...","role":"baseline","outputs":["raw response","optional retry response"]}
```

The judge must return JSON only:

```json
{"schema_version":1,"criteria":[{"criterion":"Exact rubric criterion text","score":1.0,"passed":true,"reason":"Concise reason"}],"overall_score":1.0,"overall_passed":true,"confidence":0.9}
```

Scores and confidence are finite numbers in `[0,1]`. Criteria must exactly cover
the stored rubric text without missing, unknown, or duplicate entries. TraceBench
uses the equal-weight mean of criterion scores as the case score and requires
`overall_score` to match within an absolute tolerance of `0.000001` with no
relative tolerance. A case passes only when every criterion passes, and the
returned `overall_passed` must agree.

Each criterion appears in the existing nonempty scorer-result list with scorer
name `rubric`. The additive rubric-only `judge` result contains the validated
overall fields, confidence, threshold diagnostic, attempt count, cache metadata,
and review state. `cache_hit` is always `true` or `false` for a new result; an
in-place migration represents historical A2 results with `null`. Cache hits have
zero attempts because no provider call occurs. Existing deterministic/reference
JSON case objects, the top-level report schema version, and operational-failure
JSON remain unchanged at version `1`. Raw judge outputs, cached response content,
validation failures, latency, and provider metadata remain database-only.

Successful rubric responses are cached by a SHA-256 identity covering the exact
evaluated output, ordered rubric, other rendered judge inputs, prompt version and
content hashes, judge model/provider identity, and every configured setting that
affects generation. Confidence threshold and timeout are excluded because they do
not affect a successful generation. Threshold changes can therefore reuse a
response while deriving a new review state.

Each immutable cache entry retains both the exact successful provider output and a
canonical parsed response. `response_hash` is the SHA-256 hash of the canonical
parsed JSON, not the raw string. Every hit validates the key identity, canonical
JSON, hash, and that the retained raw output strictly parses into the same
canonical response. Any mismatch is an operational integrity failure; TraceBench
does not overwrite the entry or fall back to the provider. Malformed outputs and
provider failures are never cached. When a malformed first response is repaired,
only the exact successful retry output enters the cache.

Concurrent callers may both observe a miss and generate without holding a
database write lock. The first valid cache insert wins immutably for future hits.
If another caller produced a different valid response, its run remains a miss and
uses its own audited response; it does not claim that the winning cache row was
the source of its result.

Rubric results use the review states `not_required`, `needs_review`, and
`reviewed`. Low confidence or failure of a `critical`-priority case produces
`needs_review`, with both reasons retained when applicable. `reviewed` is reserved
for a future review-editing workflow; this release adds no review-editing command
or UI.

Malformed judge JSON, schema violations, criterion coverage errors, or inconsistent
overall fields receive at most one retry using the separately versioned retry
prompt. Configuration, transport, timeout, provider-envelope, persistence, and
low-confidence outcomes are never retried. Exhausting the retry is an operational
failure with exit `3` and a null verdict.

## Local model execution with Ollama

An experiment role may use an Ollama-compatible local server instead of a fixture.
TraceBench does not install or start Ollama and does not download models. Install
Ollama, start it, and pull the selected model yourself before running the
experiment. No API key or authentication configuration is supported.

Store the versioned system prompt in a UTF-8 text file, for example
`prompts/support-answer-v1.txt`, and reference it from the experiment YAML:

```yaml
schema_version: 1
name: local-support-check
dataset: support-eval:0.1
baseline:
  provider: fixture
  path: support-baseline.jsonl
candidate:
  provider: ollama
  base_url: http://localhost:11434
  model: llama3.2:3b
  prompt_version: support-answer-v1
  system_prompt_file: ../prompts/support-answer-v1.txt
  temperature: 0
  timeout_seconds: 120
  seed: 42
gate:
  max_score_drop: 0
  max_new_failures: 0
```

Relative prompt paths are resolved from the experiment YAML directory. Prompt
files may be UTF-8 with or without a byte-order mark and must contain nonblank
text. TraceBench reads and validates them during preflight, before creating an
experiment attempt or contacting Ollama. A missing, unreadable, invalidly encoded,
or blank prompt therefore exits `2` and persists no attempt.

The provider snapshot records `prompt_version` and a SHA-256 hash of the decoded
prompt contents, not the prompt path or contents. Changing the prompt text changes
the experiment configuration hash; moving an identical file does not. Generation
requests append the evaluation input and canonical JSON context using fixed
TraceBench delimiters and use Ollama's non-streaming `/api/generate` endpoint.

Successful case rows record client-observed generation latency and selected
Ollama response metadata, including model, termination, duration, and token-count
fields when returned. These internal persistence additions do not change the
current human report or `--json` schema. Connection failures, timeouts, HTTP
errors, and invalid responses are operational failures with exit `3`; TraceBench
does not retry, start a server, pull a missing model, or select a fallback model.

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
