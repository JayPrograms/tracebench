# TraceBench demo and recruiter package

This is a 2–3 minute recording plan for the checked-in Northstar Shop fixture.
It shows the real CLI and Streamlit application; it does not manufacture a video
or GIF. Run from the repository root with Python 3.13 and the optional dashboard
extra installed.

## Recording script and shot list

### 0:00–0:20 — pitch

Say:

> TraceBench is local-first regression testing for LLM applications. It replays a
> versioned dataset through a baseline and candidate, scores reference,
> deterministic, and rubric cases, and returns a release-blocking exit code when
> quality regresses. The workflow is fixture-first for deterministic CI, with an
> optional local Ollama provider and no paid service dependency.

Show the README badges and the first paragraph of the [Northstar guide](../demos/customer-support/README.md).

### 0:20–0:45 — dataset and slices

Prepare a fresh database and display the case count:

```powershell
$demoDb = ".tracebench/northstar-support.sqlite3"
python scripts/bootstrap_customer_support_demo.py --database $demoDb --force
$env:TRACEBENCH_DB_PATH = $demoDb
tracebench dataset show northstar-support-eval:1.0
```

Expected evidence: 35 traces, seven clusters/slices with five traces each, and a
sealed 21-case dataset containing seven reference, seven deterministic, and seven
rubric cases. Point out that the selected trace IDs are generated and frozen by
the sampler manifest; the planning matrix is not used as an assumption.

### 0:45–1:10 — run the intentional regression

```powershell
tracebench experiment run `
  demos/customer-support/experiment.fixture.yaml --json
$LASTEXITCODE
```

Expected output includes `"status":"completed"`, `"verdict":"FAIL"`, baseline
score `0.933333`, candidate score `0.900000`, two newly passed cases, three newly
failed cases, and exit code `1`. Save the returned `experiment_id` for the export.

Run the same command a second time:

```powershell
tracebench experiment run `
  demos/customer-support/experiment.fixture.yaml --json
$LASTEXITCODE
```

It again exits `1`; the second report shows 14 judge-cache hits rather than new
judge calls. Explain that fixture latency is not a product performance claim.

### 1:10–1:40 — inspect the critical refund regression

Export the first attempt (use its actual ID):

```powershell
$report = tracebench experiment run `
  demos/customer-support/experiment.fixture.yaml --json
$experimentId = ($report | ConvertFrom-Json).experiment_id
tracebench experiment export $experimentId `
  --output .tracebench/northstar-detail.json --overwrite
```

Open the dashboard with the fresh export:

```powershell
python scripts/prepare_customer_support_dashboard.py `
  --output .tracebench/northstar-detail.json --overwrite
streamlit run dashboard/app.py
```

On **Overview**, show `FAIL`, the score delta, three newly failed cases, and the
critical refund-policy row. On **Regression explorer**, filter to `refund-policy`
and expand `refund-01`. Explain that the candidate promises an unsupported refund
after the 30-day deadline while the persisted baseline/candidate scores and slice
provenance come from the export.

### 1:40–2:10 — review queue

Select **Review queue**. Show the two persisted items:

- `refund-01` — critical failure requiring review.
- `account-03` — low judge confidence (`0.62` below `0.7`).

Select **Provenance** briefly to show the dataset slice source, provider snapshots,
configuration hash, and run lifecycle. Emphasize that the UI is read-only and does
not connect to SQLite or run a scorer.

### 2:10–2:30 — GitHub Actions release gate

Show the [CI workflow](https://github.com/JayPrograms/tracebench/actions/workflows/ci.yml)
and [Evaluation Gate](https://github.com/JayPrograms/tracebench/actions/workflows/eval-gate.yml)
badges. Explain the distinction:

- The local showcase candidate intentionally fails so a reviewer can see a useful
  regression story.
- Evaluation Gate builds a fresh database and runs `experiment.ci-pass.yaml` with
  a safe candidate. It exits `0`, returns `PASS`, and uploads the
  `northstar-support-ci-report` JSON artifact.

## Recruiter-facing summary

### 30-second verbal explanation

TraceBench is a Python 3.13, SQLite-backed evaluation harness for LLM changes. It
turns traces into immutable, reproducibly sampled datasets, runs baseline and
candidate providers, scores mixed reference/deterministic/rubric cases, and
returns a quality-gate exit code. Its strongest engineering boundary is that
normalized persisted results are the authority: the CLI, CI artifact, and
read-only Streamlit explorer all consume the same versioned report data. Fixtures
make the release gate deterministic, while Ollama is an explicit local opt-in.

### Resume bullets

- Built a local-first LLM regression harness with SQLite persistence, strict
  Pydantic schemas, UUID5 identities, transactional run stages, and exit-code
  release gates.
- Implemented deterministic TF-IDF/KMeans slice sampling and sealed mixed-mode
  datasets with provenance snapshots, answer-leak prevention, immutable rubric
  judge caching, and review escalation.
- Added GitHub Actions validation/evaluation gates and a read-only Streamlit
  explorer backed by versioned `ExperimentDetail` exports; the Northstar fixture
  benchmark covers 35 traces, seven slices, and 21 cases.

### Likely interview questions

**Why SQLite?** It is inspectable, transactional, dependency-light, and adequate
for a local-first benchmark. Normalized rows make relationships and audit history
explicit without requiring a service.

**How is sampling reproducible?** Traces are ordered canonically, clusters use
fixed seeds, quotas are allocated deterministically, and SHA-256 ranks plus the
immutable source-manifest/configuration identity decide selection. Tests freeze the
actual selected IDs and manifest.

**How do you prevent answer leakage?** Reference cases snapshot the source answer;
deterministic and rubric case builders reject/omit source responses and reference
answers. The case-file parser validates this contract before sealing.

**Why separate the dashboard from evaluation?** `ExperimentDetail` is a strict
versioned boundary. The dashboard reads persisted values and never opens SQLite,
rescoring or changing a verdict, so presentation cannot drift from CI.

**How are failures classified?** Preflight errors return `2` without an attempt,
quality-gate failures return `1` with a completed verdict, and provider/storage
failures return `3` with an operationally failed attempt where possible.

**How do you avoid database contention?** Providers run outside write locks. Each
run stage, judge audit, cache operation, and comparison commit uses a short
transaction with rollback on partial failure.

## Hosted dashboard strategy

The [public Northstar dashboard](https://tracebench-northstar.streamlit.app/)
uses a deterministic, read-only artifact flow:

1. The reviewed `demos/customer-support/hosted/northstar-fail-detail.json`
   snapshot is validated as strict `ExperimentDetail` data and contains only
   synthetic Northstar support data.
2. Streamlit consumes that snapshot and remains a read-only
   `ExperimentDetail` consumer. It does not receive a SQLite database, run
   Ollama, make model calls, or hold API keys.
3. Local uploads and exports remain supported for private review, but they may
   contain sensitive prompts, contexts, outputs, and reference answers. Keep
   those files out of Git and preserve the narrow AGENTS.md snapshot exception.

The public URL is a preview of the synthetic fixture, not live production
telemetry. Moving it from `feature/productization` to `main` only requires
changing the app's selected branch in Streamlit settings; the custom subdomain
can remain unchanged.
