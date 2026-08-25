# Northstar Shop customer-support demo

This self-contained demo evaluates a plausible Northstar Shop support-assistant
upgrade. The candidate gives better delivery and troubleshooting guidance, but
it weakens billing, cancellation, and refund-policy behavior. The critical
refund regression blocks the release.

The checked-in fixture workflow is deterministic and requires no model server.
The CI-safe fixture uses a second candidate and judge fixture, while the
optional Ollama configuration runs the same dataset through local models.

## What the demo covers

- 35 ingested customer-support traces
- seven reproducible TF-IDF/KMeans slices with five traces each
- a balanced 21-case slice-built dataset with immutable provenance
- seven reference, seven deterministic, and seven rubric cases
- stable fixture providers for baseline, candidate, and rubric judge roles
- judge confidence, critical-failure review escalation, and immutable cache hits
- global, evaluation-mode, and refund-policy slice gates

The frozen sampler selects these traces:

| Slice | Selected traces |
| --- | --- |
| `account-security` | `account-01`, `account-03`, `account-05` |
| `billing-invoices` | `billing-03`, `billing-04`, `billing-05` |
| `cancellation-retention` | `cancel-02`, `cancel-04`, `cancel-05` |
| `delivery-tracking` | `delivery-02`, `delivery-03`, `delivery-04` |
| `refund-policy` | `refund-01`, `refund-03`, `refund-04` |
| `returns-exchanges` | `return-01`, `return-02`, `return-05` |
| `product-troubleshooting` | `troubleshoot-03`, `troubleshoot-04`, `troubleshoot-05` |

The expected slice-manifest hash is
`05933fc8b290d4af0ad82e433d4656b2cec3010159c92b9f0357474f4aac6dee`.

## Run the fixture workflow

Run these commands from the repository root with the Python 3.13 environment
activated:

```powershell
$demoDb = ".tracebench/northstar-support.sqlite3"
python scripts/bootstrap_customer_support_demo.py --database $demoDb --force

tracebench dataset show northstar-support-eval:1.0

tracebench experiment run demos/customer-support/experiment.fixture.yaml --json
$LASTEXITCODE

tracebench experiment run demos/customer-support/experiment.fixture.yaml --json
$LASTEXITCODE
```

Both experiment commands intentionally exit `1` because the completed release
gate verdict is `FAIL`. Exit `1` is a regression decision, not an operational
error. The first run records 14 judge-cache misses; the second records 14 hits.

To inspect one attempt in a future dashboard, capture its ID from the JSON
report and export the shared detail document:

```powershell
$report = tracebench experiment run demos/customer-support/experiment.fixture.yaml --json
$experimentId = ($report | ConvertFrom-Json).experiment_id
tracebench experiment export $experimentId --output .tracebench/northstar-detail.json
```

The export includes the persisted dataset and slice provenance, provider/run
metadata, generation observations, and combined case results while reusing the
machine report's authoritative scores and gate. It excludes raw judge attempts,
malformed judge responses, and system-prompt contents. Prompts, contexts,
outputs, and reference answers are customer-support trace data; keep the JSON
file private and out of version control. The machine report remains the compact
automation/release-gate contract, while this detail document is the shared input
for the later reporting and dashboard layers.

The bootstrap script is a thin cross-platform wrapper around the real
TraceBench CLI. It ingests the checked-in traces, clusters them, applies the
seven checked-in labels, and builds the sealed dataset; it never writes SQLite
rows directly.

## Run the CI-safe gate

The evaluation workflow uses the passing candidate and judge fixtures below.
Run the same gate locally after creating a fresh database:

```powershell
$demoDb = ".tracebench/northstar-support-ci.sqlite3"
python scripts/bootstrap_customer_support_demo.py --database $demoDb --force
$env:TRACEBENCH_DB_PATH = $demoDb
tracebench experiment run demos/customer-support/experiment.ci-pass.yaml --json
$LASTEXITCODE  # 0; PASS
```

The CI-safe candidate keeps the delivery and troubleshooting improvements and
correctly handles billing, cancellation, and refund policy. The showcase
candidate above deliberately regresses those policies and must remain a local
failure example; it is never substituted into the release-gate workflow.

## Expected fixture result

| Metric | Baseline | Candidate |
| --- | ---: | ---: |
| Score | 0.933333 | 0.900000 |
| Passed cases | 19/21 | 18/21 |
| Pass rate | 0.904762 | 0.857143 |

The candidate has two newly passed cases:

- `delivery-02`: better missing-delivery checks and carrier escalation
- `troubleshoot-04`: complete earbud diagnostics

It has three newly failed cases:

- `billing-03`: omits the 10-business-day posting limit
- `cancel-02`: incorrectly restricts cancellation after a retention discount
- `refund-01`: promises an unsupported refund after the 30-day deadline

`refund-01` is critical priority and needs review for `critical_failure`.
`account-03` passes but has judge confidence `0.62`, below the configured `0.7`
threshold, so it needs review for `low_confidence`.

## Case-file contract

`cases.json` is a strict schema-version-`1` document. Each entry identifies a
trace selected by the sampler and overrides its evaluation mode, scorers or
rubric, priority, and review status. Selected traces omitted from the file retain
the existing reference-mode behavior and snapshot their source response.

Unknown keys, duplicate JSON keys, duplicate trace definitions, unsupported
schema versions, invalid mode settings, and overrides for traces not selected by
sampling fail the build without creating a dataset. Deterministic and rubric
cases never retain the source response or a reference answer.

## Optional Ollama run

Install and start Ollama, then pull the configured model:

```powershell
ollama pull llama3.2:3b
tracebench experiment run demos/customer-support/experiment.ollama.yaml
```

The configuration uses the versioned `baseline-v1.txt` and `candidate-v2.txt`
answer prompts plus TraceBench's versioned rubric judge prompts. Local-model
outputs and the resulting verdict can vary by Ollama and model version; the
fixture configuration is the reproducible acceptance scenario.
