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

TraceBench is in initial setup. The V0.1 interface and implementation have not been released yet.
