"""Prepare a fresh failing Northstar detail export for the local explorer."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from tracebench.experiment_models import ExperimentVerdict
from tracebench.reporting import ExperimentDetail

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEMO_ROOT = REPOSITORY_ROOT / "demos" / "customer-support"
FIXTURE_CONFIG = DEMO_ROOT / "experiment.fixture.yaml"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the failing Northstar fixture and export its detail JSON."
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination for the dashboard-ready detail JSON.",
    )
    parser.add_argument(
        "--database",
        type=Path,
        help="Optional database path to retain instead of using a temporary database.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output detail export.",
    )
    return parser.parse_args()


def _run(
    database_path: Path,
    *arguments: str,
    expected_exit: int = 0,
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["TRACEBENCH_DB_PATH"] = str(database_path)
    result = subprocess.run(
        [sys.executable, "-m", "tracebench", *arguments],
        cwd=REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != expected_exit:
        raise RuntimeError(
            f"TraceBench command failed with exit {result.returncode}: "
            f"{result.stdout}\n{result.stderr}"
        )
    return result


def _prepare(
    database_path: Path, output_path: Path, *, overwrite: bool
) -> ExperimentDetail:
    subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts" / "bootstrap_customer_support_demo.py"),
            "--database",
            str(database_path),
            "--force",
        ],
        cwd=REPOSITORY_ROOT,
        check=True,
    )
    run_result = _run(
        database_path,
        "experiment",
        "run",
        str(FIXTURE_CONFIG),
        "--json",
        expected_exit=1,
    )
    try:
        report = json.loads(run_result.stdout)
        experiment_id = report["experiment_id"]
        if report["status"] != "completed" or report["verdict"] != "FAIL":
            raise ValueError("the failing fixture did not produce a completed FAIL")
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise RuntimeError(
            "the failing fixture returned an invalid JSON report"
        ) from error

    export_arguments = [
        "experiment",
        "export",
        str(experiment_id),
        "--output",
        str(output_path),
    ]
    if overwrite:
        export_arguments.append("--overwrite")
    _run(
        database_path,
        *export_arguments,
    )
    try:
        detail = ExperimentDetail.model_validate_json(
            output_path.read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as error:
        raise RuntimeError(
            "the exported detail JSON failed strict validation"
        ) from error
    if detail.verdict is not ExperimentVerdict.FAIL:
        raise RuntimeError("the exported detail does not preserve the FAIL verdict")
    return detail


def main() -> None:
    """Build the authoritative demo through public CLI APIs only."""

    arguments = _parse_args()
    output_path = arguments.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if arguments.database is not None:
        detail = _prepare(
            arguments.database.expanduser().resolve(),
            output_path,
            overwrite=arguments.overwrite,
        )
    else:
        with tempfile.TemporaryDirectory(prefix="tracebench-northstar-") as folder:
            database_path = Path(folder) / "northstar-support.sqlite3"
            detail = _prepare(database_path, output_path, overwrite=arguments.overwrite)
    print(
        f"Prepared {detail.verdict.value if detail.verdict else 'NO VERDICT'} "
        f"detail for {detail.experiment_id} at {output_path}"
    )


if __name__ == "__main__":
    main()
