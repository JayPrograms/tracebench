"""Build a fresh Northstar Shop demo database through the public CLI."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEMO_ROOT = REPOSITORY_ROOT / "demos" / "customer-support"
CLUSTERING_RUN = "northstar-support-slices-v1"
DATASET_REFERENCE = "northstar-support-eval:1.0"
SLICE_LABELS = (
    "account-security",
    "billing-invoices",
    "cancellation-retention",
    "delivery-tracking",
    "refund-policy",
    "returns-exchanges",
    "product-troubleshooting",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Construct the Northstar Shop evaluation database from fixtures."
    )
    parser.add_argument(
        "--database",
        type=Path,
        required=True,
        help="Fresh SQLite database path to create.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace the exact database path and SQLite sidecar files if present.",
    )
    return parser.parse_args()


def _remove_database(database_path: Path) -> None:
    """Remove only the explicitly requested database and SQLite sidecars."""
    for path in (
        database_path,
        database_path.with_name(database_path.name + "-wal"),
        database_path.with_name(database_path.name + "-shm"),
    ):
        if path.exists():
            path.unlink()


def _run_cli(database_path: Path, *arguments: str) -> None:
    environment = os.environ.copy()
    environment["TRACEBENCH_DB_PATH"] = str(database_path)
    subprocess.run(
        [sys.executable, "-m", "tracebench", *arguments],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=True,
    )


def main() -> None:
    """Ingest, cluster, label, and build the checked-in demo dataset."""
    arguments = _parse_args()
    database_path = arguments.database.expanduser().resolve()
    database_path.parent.mkdir(parents=True, exist_ok=True)
    if database_path.exists() or any(
        sidecar.exists()
        for sidecar in (
            database_path.with_name(database_path.name + "-wal"),
            database_path.with_name(database_path.name + "-shm"),
        )
    ):
        if not arguments.force:
            raise SystemExit(
                f"database already exists: {database_path}; "
                "choose a fresh path or pass --force"
            )
        _remove_database(database_path)

    _run_cli(database_path, "ingest", str(DEMO_ROOT / "traces.jsonl"))
    _run_cli(
        database_path,
        "traces",
        "cluster",
        "--name",
        CLUSTERING_RUN,
        "--clusters",
        "7",
    )
    for cluster_number, label in enumerate(SLICE_LABELS):
        _run_cli(
            database_path,
            "slices",
            "rename",
            CLUSTERING_RUN,
            str(cluster_number),
            label,
        )
    _run_cli(
        database_path,
        "dataset",
        "build",
        "--name",
        "northstar-support-eval",
        "--version",
        "1.0",
        "--from-slices",
        CLUSTERING_RUN,
        "--size",
        "21",
        "--case-file",
        str(DEMO_ROOT / "cases.json"),
    )
    print(f"Built {DATASET_REFERENCE} in {database_path}")


if __name__ == "__main__":
    main()
