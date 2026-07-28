"""Command-line interface for TraceBench."""

import sqlite3
from contextlib import closing
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Annotated

import typer

from tracebench.ingestion import IngestionSummary, ingest_file
from tracebench.models import Trace
from tracebench.storage import (
    connect_database,
    list_traces,
    resolve_database_path,
    timestamp_to_text,
)

app = typer.Typer(
    add_completion=False,
    help="Local-first regression testing for LLM applications.",
)
traces_app = typer.Typer(help="Inspect stored traces.")
app.add_typer(traces_app, name="traces")


@app.callback()
def main() -> None:
    """Run TraceBench commands."""


@app.command("version")
def show_version() -> None:
    """Print the installed TraceBench version."""
    typer.echo(package_version("tracebench"))


@app.command()
def ingest(
    path: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="Path to a newline-delimited JSON trace file.",
        ),
    ],
) -> None:
    """Validate and store traces from a JSONL file."""
    try:
        summary = ingest_file(
            path,
            resolve_database_path(),
            error_reporter=lambda message: typer.echo(message, err=True),
        )
    except (OSError, UnicodeError, sqlite3.Error) as error:
        typer.echo(f"Error: could not ingest traces: {error}", err=True)
        raise typer.Exit(code=1) from error

    _print_ingestion_summary(summary)


@traces_app.command("list")
def show_traces() -> None:
    """List stored traces in newest-first order."""
    try:
        with closing(connect_database(resolve_database_path())) as connection:
            traces = list_traces(connection)
    except (OSError, sqlite3.Error, ValueError) as error:
        typer.echo(f"Error: could not list traces: {error}", err=True)
        raise typer.Exit(code=1) from error

    if not traces:
        typer.echo("No traces found.")
        return

    _print_trace_table(traces)


def _print_ingestion_summary(summary: IngestionSummary) -> None:
    typer.echo(f"Records read: {summary.records_read}")
    typer.echo(f"Records accepted: {summary.records_accepted}")
    typer.echo(f"Invalid records: {summary.invalid_records}")
    typer.echo(f"Duplicates skipped: {summary.duplicates_skipped}")
    typer.echo(f"Records stored: {summary.records_stored}")


def _print_trace_table(traces: list[Trace]) -> None:
    headers = ("TRACE_ID", "TIMESTAMP", "TASK_TYPE", "HAS_RESPONSE")
    rows = [
        (
            trace.trace_id,
            timestamp_to_text(trace.timestamp),
            trace.task_type,
            "yes" if trace.response is not None else "no",
        )
        for trace in traces
    ]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    typer.echo(_format_table_row(headers, widths))
    for row in rows:
        typer.echo(_format_table_row(row, widths))


def _format_table_row(row: tuple[str, ...], widths: list[int]) -> str:
    return "  ".join(
        value.ljust(width) for value, width in zip(row, widths, strict=True)
    ).rstrip()
