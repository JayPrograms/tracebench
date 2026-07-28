"""Command-line interface for TraceBench."""

from importlib.metadata import version as package_version

import typer

app = typer.Typer(
    add_completion=False,
    help="Local-first regression testing for LLM applications.",
)


@app.callback()
def main() -> None:
    """Run TraceBench commands."""


@app.command("version")
def show_version() -> None:
    """Print the installed TraceBench version."""
    typer.echo(package_version("tracebench"))
