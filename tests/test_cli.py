"""Tests for the TraceBench command-line interface."""

from importlib.metadata import version as package_version

from typer.testing import CliRunner

from tracebench.cli import app

runner = CliRunner()


def test_version_command_succeeds() -> None:
    """The version command exits successfully."""
    result = runner.invoke(app, ["version"])

    assert result.exit_code == 0, result.output


def test_version_command_reports_installed_version() -> None:
    """The version command prints the installed distribution version."""
    result = runner.invoke(app, ["version"])

    assert result.stdout == f"{package_version('tracebench')}\n"
