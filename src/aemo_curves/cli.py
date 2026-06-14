"""`aemo-run` CLI entry point. Thin wrapper around the scripts."""
from __future__ import annotations

import typer

from .config import ARTIFACTS_DIR, MLFLOW_TRACKING_URI, PROJECT_ROOT, TEST_MONTHS, TRAIN_MONTHS

app = typer.Typer(add_completion=False, help="AEMO STEM curve-prediction tools")


@app.command()
def info() -> None:
    """Show project paths and the canonical train/test split."""
    print(f"PROJECT_ROOT:        {PROJECT_ROOT}")
    print(f"ARTIFACTS_DIR:       {ARTIFACTS_DIR}")
    print(f"MLFLOW_TRACKING_URI: {MLFLOW_TRACKING_URI}")
    print(f"Train months ({len(TRAIN_MONTHS)}): {TRAIN_MONTHS[0]} -> {TRAIN_MONTHS[-1]}")
    print(f"Test months  ({len(TEST_MONTHS)}): {TEST_MONTHS[0]} -> {TEST_MONTHS[-1]}")


@app.command()
def build_curves(side: str = "supply") -> None:
    """Build and cache train/test curve tables (supply/demand/both)."""
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "build_curves.py"), "--side", side],
        check=True,
    )


@app.command()
def build_embeddings(side: str = "supply") -> None:
    """Build and cache train/test curve embeddings."""
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "build_embeddings.py"), "--side", side],
        check=True,
    )


@app.command("run")
def run(name: str, mlflow: bool = True, subset_days: int = 0) -> None:
    """Run an experiment by its dotted name: e.g. `aemo-run run demand.7d.vgrid.rolling_ridge_exog`."""
    from .core.experiment import Experiment
    from .runner import run_experiment

    path = PROJECT_ROOT.joinpath("experiments", *name.split(".")).with_suffix(".yaml")
    if not path.exists():
        raise typer.BadParameter(f"No experiment at {path}")
    subset = {"test_days": subset_days} if subset_days else None
    res = run_experiment(Experiment.from_yaml(path), log_to_mlflow=mlflow, subset=subset)
    print(f"\n=== {res.experiment_name} ===")
    for k, v in res.metrics.items():
        print(f"  {k}: {v}")


@app.command("group")
def group(name: str, cache: bool = True, mlflow: bool = True, subset_days: int = 0) -> None:
    """Run a comparable group and rank it: e.g. `aemo-run group demand.7d.hc_mae`."""
    from .runner import run_group

    subset = {"test_days": subset_days} if subset_days else None
    table = run_group(name, use_cache=cache, log_to_mlflow=mlflow, subset=subset)
    print(f"\n=== {name} (lower is better) ===")
    print(table.to_string(index=False))


@app.command()
def runs() -> None:
    """List MLflow runs."""
    from .tracking import get_experiment_runs

    df = get_experiment_runs()
    if df.empty:
        print("No MLflow runs found.")
        return
    print(df.to_string(index=False))


@app.command("ui")
def ui(port: int = 5000) -> None:
    """Launch the local MLflow UI."""
    import subprocess

    subprocess.run(
        ["mlflow", "ui", "--backend-store-uri", MLFLOW_TRACKING_URI, "--port", str(port)],
        check=True,
    )


if __name__ == "__main__":
    app()
