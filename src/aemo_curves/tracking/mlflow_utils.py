"""MLflow setup helpers and a uniform way to log ExperimentResult objects."""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import mlflow
import pandas as pd
import yaml

from aemo_curves.core.result import ExperimentResult

from ..config import MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI, MLRUNS_DIR

log = logging.getLogger(__name__)


def init_mlflow(
    experiment_name: str = MLFLOW_EXPERIMENT_NAME,
    tracking_uri: str = MLFLOW_TRACKING_URI,
) -> str:
    """Point MLflow at the local store and create the experiment if needed.

    Returns the experiment id.
    """
    mlflow.set_tracking_uri(tracking_uri)
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        exp_id = mlflow.create_experiment(experiment_name)
    else:
        exp_id = exp.experiment_id
    mlflow.set_experiment(experiment_name)
    return exp_id


def log_experiment_result(result: ExperimentResult) -> str:
    """Persist an ExperimentResult as an MLflow run. Returns the run id.

    Logs parameters, metrics, and the predictions + result.json as artifacts.
    """
    init_mlflow()
    with mlflow.start_run(run_name=f"{result.experiment_name}") as run:
        for k, v in (result.params or {}).items():
            try:
                mlflow.log_param(k, v)
            except Exception as exc:
                log.warning("Could not log param %s=%r: %s", k, v, exc)
        for k, v in (result.metrics or {}).items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(k, float(v))
        mlflow.set_tag("model_name", result.model_name)
        mlflow.set_tag("experiment_name", result.experiment_name)
        if result.notes:
            mlflow.set_tag("notes", result.notes)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            result.mlflow_run_id = run.info.run_id
            result.save(tmp_path)
            mlflow.log_artifacts(str(tmp_path), artifact_path="result")
        return run.info.run_id


def get_experiment_runs(
    experiment_name: str = MLFLOW_EXPERIMENT_NAME,
    tracking_uri: str = MLFLOW_TRACKING_URI,
) -> pd.DataFrame:
    """Return all finished runs of an experiment as a dataframe."""
    mlflow.set_tracking_uri(tracking_uri)
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        return pd.DataFrame()
    return mlflow.search_runs([exp.experiment_id])


def load_run_result(
    run_id: str,
    tracking_uri: str = MLFLOW_TRACKING_URI,
) -> ExperimentResult:
    """Rehydrate an ExperimentResult saved as an MLflow artifact."""
    mlflow.set_tracking_uri(tracking_uri)
    with tempfile.TemporaryDirectory() as tmp:
        local = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path="result", dst_path=tmp
        )
        return ExperimentResult.load(Path(local))


def _run_start_time(run_dir: Path) -> int:
    meta_path = run_dir / "meta.yaml"
    if not meta_path.exists():
        return 0
    try:
        with open(meta_path) as f:
            meta = yaml.safe_load(f)
        return int(meta.get("start_time", 0))
    except Exception:
        return 0


def find_run_by_name(
    name: str,
    *,
    latest: bool = True,
    mlruns_dir: Path = MLRUNS_DIR,
) -> Path | None:
    """Return the run dir whose experiment_name tag == `name`, or None.

    Several scripts reuse experiment-name tags, so a glob can hit multiple run
    dirs. With `latest=True` (default) the most recent run by meta.yaml
    start_time wins; with `latest=False` the first glob hit is returned.
    """
    candidates = [
        tag.parent.parent
        for tag in Path(mlruns_dir).glob("*/*/tags/experiment_name")
        if tag.read_text().strip() == name
    ]
    if not candidates:
        return None
    if not latest:
        return candidates[0]
    return max(candidates, key=_run_start_time)


def load_predictions_by_name(
    name: str,
    *,
    latest: bool = True,
    mlruns_dir: Path = MLRUNS_DIR,
) -> pd.DataFrame:
    """Load the scored predictions of the MLflow run tagged `name`."""
    rd = find_run_by_name(name, latest=latest, mlruns_dir=mlruns_dir)
    if rd is None:
        raise FileNotFoundError(f"no MLflow run with experiment_name={name!r}")
    df = pd.read_pickle(rd / "artifacts" / "result" / "predictions.pkl")
    df["Trading Date"] = pd.to_datetime(df["Trading Date"])
    return df
