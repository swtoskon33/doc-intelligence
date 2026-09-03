"""MLflow tracking for extraction benchmark runs.

Each backend evaluated on the golden set becomes a tracked run: backend and golden-set
size as params, accuracy / review rate / latency as metrics. That makes backend choices
reproducible and comparable over time instead of a number in a terminal.

Tracking is optional: if MLflow is not installed the helpers no-op, so the benchmark
still runs offline.
"""
from __future__ import annotations

import os
from contextlib import contextmanager


def mlflow_available() -> bool:
    try:
        import mlflow  # noqa: F401
    except ImportError:
        return False
    return True


def _noop_logger(params=None, metrics=None, artifact=None) -> None:
    return None


def _mlflow_logger(params=None, metrics=None, artifact=None) -> None:
    import mlflow

    if params:
        mlflow.log_params(params)
    if metrics:
        mlflow.log_metrics(metrics)
    if artifact and os.path.exists(artifact):
        mlflow.log_artifact(artifact)


@contextmanager
def track_run(experiment: str, run_name: str):
    """Yield a logger callable; a no-op when MLflow is unavailable."""
    if not mlflow_available():
        yield _noop_logger
        return

    import mlflow

    uri = os.getenv("MLFLOW_TRACKING_URI")
    if uri:
        mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name):
        yield _mlflow_logger
