"""Shared utilities for model_training."""

from __future__ import annotations

import mlflow


def mlflow_active() -> bool:
    """Check whether an MLflow run is currently active."""
    return mlflow.active_run() is not None
