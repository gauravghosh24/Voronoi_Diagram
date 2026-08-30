"""Helpers for training and using ML cutoff-radius predictors."""

import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def calculate_oracle_cutoff_from_dist_map(dist_map):
    """Return the smallest integer cutoff that covers the distance map."""

    dist_map = np.asarray(dist_map)
    if dist_map.size == 0:
        return 0

    max_distance = np.max(dist_map)
    if np.issubdtype(dist_map.dtype, np.integer):
        if max_distance >= np.iinfo(dist_map.dtype).max:
            return 0
    elif not np.isfinite(max_distance):
        return 0

    return int(np.ceil(np.sqrt(max_distance)))


def load_cutoff_model(model_path):
    """Load a joblib cutoff model artifact."""

    model_artifact = joblib.load(model_path)
    if isinstance(model_artifact, dict) and "model" in model_artifact:
        return model_artifact

    feature_columns = _load_feature_columns_from_sidecar(model_path)
    if feature_columns is not None:
        return {"model": model_artifact, "feature_columns": feature_columns}

    return model_artifact


def _load_feature_columns_from_sidecar(model_path):
    sidecar_path = Path(model_path).with_name("cutoff_feature_columns.json")
    if not sidecar_path.exists():
        return None

    with sidecar_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict):
        return data.get("feature_columns")
    return data


def _split_model_artifact(model_artifact):
    if isinstance(model_artifact, dict) and "model" in model_artifact:
        return model_artifact["model"], model_artifact.get("feature_columns")
    return model_artifact, None


def _prepare_prediction_frame(feature_dict, feature_columns):
    frame = pd.DataFrame([feature_dict])
    if "mode" in frame.columns:
        frame = pd.get_dummies(frame, columns=["mode"])

    if feature_columns is not None:
        frame = frame.reindex(columns=feature_columns, fill_value=0)

    return frame


def predict_cutoff(
    model,
    feature_dict,
    safety_factor=1.10,
    safety_add=5,
    max_cutoff=None,
):
    """Predict a conservative integer cutoff radius from extracted features."""

    estimator, feature_columns = _split_model_artifact(model)
    if feature_columns is None and hasattr(estimator, "feature_names_in_"):
        feature_columns = list(estimator.feature_names_in_)

    frame = _prepare_prediction_frame(feature_dict, feature_columns)
    raw_pred = float(estimator.predict(frame)[0])
    cutoff = int(math.ceil(raw_pred * float(safety_factor) + float(safety_add)))

    if max_cutoff is not None:
        cutoff = min(cutoff, int(max_cutoff))

    cutoff = max(0, cutoff)
    return cutoff
