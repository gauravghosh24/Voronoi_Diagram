"""Train a RandomForest cutoff-radius predictor."""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import CSV_DIR
from src.utils import ensure_dirs


MODELS_DIR = PROJECT_ROOT / "models"


def _prepare_features(df):
    drop_columns = [column for column in ("seed", "target_cutoff") if column in df.columns]
    features = df.drop(columns=drop_columns)
    if "mode" in features.columns:
        features = pd.get_dummies(features, columns=["mode"])
    return features


def _write_json(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def build_parser():
    parser = argparse.ArgumentParser(description="Train ML cutoff predictor.")
    parser.add_argument(
        "--input",
        type=str,
        default=str(CSV_DIR / "cutoff_training_dataset.csv"),
    )
    parser.add_argument(
        "--model-output",
        type=str,
        default=str(MODELS_DIR / "cutoff_predictor.joblib"),
    )
    parser.add_argument(
        "--feature-columns-output",
        type=str,
        default=str(MODELS_DIR / "cutoff_feature_columns.json"),
    )
    parser.add_argument(
        "--metrics-output",
        type=str,
        default=str(CSV_DIR / "cutoff_model_metrics.csv"),
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    return parser


def main():
    args = build_parser().parse_args()
    ensure_dirs()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = PROJECT_ROOT / input_path

    df = pd.read_csv(input_path)
    if "target_cutoff" not in df.columns:
        raise ValueError("Training dataset must contain target_cutoff")

    y = df["target_cutoff"].astype(float)
    x = _prepare_features(df)
    feature_columns = list(x.columns)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        min_samples_leaf=2,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    errors = predictions - y_test.to_numpy()
    overpredictions = np.maximum(errors, 0.0)
    underprediction_count = int(np.count_nonzero(errors < 0.0))
    underprediction_percent = float(100.0 * underprediction_count / len(y_test))

    metrics = {
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": float(mean_squared_error(y_test, predictions, squared=False)),
        "underprediction_count": underprediction_count,
        "underprediction_percent": underprediction_percent,
        "mean_overprediction": float(np.mean(overpredictions)),
    }

    model_output = Path(args.model_output)
    if not model_output.is_absolute():
        model_output = PROJECT_ROOT / model_output
    model_output.parent.mkdir(parents=True, exist_ok=True)

    artifact = {
        "model": model,
        "feature_columns": feature_columns,
        "metrics": metrics,
    }
    joblib.dump(artifact, model_output)

    columns_path = Path(args.feature_columns_output)
    if not columns_path.is_absolute():
        columns_path = PROJECT_ROOT / columns_path
    _write_json({"feature_columns": feature_columns}, columns_path)

    metrics_path = Path(args.metrics_output)
    if not metrics_path.is_absolute():
        metrics_path = PROJECT_ROOT / metrics_path
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)

    print("Model saved to {}".format(model_output))
    print("Feature columns saved to {}".format(columns_path))
    print("Metrics saved to {}".format(metrics_path))
    print("MAE: {:.4f}".format(metrics["mae"]))
    print("RMSE: {:.4f}".format(metrics["rmse"]))


if __name__ == "__main__":
    main()
