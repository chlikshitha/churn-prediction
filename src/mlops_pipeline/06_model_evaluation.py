"""Evaluate the newest registered candidate against the configured holdout."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

import mlflow
import mlflow.sklearn
import numpy as np
from mlflow.tracking import MlflowClient
import yaml
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[2]


def run_evaluation():
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or config["mlflow"].get("tracking_uri")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    client = MlflowClient()
    versions = client.search_model_versions(f"name='{config['registry']['model_name']}'")
    if not versions:
        raise RuntimeError(f"No registered versions found for {config['registry']['model_name']!r}")
    latest = max(versions, key=lambda item: int(item.version))
    model_uri = f"models:/{latest.name}/{latest.version}"
    model = mlflow.sklearn.load_model(model_uri)

    processed_dir = ROOT / config["data"]["processed_dir"]
    X_test = np.load(processed_dir / "X_test_final.npy")
    y_test = np.load(processed_dir / "y_test.npy").astype(int)
    if len(X_test) != len(y_test):
        raise ValueError(f"Holdout feature/label row mismatch: {len(X_test)} != {len(y_test)}")
    expected_features = getattr(model, "n_features_in_", None)
    if expected_features is not None and X_test.shape[1] != expected_features:
        raise ValueError(
            f"Latest registered model {latest.version} expects {expected_features} features; "
            f"configured holdout contains {X_test.shape[1]}. Refusing evaluation."
        )

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1_score": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
    }
    quality_cfg = config["quality_gate"]
    run = client.get_run(latest.run_id)
    report = {
        "stage": "Model evaluation",
        "evaluated_on": "configured held-out test split",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "test_rows": int(len(y_test)),
        "feature_count": int(X_test.shape[1]),
        "model": {
            "name": latest.name,
            "version": str(latest.version),
            "run_id": latest.run_id,
            "uri": model_uri,
            "source": latest.source,
            "registered_stage": latest.current_stage,
            "run_name": run.info.run_name,
        },
        "quality_gate_config": {
            "metric": quality_cfg["metric"],
            "minimum_value": float(quality_cfg["minimum_value"]),
        },
        "metrics": metrics,
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
    }

    report_path = ROOT / config["paths"]["evaluation_report"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[SUCCESS] Evaluated {model_uri} on {len(y_test)} holdout rows.")
    for name, value in metrics.items():
        print(f"{name:10}: {value:.6f}")
    print(f"[INFO] Evaluation report: {report_path}")
    return report


if __name__ == "__main__":
    try:
        run_evaluation()
    except Exception as exc:
        print(f"[ERROR] Model evaluation failed: {exc}")
        sys.exit(1)
