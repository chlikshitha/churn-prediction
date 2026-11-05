"""Apply the configured quality gate to the current evaluation report."""
import json
import os
from pathlib import Path
import sys

import mlflow
from mlflow.tracking import MlflowClient
import yaml

ROOT = Path(__file__).resolve().parents[2]


def run_quality_gate():
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or config["mlflow"].get("tracking_uri")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    report_path = ROOT / config["paths"]["evaluation_report"]
    output_path = ROOT / config["paths"]["quality_gate_report"]
    metric_name = config["quality_gate"]["metric"]
    minimum_value = float(config["quality_gate"]["minimum_value"])

    if not report_path.exists():
        raise FileNotFoundError(f"Evaluation report not found: {report_path}; run 06_model_evaluation.py first")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not report.get("model", {}).get("version") or not report.get("model", {}).get("run_id"):
        raise ValueError("Evaluation report is missing model version/run provenance; refusing a potentially stale gate.")
    report_gate = report.get("quality_gate_config", {})
    if report_gate.get("metric") != metric_name or float(report_gate.get("minimum_value", -1)) != minimum_value:
        raise ValueError("Evaluation report quality-gate settings do not match current config.yaml; regenerate evaluation.")
    if report["model"].get("name") != config["registry"]["model_name"]:
        raise ValueError("Evaluation report model name does not match configured registry model")
    versions = MlflowClient().search_model_versions(f"name='{config['registry']['model_name']}'")
    if not versions:
        raise RuntimeError("No registered model versions are available for quality-gate provenance verification")
    latest = max(versions, key=lambda item: int(item.version))
    if str(latest.version) != str(report["model"].get("version")) or latest.run_id != report["model"].get("run_id"):
        raise ValueError("Evaluation report is stale: it does not reference the latest registered model version/run")
    if metric_name not in report.get("metrics", {}):
        raise KeyError(f"Configured quality metric {metric_name!r} is missing from evaluation report.")

    actual_value = float(report["metrics"][metric_name])
    passed = actual_value >= minimum_value
    gate_report = {
        "stage": "Model Quality Gate",
        "metric": metric_name,
        "minimum_value": minimum_value,
        "actual_value": actual_value,
        "status": "PASSED" if passed else "FAILED",
        "model": report["model"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(gate_report, indent=2) + "\n", encoding="utf-8")
    print(f"[INFO] Quality Gate: {metric_name}={actual_value:.6f} | required>={minimum_value:.6f}")
    print(f"[{'SUCCESS' if passed else 'ERROR'}] Model Quality Gate {'PASSED' if passed else 'FAILED'}.")
    return passed


if __name__ == "__main__":
    try:
        passed = run_quality_gate()
    except Exception as exc:
        print(f"[ERROR] Quality gate failed: {exc}")
        sys.exit(1)
    sys.exit(0 if passed else 1)
