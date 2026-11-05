import json
import os
import sys
from pathlib import Path

import mlflow
import yaml
from mlflow.tracking import MlflowClient

ROOT = Path(__file__).resolve().parents[2]


def configure_mlflow(config):
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or config["mlflow"].get("tracking_uri")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)


def save_lifecycle_report(report):
    path = ROOT / "reports" / "lifecycle_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def automate_model_lifecycle():
    print("[INFO] Starting Automated Model Lifecycle Manager...")

    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))

    configure_mlflow(config)
    client = MlflowClient()
    model_name = config["registry"]["model_name"]
    metric_to_optimize = config["registry"]["decision_metric"]

    report = {
        "status": "success",
        "model_name": model_name,
        "decision_metric": metric_to_optimize,
        "candidate_version": None,
        "candidate_run_id": None,
        "candidate_metric": None,
        "production_version_before": None,
        "production_metric_before": None,
        "promoted": False,
        "decision": None,
        "production_version_after": None,
        "candidate_model_uri": None,
        "quality_gate_status": None,
    }

    try:
        evaluation_path = ROOT / config["paths"]["evaluation_report"]
        gate_path = ROOT / config["paths"]["quality_gate_report"]
        if not evaluation_path.is_file() or not gate_path.is_file():
            raise FileNotFoundError("Evaluation and quality-gate reports are required before lifecycle decisions")
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        candidate_identity = evaluation.get("model", {})
        candidate_version = str(candidate_identity.get("version", ""))
        candidate_run_id = candidate_identity.get("run_id")
        if not candidate_version or not candidate_run_id:
            raise ValueError("Evaluation report is missing registered candidate version/run provenance")
        if gate.get("status") != "PASSED":
            raise RuntimeError("Candidate did not pass the configured quality gate")
        if gate.get("metric") != metric_to_optimize:
            raise ValueError("Lifecycle decision metric and quality-gate report metric do not match")
        if str(gate.get("model", {}).get("version")) != candidate_version or gate.get("model", {}).get("run_id") != candidate_run_id:
            raise ValueError("Evaluation and quality-gate reports refer to different model versions/runs")

        all_versions = client.search_model_versions(f"name='{model_name}'")
        candidate = next((mv for mv in all_versions if str(mv.version) == candidate_version), None)
        if candidate is None or candidate.run_id != candidate_run_id:
            raise ValueError("Evaluation report candidate does not match the registered model version/run")
        report["candidate_version"] = int(candidate.version)
        report["candidate_run_id"] = candidate.run_id
        report["candidate_model_uri"] = f"models:/{model_name}/{candidate.version}"
        report["quality_gate_status"] = gate["status"]

        if candidate.current_stage not in {"None", "Staging"}:
            raise RuntimeError(f"Evaluated candidate is already in stage {candidate.current_stage!r}; refusing to transition it")
        if candidate.current_stage == "None":
            client.transition_model_version_stage(
                name=model_name, version=candidate.version, stage="Staging",
                archive_existing_versions=False,
            )
            candidate = next(
                mv for mv in client.search_model_versions(f"name='{model_name}'")
                if str(mv.version) == candidate_version
            )
        production_models = [mv for mv in all_versions if mv.current_stage == "Production"]
        if len(production_models) > 1:
            raise RuntimeError(f"Multiple Production versions found for {model_name!r}")
        production_model = production_models[0] if production_models else None

        if production_model:
            production_run = client.get_run(production_model.run_id)
            production_score = production_run.data.metrics.get(metric_to_optimize, 0.0)
            report["production_version_before"] = int(production_model.version)
            report["production_metric_before"] = production_score
        else:
            production_score = None

        candidate_run = client.get_run(candidate.run_id)
        candidate_score = candidate_run.data.metrics.get(metric_to_optimize)
        if candidate_score is None:
            raise ValueError(f"Candidate run lacks configured decision metric {metric_to_optimize!r}")
        gate_score = float(gate["actual_value"])
        if abs(float(candidate_score) - gate_score) > 1e-6:
            raise ValueError(
                f"Registry metric {candidate_score} does not match evaluated gate metric {gate_score}"
            )
        if gate_score < float(config["quality_gate"]["minimum_value"]):
            raise RuntimeError("Quality-gate report does not satisfy current config.yaml threshold")
        report["candidate_metric"] = float(candidate_score)
        report["candidate_source"] = candidate.source
        print(f"[INFO] Evaluated Version {candidate.version} | {metric_to_optimize}={candidate_score:.6f}")

        if production_model is None:
            move_to_production = True
            print("[INFO] No Production model exists. Staging model will become Production.")
        else:
            move_to_production = candidate_score > production_score

            if move_to_production:
                print("[SUCCESS] Staging model is better than Production.")
            else:
                print("[INFO] Production model remains unchanged.")

        if move_to_production:
            client.transition_model_version_stage(
                name=model_name,
                version=candidate.version,
                stage="Production",
                archive_existing_versions=False
            )

            if production_model:
                client.transition_model_version_stage(
                    name=model_name,
                    version=production_model.version,
                    stage="Archived",
                    archive_existing_versions=False
                )

            report["promoted"] = True
            report["decision"] = "PROMOTED"
            report["production_version_after"] = int(candidate.version)

            print(f"[SUCCESS] Version {candidate.version} is now Production.")
        else:
            client.transition_model_version_stage(
                name=model_name,
                version=candidate.version,
                stage="Archived",
                archive_existing_versions=False
            )

            report["promoted"] = False
            report["decision"] = "ARCHIVED"
            report["production_version_after"] = report["production_version_before"]

            print(f"[INFO] Version {candidate.version} archived because it did not improve.")

        save_lifecycle_report(report)
        print("[SUCCESS] Automated Model Lifecycle execution complete!")
        print("[INFO] Lifecycle report saved to reports/lifecycle_report.json")
        return True

    except Exception as exc:
        report["status"] = "failed"
        report["decision"] = "ERROR"
        report["error"] = str(exc)
        save_lifecycle_report(report)
        print(f"[ERROR] Failed to automate model lifecycle: {exc}")
        return False


if __name__ == "__main__":
    passed = automate_model_lifecycle()
    sys.exit(0 if passed else 1)
