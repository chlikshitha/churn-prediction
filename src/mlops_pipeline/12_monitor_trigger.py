import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DRIFT_SCRIPTS = [
    PROJECT_ROOT / "src/mlops_pipeline/12_data_drift.py",
    PROJECT_ROOT / "src/mlops_pipeline/12_prediction_drift.py",
    PROJECT_ROOT / "src/mlops_pipeline/12_concept_drift.py",
    PROJECT_ROOT / "src/mlops_pipeline/12_drift_analysis.py",
]
DRIFT_REPORT = PROJECT_ROOT / "outputs/drift/drift_analysis_report.csv"
NEW_DATA_PATH = PROJECT_ROOT / "data/raw/new_churn.csv"
PIPELINE_SCRIPT = PROJECT_ROOT / "src/mlops_pipeline/churn_prediction_pipeline.py"
TRIGGER_REPORT = PROJECT_ROOT / "outputs/monitoring/monitor_trigger_report.json"
DATA_LOAD_REPORT = PROJECT_ROOT / "artifacts/data_load_report.json"
LAST_DRIFT_FAILURE = None


def run_drift_analysis():
    global LAST_DRIFT_FAILURE
    LAST_DRIFT_FAILURE = None
    # Invalidate the prior combined report before beginning a fresh run so a
    # later-stage failure cannot leave a stale report looking current.
    DRIFT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{
        "category": "run",
        "metric": "validation",
        "value": None,
        "status": "INVALIDATED",
        "details": "A monitoring run is in progress; this report is valid only after all drift stages pass.",
    }]).to_csv(DRIFT_REPORT, index=False)
    for script in DRIFT_SCRIPTS:
        print(f"[INFO] Running {script}...")
        if not script.is_file():
            LAST_DRIFT_FAILURE = {"script": str(script), "error": "script file is missing"}
            return False
        result = subprocess.run(
            [sys.executable, str(script)], cwd=PROJECT_ROOT,
            capture_output=True, text=True,
        )
        if result.stdout:
            print(result.stdout, end="")
        if result.returncode != 0:
            print(f"[ERROR] Drift stage failed: {script}")
            LAST_DRIFT_FAILURE = {
                "script": str(script.relative_to(PROJECT_ROOT)),
                "exit_code": result.returncode,
                "stderr": (result.stderr or "")[-2000:],
            }
            return False
    return True


def detect_drift():
    if not DRIFT_REPORT.exists():
        raise FileNotFoundError(f"Drift report not found: {DRIFT_REPORT}")

    df = pd.read_csv(DRIFT_REPORT)
    if df.empty or "status" not in df.columns:
        raise ValueError(f"Combined drift report is empty or missing 'status': {DRIFT_REPORT}")
    return df["status"].astype(str).str.strip().str.upper().eq("DRIFT").any()


def get_data_load_status():
    if not DATA_LOAD_REPORT.exists():
        return None

    with DATA_LOAD_REPORT.open("r", encoding="utf-8") as f:
        report = json.load(f)

    return report.get("status")


def save_trigger_report(report):
    TRIGGER_REPORT.parent.mkdir(parents=True, exist_ok=True)

    with TRIGGER_REPORT.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, default=str)


def run_monitor_trigger():
    print("=" * 70)
    print("LAB 13 MONITORING + RETRAINING TRIGGER")
    print("=" * 70)

    drift_analysis_passed = run_drift_analysis()

    if not drift_analysis_passed:
        report = {
            "status": "drift_analysis_failed",
            "failure": LAST_DRIFT_FAILURE,
            "drift_detected": False,
            "new_data_present": NEW_DATA_PATH.exists(),
            "pipeline_called": False,
            "new_data_ingested": False,
            "retraining_executed": False,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        save_trigger_report(report)
        print("[ERROR] Drift analysis failed.")
        return False

    try:
        drift_detected = detect_drift()
    except Exception as exc:
        save_trigger_report({
            "status": "drift_analysis_failed",
            "failure": {"error": str(exc)},
            "drift_detected": False,
            "new_data_present": NEW_DATA_PATH.exists(),
            "pipeline_called": False,
            "new_data_ingested": False,
            "retraining_executed": False,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        print(f"[ERROR] Combined drift report validation failed: {exc}")
        return False
    new_data_present = NEW_DATA_PATH.exists()

    print(f"[INFO] Drift detected: {drift_detected}")
    print(f"[INFO] New dataset present: {new_data_present}")

    if not drift_detected:
        report = {
            "status": "no_drift",
            "drift_detected": False,
            "new_data_present": new_data_present,
            "pipeline_called": False,
            "new_data_ingested": False,
            "retraining_executed": False,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        save_trigger_report(report)
        print("[INFO] No drift detected. Waiting for future monitoring cycles.")
        return True

    if not new_data_present:
        report = {
            "status": "drift_waiting_for_data",
            "drift_detected": True,
            "new_data_present": False,
            "pipeline_called": False,
            "new_data_ingested": False,
            "retraining_executed": False,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        save_trigger_report(report)
        print("[INFO] Drift detected, but no new dataset is available.")
        print("[INFO] Waiting for new_churn.csv.")
        return True

    print("[INFO] Drift detected and new dataset is available.")
    print("[INFO] Triggering existing MLOps pipeline...")

    pipeline_result = subprocess.run(
        [sys.executable, str(PIPELINE_SCRIPT)], cwd=PROJECT_ROOT,
        capture_output=True, text=True,
    )
    if pipeline_result.stdout:
        print(pipeline_result.stdout, end="")
    if pipeline_result.stderr:
        print(pipeline_result.stderr, file=sys.stderr, end="")

    data_load_status = get_data_load_status()
    new_data_ingested = data_load_status == "ingested"
    retraining_executed = new_data_ingested and pipeline_result.returncode == 0

    if retraining_executed:
        status = "retraining_completed"
    elif data_load_status == "unlabeled_prediction_data":
        status = "prediction_data_waiting_for_labels"
    elif data_load_status in {"already_ingested", "no_new_data"}:
        status = "no_new_data_retraining_skipped"
    else:
        status = "retraining_not_confirmed"

    report = {
        "status": status,
        "drift_detected": True,
        "new_data_present": True,
        "pipeline_called": True,
        "pipeline_exit_code": pipeline_result.returncode,
        "pipeline_status": "passed" if pipeline_result.returncode == 0 else "failed",
        "data_load_status": data_load_status,
        "new_data_ingested": new_data_ingested,
        "retraining_executed": retraining_executed,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    save_trigger_report(report)

    if pipeline_result.returncode != 0:
        print("[ERROR] Existing MLOps pipeline failed.")
        return False

    if new_data_ingested:
        print("[SUCCESS] New data was ingested and retraining completed.")
    elif data_load_status == "unlabeled_prediction_data":
        print("[INFO] Incoming batch is prediction-only; supervised retraining awaits labels.")
    else:
        print("[INFO] Pipeline completed, but no new data was ingested. Retraining was skipped.")

    return True


if __name__ == "__main__":
    passed = run_monitor_trigger()
    sys.exit(0 if passed else 1)
