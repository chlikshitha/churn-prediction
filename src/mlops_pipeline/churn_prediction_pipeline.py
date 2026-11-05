import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


STAGES = [
    ("Data Loading", "src/mlops_pipeline/01_load_data.py"),
    ("Raw Data Validation", "src/mlops_pipeline/02_validate_data.py"),
    ("Preprocessing", "src/mlops_pipeline/03_preprocessing.py"),
    ("Processed Output Validation", "src/mlops_pipeline/04_validate_outputs.py"),
    ("Training + MLflow + Model Registry", "src/mlops_pipeline/05_train_registry.py"),
    ("Model Evaluation", "src/mlops_pipeline/06_model_evaluation.py"),
    ("Model Quality Gate", "src/mlops_pipeline/07_quality_gate.py"),
    ("Automated Model Lifecycle", "src/mlops_pipeline/08_automate_lifecycle.py"),
    ("Deployment Update", "src/mlops_pipeline/deploy_if_promoted.py"),
    ("Production Inference", "src/mlops_pipeline/09_inference.py"),
]
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_data_load_report():
    report_path = PROJECT_ROOT / "artifacts/data_load_report.json"

    if not os.path.exists(report_path):
        raise FileNotFoundError(f"Data load report not found: {report_path}")

    with report_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def run_pipeline(force=False):
    print("[INFO] ==================================================")
    print("[INFO] Starting Churn Prediction MLOps Pipeline")
    print("[INFO] ==================================================")

    started_at = datetime.now(timezone.utc).isoformat()
    run_log = {
        "pipeline": "Churn Prediction MLOps",
        "started_at": started_at,
        "force": force,
        "stage_count": len(STAGES),
        "implementation_stages": [
            {"stage": name, "script": script} for name, script in STAGES
        ],
        "stages": [],
    }

    for stage_index, (stage_name, script_path) in enumerate(STAGES):
        print("\n" + "=" * 70)
        print(f"RUNNING: {stage_name}")
        print("=" * 70)

        try:
            result = subprocess.run(
                [sys.executable, str(PROJECT_ROOT / script_path)],
                cwd=PROJECT_ROOT,
            )
            exit_code = result.returncode
        except OSError as exc:
            exit_code = 127
            run_log["stages"].append({
                "stage": stage_name, "script": script_path,
                "status": "FAILED", "exit_code": exit_code, "error": str(exc),
            })
            run_log["overall_status"] = "FAILED"
            break
        status = "PASSED" if exit_code == 0 else "FAILED"

        run_log["stages"].append({
            "stage": stage_name, "script": script_path,
            "status": status, "exit_code": exit_code,
        })

        print(f"--> {stage_name}: {status}")

        if exit_code != 0:
            print("[HALT] Pipeline stopped because this stage failed.")
            run_log["overall_status"] = "FAILED"
            break

        if stage_index == 0 and not force:
            report = load_data_load_report()
            data_status = report.get("status")

            if data_status in {"already_ingested", "no_new_data", "unlabeled_prediction_data"}:
                print(f"[INFO] Data load status: {data_status}")
                if data_status == "unlabeled_prediction_data":
                    print("[INFO] Unlabeled prediction rows are not supervised training data.")
                    run_log["overall_status"] = "WAITING_FOR_LABELS"
                else:
                    print("[INFO] No new labelled dataset was ingested.")
                    run_log["overall_status"] = "NO_NEW_DATA"
                print("[INFO] Training and downstream lifecycle stages will not run.")
                break

            if data_status != "ingested":
                print(f"[ERROR] Unexpected data load status: {data_status}")
                print("[HALT] Pipeline stopped because the ingestion state is unknown.")
                run_log["overall_status"] = "FAILED"
                break

            print("[INFO] New dataset was successfully ingested.")
            print("[INFO] Continuing with the ML pipeline.")

    all_passed = len(run_log["stages"]) == len(STAGES) and all(stage["status"] == "PASSED" for stage in run_log["stages"])

    if all_passed:
        run_log["overall_status"] = "PASSED"

    run_log["finished_at"] = datetime.now(timezone.utc).isoformat()
    log_path = PROJECT_ROOT / "logs" / "churn_prediction_pipeline_run_log.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("w", encoding="utf-8") as f:
        json.dump(run_log, f, indent=4)

    print("\n" + "=" * 70)

    if run_log["overall_status"] == "PASSED":
        print("[SUCCESS] Churn Prediction MLOps Pipeline completed successfully.")
    elif run_log["overall_status"] == "NO_NEW_DATA":
        print("[INFO] No new dataset available. Pipeline finished without retraining.")
    elif run_log["overall_status"] == "WAITING_FOR_LABELS":
        print("[INFO] Prediction data has no labels. Pipeline is waiting for labelled training data.")
    else:
        print("[ERROR] Churn Prediction MLOps Pipeline failed.")

    print(f"[INFO] Run log: {log_path}")
    print("=" * 70)

    return run_log["overall_status"] in {"PASSED", "NO_NEW_DATA", "WAITING_FOR_LABELS"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Churn Prediction MLOps pipeline.")
    parser.add_argument("--force", action="store_true", help="Run all ML stages even when no new dataset was ingested.")
    args = parser.parse_args()

    passed = run_pipeline(force=args.force)
    sys.exit(0 if passed else 1)

# import json
# import os
# import subprocess
# import sys


# STAGES = [
#     ("Data Loading", "src/mlops_pipeline/01_load_data.py"),
#     ("Raw Data Validation", "src/mlops_pipeline/02_validate_data.py"),
#     ("Preprocessing", "src/mlops_pipeline/03_preprocessing.py"),
#     ("Processed Output Validation", "src/mlops_pipeline/04_validate_outputs.py"),
#     ("Training + MLflow + Model Registry", "src/mlops_pipeline/05_train_registry.py"),
#     ("Model Evaluation", "src/mlops_pipeline/06_model_evaluation.py"),
#     ("Model Quality Gate", "src/mlops_pipeline/07_quality_gate.py"),
#     ("Automated Model Lifecycle", "src/mlops_pipeline/08_automate_lifecycle.py"),
#     ("Production Inference", "src/mlops_pipeline/09_inference.py"),
# ]


# def run_pipeline():
#     print("[INFO] ==================================================")
#     print("[INFO] Starting Churn Prediction MLOps Pipeline")
#     print("[INFO] ==================================================")

#     run_log = {"pipeline": "Churn Prediction MLOps", "stages": []}

#     for stage_name, script_path in STAGES:
#         print("\n" + "=" * 70)
#         print(f"RUNNING: {stage_name}")
#         print("=" * 70)

#         result = subprocess.run([sys.executable, script_path])
#         status = "PASSED" if result.returncode == 0 else "FAILED"

#         run_log["stages"].append(
#             {
#                 "stage": stage_name,
#                 "status": status,
#                 "exit_code": result.returncode,
#             }
#         )

#         print(f"--> {stage_name}: {status}")

#         if result.returncode != 0:
#             print("[HALT] Pipeline stopped because this stage failed.")
#             break

#     all_passed = len(run_log["stages"]) == len(STAGES) and all(
#         stage["status"] == "PASSED" for stage in run_log["stages"]
#     )
#     run_log["overall_status"] = "PASSED" if all_passed else "FAILED"

#     os.makedirs("logs", exist_ok=True)
#     log_path = "logs/churn_prediction_pipeline_run_log.json"
#     with open(log_path, "w", encoding="utf-8") as f:
#         json.dump(run_log, f, indent=4)

#     print("\n" + "=" * 70)
#     if all_passed:
#         print("[SUCCESS] Churn Prediction MLOps Pipeline completed successfully.")
#     else:
#         print("[ERROR] Churn Prediction MLOps Pipeline failed.")
#     print(f"[INFO] Run log: {log_path}")
#     print("=" * 70)

#     return all_passed


# if __name__ == "__main__":
#     passed = run_pipeline()
#     sys.exit(0 if passed else 1)
