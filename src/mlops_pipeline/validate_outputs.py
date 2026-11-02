import os
import json
import numpy as np
import sys


def validate_arrays(X_train, X_test, y_train, y_test, require_raw_splits=True):
    errors = []
    if not np.isfinite(X_train).all():
        errors.append("Non-finite values detected in X_train after preprocessing.")
    if not np.isfinite(X_test).all():
        errors.append("Non-finite values detected in X_test after preprocessing.")
    if X_train.shape[1] != X_test.shape[1]:
        errors.append(f"Feature mismatch: X_train has {X_train.shape[1]} cols, X_test has {X_test.shape[1]} cols.")
    if X_train.shape[0] != y_train.shape[0]:
        errors.append("Row mismatch between X_train and y_train.")
    if X_test.shape[0] != y_test.shape[0]:
        errors.append("Row mismatch between X_test and y_test.")
    if y_train.ndim != 1 or y_test.ndim != 1:
        errors.append("Target arrays must be one-dimensional.")
    if not set(np.unique(y_train)).issubset({0, 1}) or not set(np.unique(y_test)).issubset({0, 1}):
        errors.append("Target arrays must contain only 0 and 1.")
    if require_raw_splits:
        for split in ["data/processed/X_train_raw.csv", "data/processed/X_test_raw.csv"]:
            if not os.path.exists(split):
                errors.append(f"Required raw split artifact missing: {split}")
    return errors


def validate_preprocessing_outputs():
    print("[INFO] Validating Preprocessing Outputs...")

    try:
        X_train = np.load("data/processed/X_train_final.npy")
        X_test = np.load("data/processed/X_test_final.npy")
        y_train = np.load("data/processed/y_train.npy")
        y_test = np.load("data/processed/y_test.npy")
    except FileNotFoundError as exc:
        print(f"[ERROR] Could not find processed data: {exc}")
        return False

    errors = validate_arrays(X_train, X_test, y_train, y_test)

    report = {
        "validation_status": "PASSED" if not errors else "FAILED",
        "matrix_dimensions": {
            "X_train_shape": list(X_train.shape),
            "X_test_shape": list(X_test.shape),
            "y_train_shape": list(y_train.shape),
            "y_test_shape": list(y_test.shape),
        },
        "data_quality": {
            "missing_values_X_train": int(np.isnan(X_train).sum()),
            "missing_values_X_test": int(np.isnan(X_test).sum()),
        },
        "errors": errors,
    }

    os.makedirs("artifacts", exist_ok=True)
    report_path = "artifacts/preprocessing_summary_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    if errors:
        print("[ERROR] Output Validation FAILED!")
        for error in errors:
            print(f"  - {error}")
        return False

    print("[SUCCESS] Output Validation PASSED.")
    print(f"[INFO] Summary report saved to {report_path}")
    return True


if __name__ == "__main__":
    passed = validate_preprocessing_outputs()
    sys.exit(0 if passed else 1)
