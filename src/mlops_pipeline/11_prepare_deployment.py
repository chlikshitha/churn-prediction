import json
import hashlib
import os
import tempfile
from pathlib import Path
from datetime import datetime, timezone

import joblib
import mlflow
import mlflow.sklearn
import sklearn
import yaml
from mlflow import MlflowClient


def load_config():
    project_root = Path(__file__).resolve().parents[2]

    with open(project_root / "config.yaml", "r", encoding="utf-8") as file:
        return project_root, yaml.safe_load(file)


def get_production_model(client, model_name):
    versions = client.search_model_versions(f"name='{model_name}'")

    production_versions = [
        version for version in versions
        if version.current_stage == "Production"
    ]

    if not production_versions:
        raise RuntimeError(f"No Production version found for '{model_name}'.")

    if len(production_versions) > 1:
        raise RuntimeError(
            f"Multiple Production versions found for '{model_name}'."
        )

    return production_versions[0]


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as artifact:
        for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_deployment():
    project_root, config = load_config()

    deployment_dir = project_root / "deployment"

    deployment_dir.mkdir(parents=True, exist_ok=True)

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or config["mlflow"].get("tracking_uri")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    client = MlflowClient()

    model_name = config["registry"]["model_name"]
    model_version = get_production_model(client, model_name)

    model_name = model_version.name
    version = model_version.version
    run_id = model_version.run_id
    model_source = model_version.source

    print("[INFO] Production model found")
    print(f"[INFO] Model name : {model_name}")
    print(f"[INFO] Version    : {version}")
    print(f"[INFO] Run ID     : {run_id}")
    print(f"[INFO] Source     : {model_source}")

    print("[INFO] Downloading Production model...")

    model_uri = f"models:/{model_name}/{version}"
    model = mlflow.sklearn.load_model(model_uri)

    print("[INFO] Downloading matching preprocessor...")

    preprocessor_path = client.download_artifacts(
        run_id,
        "preprocessing_pipeline/preprocessor.pkl",
        dst_path=str(deployment_dir),
    )

    downloaded_preprocessor = Path(preprocessor_path)

    final_preprocessor_path = deployment_dir / "preprocessor.pkl"
    print("[INFO] Verifying deployment artifacts before replacing the active bundle...")
    loaded_preprocessor = joblib.load(downloaded_preprocessor)

    if loaded_preprocessor is None:
        raise RuntimeError("Preprocessor verification failed.")

    model_features = getattr(model, "n_features_in_", None)
    try:
        preprocessor_output_features = len(loaded_preprocessor.get_feature_names_out())
    except (AttributeError, ValueError) as exc:
        raise RuntimeError("Unable to determine deployment preprocessor output feature count") from exc
    if model_features is not None and model_features != preprocessor_output_features:
        raise ValueError(
            f"Production model expects {model_features} features, matching preprocessor emits "
            f"{preprocessor_output_features}; deployment bundle not updated."
        )

    model_path = deployment_dir / "model.pkl"
    with tempfile.NamedTemporaryFile(dir=deployment_dir, suffix=".model.tmp", delete=False) as temp_model_file:
        model_temp_path = Path(temp_model_file.name)
    with tempfile.NamedTemporaryFile(dir=deployment_dir, suffix=".preprocessor.tmp", delete=False) as temp_preprocessor_file:
        preprocessor_temp_path = Path(temp_preprocessor_file.name)
    try:
        joblib.dump(model, model_temp_path)
        joblib.dump(loaded_preprocessor, preprocessor_temp_path)
        loaded_model = joblib.load(model_temp_path)
        verified_preprocessor = joblib.load(preprocessor_temp_path)
        if getattr(loaded_model, "n_features_in_", model_features) != model_features:
            raise RuntimeError("Serialized deployment model feature count changed unexpectedly")
        if len(verified_preprocessor.get_feature_names_out()) != preprocessor_output_features:
            raise RuntimeError("Serialized preprocessor feature count changed unexpectedly")
        model_temp_path.replace(model_path)
        preprocessor_temp_path.replace(final_preprocessor_path)
    finally:
        model_temp_path.unlink(missing_ok=True)
        preprocessor_temp_path.unlink(missing_ok=True)

    print(f"[SUCCESS] Model saved to {model_path}")
    print(f"[SUCCESS] Preprocessor saved to {final_preprocessor_path}")

    model_sha256 = sha256_file(model_path)
    preprocessor_sha256 = sha256_file(final_preprocessor_path)
    bundle_fingerprint = hashlib.sha256(
        f"{model_sha256}:{preprocessor_sha256}:{model_name}:{version}:{run_id}".encode("utf-8")
    ).hexdigest()

    metadata = {
        "model_name": model_name,
        "model_version": str(version),
        "model_stage": model_version.current_stage,
        "run_id": run_id,
        "model_source": model_source,
        "model_id": model_source.split("/")[-1],
        "model_uri": model_uri,
        "model_sha256": model_sha256,
        "preprocessor_sha256": preprocessor_sha256,
        "bundle_fingerprint": bundle_fingerprint,
        "model_feature_count": model_features,
        "preprocessor_output_feature_count": preprocessor_output_features,
        "preprocessor_artifact": "preprocessing_pipeline/preprocessor.pkl",
        "source": "MLflow",
        "provenance_status": "verified",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sklearn_version": sklearn.__version__,
        "mlflow_version": mlflow.__version__,
    }

    metadata_path = deployment_dir / "model_metadata.json"

    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    print(f"[SUCCESS] Metadata saved to {metadata_path}")

    print("\n[INFO] Deployment bundle:")
    print("deployment/")
    print("├── model.pkl")
    print("├── preprocessor.pkl")
    print("└── model_metadata.json")

    print("\n[SUCCESS] Deployment preparation completed.")


if __name__ == "__main__":
    prepare_deployment()
