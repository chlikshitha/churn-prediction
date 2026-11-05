import json
import hashlib
import subprocess
import sys
from pathlib import Path

import joblib
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEPLOYMENT_DIR = PROJECT_ROOT / "deployment"
METADATA_PATH = DEPLOYMENT_DIR / "model_metadata.json"
K8S_DEPLOYMENT_PATH = PROJECT_ROOT / "k8s" / "deployment.yaml"
REPORT_PATH = PROJECT_ROOT / "reports" / "deployment_automation_report.json"

IMAGE_NAME = "churn-prediction-api"


def run_command(command, capture_output=False):
    print(f"[INFO] Running: {' '.join(command)}")
    result = subprocess.run(command, cwd=PROJECT_ROOT, text=True, capture_output=capture_output)

    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(command)}")

    return result


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as artifact:
        for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_metadata():
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Deployment metadata not found: {METADATA_PATH}")

    with METADATA_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_kubernetes_config():
    if not K8S_DEPLOYMENT_PATH.exists():
        raise FileNotFoundError(f"Kubernetes deployment file not found: {K8S_DEPLOYMENT_PATH}")

    with K8S_DEPLOYMENT_PATH.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    deployment_name = config["metadata"]["name"]
    app_label = config["metadata"]["labels"]["app"]
    container_name = config["spec"]["template"]["spec"]["containers"][0]["name"]

    return deployment_name, app_label, container_name


def save_report(report):
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with REPORT_PATH.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)


def deploy_production():
    print("=" * 70)
    print("PRODUCTION DEPLOYMENT")
    print("=" * 70)

    metadata = load_metadata()
    deployment_name, app_label, container_name = load_kubernetes_config()

    model_name = metadata["model_name"]
    model_version = metadata["model_version"]
    model_stage = metadata["model_stage"]
    run_id = metadata["run_id"]

    if metadata.get("provenance_status") != "verified" or not model_version or not run_id:
        raise RuntimeError("Deployment blocked: model version/run provenance is not verified.")
    if model_stage != "Production":
        raise RuntimeError(f"Deployment blocked. Model stage is '{model_stage}', not 'Production'.")

    model_path = DEPLOYMENT_DIR / "model.pkl"
    preprocessor_path = DEPLOYMENT_DIR / "preprocessor.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")

    if not preprocessor_path.exists():
        raise FileNotFoundError(f"Preprocessor artifact not found: {preprocessor_path}")

    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    model_features = getattr(model, "n_features_in_", None)
    try:
        preprocessor_features = len(preprocessor.get_feature_names_out())
    except (AttributeError, ValueError) as exc:
        raise ValueError("Cannot verify deployment preprocessor output feature count") from exc
    if model_features != preprocessor_features:
        raise ValueError(
            f"Deployment bundle mismatch: model expects {model_features} features, "
            f"preprocessor emits {preprocessor_features}"
        )
    if metadata.get("model_feature_count") not in (None, model_features):
        raise ValueError("Deployment model feature count does not match model metadata")
    if metadata.get("preprocessor_output_feature_count") not in (None, preprocessor_features):
        raise ValueError("Deployment preprocessor feature count does not match model metadata")

    model_hash = sha256_file(model_path)
    preprocessor_hash = sha256_file(preprocessor_path)
    computed_fingerprint = hashlib.sha256(
        f"{model_hash}:{preprocessor_hash}:{model_name}:{model_version}:{run_id}".encode("utf-8")
    ).hexdigest()
    fingerprint = metadata.get("bundle_fingerprint") or computed_fingerprint
    if fingerprint != computed_fingerprint:
        raise ValueError("Deployment bundle fingerprint does not match model metadata and artifact hashes")
    if metadata.get("model_sha256") not in (None, model_hash):
        raise ValueError("Deployment model checksum does not match model_metadata.json")
    if metadata.get("preprocessor_sha256") not in (None, preprocessor_hash):
        raise ValueError("Deployment preprocessor checksum does not match model_metadata.json")
    image_tag = f"{IMAGE_NAME}:model-v{model_version}-{fingerprint[:12]}"

    print(f"[INFO] Model        : {model_name}")
    print(f"[INFO] Model version: {model_version}")
    print(f"[INFO] Run ID       : {run_id}")
    print(f"[INFO] Docker image : {image_tag}")
    print(f"[INFO] Kubernetes   : {deployment_name}")
    print(f"[INFO] Container    : {container_name}")

    labels = {
        "com.churn.model-name": str(model_name),
        "com.churn.model-version": str(model_version),
        "com.churn.model-run-id": str(run_id),
        "com.churn.bundle-sha256": fingerprint,
    }
    inspect_format = "{{json .Config.Labels}}"
    inspected = subprocess.run(
        ["docker", "image", "inspect", "--format", inspect_format, image_tag],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
    )
    image_labels = {}
    if inspected.returncode == 0:
        try:
            image_labels = json.loads(inspected.stdout.strip()) or {}
        except json.JSONDecodeError:
            image_labels = {}
    image_matches = inspected.returncode == 0 and all(
        image_labels.get(key) == value for key, value in labels.items()
    )

    if image_matches:
        print(f"[INFO] Docker image bundle verified: {image_tag}")
    else:
        print("[INFO] Docker image is missing or its provenance labels do not match; building.")
        build_command = ["docker", "build", "-t", image_tag]
        for key, value in {
            "MODEL_NAME": model_name,
            "MODEL_VERSION": model_version,
            "MODEL_RUN_ID": run_id,
            "MODEL_BUNDLE_SHA256": fingerprint,
        }.items():
            build_command.extend(["--build-arg", f"{key}={value}"])
        build_command.append(".")
        run_command(build_command)

    context_result = run_command(["kubectl", "config", "current-context"], capture_output=True)
    kubernetes_context = context_result.stdout.strip()
    if not kubernetes_context:
        raise RuntimeError("kubectl returned an empty current context")

    print("[INFO] Updating Kubernetes deployment...")
    run_command([
        "kubectl",
        "set",
        "image",
        f"deployment/{deployment_name}",
        f"{container_name}={image_tag}"
    ])

    deployed_image = run_command([
        "kubectl", "get", "deployment", deployment_name,
        "-o", "jsonpath={.spec.template.spec.containers[0].image}",
    ], capture_output=True).stdout.strip()
    if deployed_image != image_tag:
        raise RuntimeError(f"Kubernetes deployment image mismatch: expected {image_tag}, got {deployed_image}")

    print("[INFO] Waiting for Kubernetes rollout...")
    run_command([
        "kubectl",
        "rollout",
        "status",
        f"deployment/{deployment_name}",
        "--timeout=120s"
    ])

    report = {
        "status": "success",
        "model_name": model_name,
        "model_version": str(model_version),
        "run_id": run_id,
        "docker_image": image_tag,
        "kubernetes_context": kubernetes_context,
        "kubernetes_deployment": deployment_name,
        "kubernetes_container": container_name,
        "app_label": app_label,
        "deployment_bundle": {
            "model": str(model_path.relative_to(PROJECT_ROOT)),
            "preprocessor": str(preprocessor_path.relative_to(PROJECT_ROOT)),
            "metadata": str(METADATA_PATH.relative_to(PROJECT_ROOT)),
            "fingerprint": fingerprint,
            "model_sha256": model_hash,
            "preprocessor_sha256": preprocessor_hash,
        },
        "image_labels_verified": True,
        "deployed_image_verified": True,
    }

    save_report(report)

    print("[SUCCESS] Docker image built.")
    print("[SUCCESS] Kubernetes deployment updated.")
    print("[SUCCESS] Kubernetes rollout completed.")
    print(f"[SUCCESS] Deployment report saved to {REPORT_PATH}")

    return True


if __name__ == "__main__":
    try:
        passed = deploy_production()
        sys.exit(0 if passed else 1)
    except Exception as exc:
        print(f"[ERROR] Production deployment failed: {exc}")
        sys.exit(1)
