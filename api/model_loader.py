from pathlib import Path
from typing import Any
import hashlib
import json
import joblib
import sklearn
import yaml


class ModelLoader:
    """
    Loads the ML model and preprocessing pipeline from the
    verified MLflow deployment bundle.
    """

    def __init__(self, config_path: str = "config.yaml"):
        self.project_root = Path(__file__).resolve().parents[1]
        self.config_path = self.project_root / config_path

        self.config: dict[str, Any] = {}

        self.model = None
        self.preprocessor = None

        self.model_uri = None
        self.model_name = None
        self.model_version = None
        self.run_id = None
        self.model_id = None
        self.metadata = {}

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as artifact:
            for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def load_config(self) -> None:
        """Load project configuration from config.yaml."""

        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {self.config_path}"
            )

        with open(self.config_path, "r", encoding="utf-8") as file:
            self.config = yaml.safe_load(file) or {}

    def load_deployment_bundle(self) -> None:
        """Load model, preprocessor, and metadata from deployment bundle."""

        deployment_dir = self.project_root / "deployment"

        model_path = deployment_dir / "model.pkl"
        preprocessor_path = deployment_dir / "preprocessor.pkl"
        metadata_path = deployment_dir / "model_metadata.json"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}"
            )

        if not preprocessor_path.exists():
            raise FileNotFoundError(
                f"Preprocessor not found: {preprocessor_path}"
            )

        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Model metadata not found: {metadata_path}"
            )

        print(f"[INFO] Loading deployment metadata from: {metadata_path}")

        with open(metadata_path, "r", encoding="utf-8") as file:
            self.metadata = json.load(file)

        self.model_name = self.metadata.get("model_name")
        self.model_version = self.metadata.get("model_version")
        self.run_id = self.metadata.get("run_id")
        self.model_id = self.metadata.get("model_id")
        self.model_uri = str(model_path)

        print(f"[INFO] Model name    : {self.model_name}")
        print(f"[INFO] Model version : {self.model_version}")
        print(f"[INFO] MLflow Run ID : {self.run_id}")
        print(f"[INFO] Model ID      : {self.model_id}")

        print(f"[INFO] Loading model from: {model_path}")

        self.model = joblib.load(model_path)

        print("[SUCCESS] Model loaded successfully.")

        print(f"[INFO] Loading preprocessor from: {preprocessor_path}")

        self.preprocessor = joblib.load(preprocessor_path)

        expected_model_hash = self.metadata.get("model_sha256")
        expected_preprocessor_hash = self.metadata.get("preprocessor_sha256")
        if expected_model_hash and self._sha256(model_path) != expected_model_hash:
            raise ValueError("Deployment model hash does not match model_metadata.json")
        if expected_preprocessor_hash and self._sha256(preprocessor_path) != expected_preprocessor_hash:
            raise ValueError("Deployment preprocessor hash does not match model_metadata.json")

        model_features = getattr(self.model, "n_features_in_", None)
        preprocessor_features = getattr(self.preprocessor, "n_features_in_", None)
        if model_features is not None and preprocessor_features is not None:
            try:
                output_features = len(self.preprocessor.get_feature_names_out())
            except (AttributeError, ValueError):
                output_features = None
            if output_features is not None and output_features != model_features:
                raise ValueError(
                    f"Deployment model expects {model_features} features, but preprocessor emits {output_features}"
                )
            if self.metadata.get("model_feature_count") not in (None, model_features):
                raise ValueError("Deployment model feature count does not match model_metadata.json")
            if self.metadata.get("preprocessor_output_feature_count") not in (None, output_features):
                raise ValueError("Deployment preprocessor feature count does not match model_metadata.json")
        if expected_model_hash and expected_preprocessor_hash and self.metadata.get("bundle_fingerprint"):
            expected_fingerprint = hashlib.sha256(
                (
                    f"{expected_model_hash}:{expected_preprocessor_hash}:"
                    f"{self.model_name}:{self.model_version}:{self.run_id}"
                ).encode("utf-8")
            ).hexdigest()
            if expected_fingerprint != self.metadata["bundle_fingerprint"]:
                raise ValueError("Deployment bundle fingerprint does not match model_metadata.json")
        self.model_uri = self.metadata.get("model_uri", str(model_path))

        print("[SUCCESS] Preprocessor loaded successfully.")
        self.metadata.setdefault("runtime_sklearn_version", sklearn.__version__)

    def load(self) -> None:
        """Load configuration and deployment artifacts."""

        print("[INFO] Initializing ModelLoader...")

        self.load_config()
        self.load_deployment_bundle()

        print("[SUCCESS] ModelLoader initialization completed.")

    def is_ready(self) -> bool:
        """Return True when both model and preprocessor are loaded."""

        return self.model is not None and self.preprocessor is not None

    def require_verified_production(self) -> None:
        """Reject local or stale bundles for production-only monitoring decisions."""
        if (
            self.metadata.get("provenance_status") != "verified"
            or self.metadata.get("model_stage") != "Production"
            or not self.model_version
            or not self.run_id
        ):
            raise RuntimeError(
                "Monitoring requires a verified MLflow Production model bundle; "
                "current deployment metadata is unverified or not Production."
            )


if __name__ == "__main__":
    loader = ModelLoader()

    try:
        loader.load()

        print("\n========== MODEL LOADER STATUS ==========")
        print(f"Model loaded       : {loader.model is not None}")
        print(f"Preprocessor loaded: {loader.preprocessor is not None}")
        print(f"Model name         : {loader.model_name}")
        print(f"Model version      : {loader.model_version}")
        print(f"MLflow Run ID      : {loader.run_id}")
        print(f"Model ID           : {loader.model_id}")
        print(f"Model path         : {loader.model_uri}")
        print("Ready              :", loader.is_ready())
        print("=========================================\n")

    except Exception as exc:
        print(f"[ERROR] Model loading failed: {exc}")
        raise
