"""Compatibility entry point for production alias management."""
from pathlib import Path
import runpy
if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).parent / "mlops_pipeline" / "08_automate_lifecycle.py"), run_name="__main__")
