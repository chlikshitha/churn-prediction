from pathlib import Path
import runpy
if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("10_failure_injection.py")), run_name="__main__")
