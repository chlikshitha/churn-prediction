"""Unit tests for schema and preprocessing failure detection."""
import importlib.util
from pathlib import Path
import unittest
import sys
import pandas as pd
import pandera as pa
import validation as schema_validation

ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = ROOT / "src" / "mlops_pipeline"
sys.path.insert(0, str(PIPELINE_DIR))
spec = importlib.util.spec_from_file_location("validate_data", ROOT / "src/mlops_pipeline/02_validate_data.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class ValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = pd.read_csv(ROOT / "data/raw/churn.csv")

    def test_current_data_passes(self):
        validated = schema_validation.get_telco_schema().validate(self.data, lazy=True)
        self.assertEqual(len(validated), len(self.data))

    def test_missing_column_fails(self):
        with self.assertRaises(pa.errors.SchemaErrors):
            schema_validation.get_telco_schema().validate(
                self.data.drop(columns=["Contract"]),
                lazy=True,
            )

    def test_non_numeric_charge_fails(self):
        bad = self.data.copy()
        bad["MonthlyCharges"] = bad["MonthlyCharges"].astype(object)
        bad.loc[bad.index[0], "MonthlyCharges"] = "INVALID_NUMBER"
        with self.assertRaises(pa.errors.SchemaErrors):
            schema_validation.get_telco_schema().validate(bad, lazy=True)

if __name__ == "__main__":
    unittest.main()
