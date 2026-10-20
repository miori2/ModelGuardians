import tempfile
import unittest
from pathlib import Path

import pandas as pd

from fraud_model import (
    FraudModel,
    load_dataset,
    load_fraud_model,
    train_fraud_model,
)


def write_training_csv(path: Path) -> None:
    rows = [
        {
            "id": index,
            "V1": float(index % 2),
            "Amount": float(index + 1),
            "Class": index % 2,
        }
        for index in range(80)
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


class FraudModelTests(unittest.TestCase):
    def test_dataset_loader_uses_binary_label_and_excludes_identifier(self):
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "fraud.csv"
            write_training_csv(csv_path)

            features, target, target_column = load_dataset(csv_path)

            self.assertEqual(target_column, "Class")
            self.assertEqual(list(features.columns), ["V1", "Amount"])
            self.assertEqual(target.value_counts().to_dict(), {0: 40, 1: 40})

    def test_training_saves_evaluable_model_and_supports_prediction(self):
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "fraud.csv"
            model_path = Path(directory) / "model.joblib"
            write_training_csv(csv_path)

            model = train_fraud_model(
                csv_path,
                model_path,
                model_params={"max_iter": 5, "max_leaf_nodes": 5},
            )
            loaded = load_fraud_model(model_path)

            self.assertIsInstance(loaded, FraudModel)
            self.assertEqual(model.public()["feature_count"], 2)
            self.assertIn("average_precision", model.metrics)
            prediction = loaded.predict({"V1": 1.0, "Amount": 50.0})
            self.assertGreaterEqual(prediction["fraud_probability"], 0)
            self.assertLessEqual(prediction["fraud_probability"], 1)
            self.assertIn(prediction["predicted_class"], (0, 1))
            with self.assertRaisesRegex(ValueError, "Missing required"):
                loaded.predict({"V1": 1.0})


if __name__ == "__main__":
    unittest.main()
