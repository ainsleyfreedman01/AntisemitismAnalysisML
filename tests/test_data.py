from pathlib import Path
import tempfile
import unittest

import pandas as pd

from campus_trends.data import REQUIRED_COLUMNS, load_reports


class DataTests(unittest.TestCase):
    def test_loading_normalizes_and_sorts_reports(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reports.csv"
            pd.DataFrame(
                [
                    ["03/15/24", "City", "State", "Vandalism", " Description ", "Campus"],
                    ["01/15/24", "City", "State", "Harassment", "Other description", "Campus"],
                ],
                columns=list(REQUIRED_COLUMNS),
            ).to_csv(path, index=False)
            reports = load_reports(path)
            self.assertEqual(reports["Date"].dt.month.tolist(), [1, 3])
            self.assertEqual(reports.iloc[1]["Description"], "Description")

    def test_invalid_data_is_not_silently_discarded(self) -> None:
        raw = pd.DataFrame(
            [["01/15/24", "City", "State", "Vandalism", "Description", "Campus"]],
            columns=list(REQUIRED_COLUMNS),
        )
        variants = [
            raw.drop(columns="Description"),
            raw.assign(Description=" "),
            raw.assign(Description=None),
            raw.assign(**{"Date of Incident": "not a date"}),
            raw.assign(**{"Incident Type": "Unknown"}),
            pd.concat([raw, raw]),
            raw.iloc[:0],
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reports.csv"
            for invalid in variants:
                with self.subTest(shape=invalid.shape):
                    invalid.to_csv(path, index=False)
                    with self.assertRaises(ValueError):
                        load_reports(path)


if __name__ == "__main__":
    unittest.main()
