from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from campus_trends.data import REQUIRED_COLUMNS
from campus_trends.themes import clean_description, find_themes, run_themes, theme_label


def reports_fixture() -> pd.DataFrame:
    rows = []
    for index in range(36):
        subject = (
            "Graffiti painted wall spray paint building"
            if index % 2
            else "Rally march banners crowd demonstration"
        )
        rows.append(
            [
                "01/15/24" if index < 18 else "03/15/24",
                "Exampleville",
                "Examplestate",
                "Vandalism" if index % 2 else "Protest/Action",
                f"{subject} {'afternoon' if index % 3 else 'evening'} {index}",
                "Example Institution",
            ]
        )
    raw = pd.DataFrame(rows, columns=list(REQUIRED_COLUMNS))
    raw["Date"] = pd.to_datetime(raw["Date of Incident"], format="%m/%d/%y")
    return raw


class ThemeTests(unittest.TestCase):
    def test_chart_labels_do_not_repeat_overlapping_phrases(self) -> None:
        self.assertEqual(
            theme_label("Theme 1", "rally signs, rally, signs, march, march crowd"),
            "Theme 1: rally signs, march",
        )
        self.assertEqual(theme_label("Mixed themes", "unused"), "Mixed themes")

    def test_contact_strings_removed_from_model_text(self) -> None:
        cleaned = clean_description(
            "Wall graffiti person@example.org https://example.org 212-555-0123"
        )
        self.assertIn("graffiti", cleaned)
        self.assertNotIn("example", cleaned)
        self.assertNotIn("555", cleaned)

    def test_themes_are_deterministic_and_counts_conserved(self) -> None:
        reports = reports_fixture()
        first = find_themes(reports, 2)
        second = find_themes(reports, 2)
        for left, right in zip(first, second):
            pd.testing.assert_frame_equal(left, right)
        summary, monthly, shares, assigned = first
        self.assertEqual(summary["reports"].sum(), len(reports))
        self.assertEqual(monthly.to_numpy().sum(), len(reports))
        self.assertTrue(shares.loc["2024-02"].isna().all())
        np.testing.assert_allclose(shares.loc[["2024-01", "2024-03"]].sum(axis=1), 1)
        self.assertTrue(assigned["dominant_weight_share"].between(0, 1).all())
        self.assertTrue(any("graffiti" in words for words in summary["keywords"]))
        self.assertTrue(any("rally" in words for words in summary["keywords"]))

    def test_unmatched_description_not_forced_into_a_theme(self) -> None:
        reports = reports_fixture()
        reports.loc[0, "Description"] = "the and or"
        _, monthly, _, assigned = find_themes(reports, 2)
        self.assertEqual(assigned.loc[0, "theme"], "Unassigned")
        self.assertEqual(monthly["Unassigned"].sum(), 1)

    def test_invalid_topic_counts_fail(self) -> None:
        for count in (0, 1, 40):
            with self.subTest(count=count), self.assertRaises(ValueError):
                find_themes(reports_fixture(), count)

    def test_only_three_chart_images_are_written(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "input.csv"
            reports_fixture().drop(columns="Date").to_csv(path, index=False)
            output = root / "themes"
            run_themes(path, output, 2)
            self.assertEqual(
                {item.name for item in output.iterdir()},
                {"description_themes.png", "theme_counts_by_month.png", "theme_share_by_month.png"},
            )
            for image in output.iterdir():
                self.assertGreater(image.stat().st_size, 0)
                self.assertEqual(image.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")


if __name__ == "__main__":
    unittest.main()
