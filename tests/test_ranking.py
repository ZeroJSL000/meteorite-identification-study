"""Checks for the final submission's count calibration rule."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from utils.ranking import select_top_count


class SelectTopCountTests(unittest.TestCase):
    def test_exact_count_follows_probability_order(self) -> None:
        self.assertEqual(select_top_count([0.2, 0.9, 0.6, 0.1], 2), [0, 1, 1, 0])

    def test_ties_keep_original_row_order(self) -> None:
        self.assertEqual(select_top_count([0.7, 0.7, 0.1], 1), [1, 0, 0])

    def test_final_stage_has_exactly_86_positives(self) -> None:
        labels = select_top_count([index / 194 for index in range(194)], 86)
        self.assertEqual(len(labels), 194)
        self.assertEqual(sum(labels), 86)
        self.assertEqual(labels[:108], [0] * 108)
        self.assertEqual(labels[108:], [1] * 86)

    def test_invalid_inputs_fail_before_writing_submission(self) -> None:
        with self.assertRaises(ValueError):
            select_top_count([0.1], 2)
        with self.assertRaises(ValueError):
            select_top_count([float("nan")], 1)


if __name__ == "__main__":
    unittest.main()
