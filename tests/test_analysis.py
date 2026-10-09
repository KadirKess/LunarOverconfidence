import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from lunar_overconfidence.analysis import (
    analyze_bias_performance,
    compare_final_episodes,
    save_bias_performance,
)
from tests.fixtures import make_results


class PerformanceTests(unittest.TestCase):
    def test_single_run_summary_has_no_p_value(self):
        evaluator = SimpleNamespace(
            history=[SimpleNamespace(rewards=torch.tensor([1.0, 3.0]))]
        )
        output = StringIO()
        with redirect_stdout(output):
            summaries = compare_final_episodes(evaluator, evaluator)
        self.assertEqual(summaries["DDPG"]["mean"], 2)
        self.assertAlmostEqual(summaries["DDPG"]["std"], np.sqrt(2), places=6)
        self.assertNotIn("p-value", output.getvalue())


class BiasPerformanceTests(unittest.TestCase):
    make_results = staticmethod(make_results)

    def test_shared_checkpoint_correlations_and_paired_changes(self):
        results = self.make_results()
        results[("TD3", "baseline", 1)].append(
            {
                **results[("TD3", "baseline", 1)][3],
                "step": 30,
            }
        )
        report = analyze_bias_performance(results)
        self.assertEqual(report["step"], 20)
        self.assertEqual(len(report["models"]), 6)
        self.assertEqual(len(report["normalization_pairs"]), 3)
        for row in report["associations"]:
            self.assertAlmostEqual(row["pearson"], -1.0)
            self.assertAlmostEqual(row["spearman"], -1.0)
            self.assertEqual(row["n_training_seeds"], 3)
            self.assertNotIn("p_value", row)
        for row in report["normalization_pairs"]:
            self.assertEqual(row["delta_bias"], -5)
            self.assertEqual(row["delta_reward"], 10)
        earlier = analyze_bias_performance(results, step=10)
        self.assertEqual(earlier["models"][0]["bias"], 11)

    def test_rejects_missing_duplicate_and_incompatible_evaluations(self):
        results = self.make_results()
        with self.assertRaisesRegex(ValueError, "not available"):
            analyze_bias_performance(results, step=30)
        with self.assertRaisesRegex(ValueError, "No other"):
            analyze_bias_performance(results, critic="other")
        rows = results[("TD3", "baseline", 1)]
        rows.append(dict(rows[3]))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            analyze_bias_performance(results)
        rows.pop()
        rows[3]["score_horizon"] = 500
        with self.assertRaisesRegex(ValueError, "score_horizon"):
            analyze_bias_performance(results)

    def test_capped_and_nonfinite_models_are_reported(self):
        results = self.make_results()
        row = results[("TD3", "baseline", 1)][3]
        row.update(
            bias=float("nan"),
            completed_episodes=0,
            capped_episodes=10,
            completion_rate=0.0,
        )
        report = analyze_bias_performance(results)
        self.assertEqual(len(report["models"]), 6)
        self.assertEqual(report["associations"][0]["excluded_models"], 1)
        self.assertEqual(len(report["normalization_pairs"]), 2)
        self.assertTrue(any("capped episodes" in note for note in report["notes"]))
        baseline = next(
            row for row in report["associations"] if row["variant"] == "baseline"
        )
        self.assertTrue(np.isnan(baseline["pearson"]))

    def test_constant_values_and_unmatched_seeds(self):
        results = self.make_results()
        del results[("TD3", "layer-norm", 3)]
        for rows in results.values():
            for row in rows:
                row["bias"] = 1.0
        report = analyze_bias_performance(results)
        self.assertTrue(all(np.isnan(row["pearson"]) for row in report["associations"]))
        self.assertEqual(len(report["normalization_pairs"]), 2)
        self.assertTrue(any("No matched" in note for note in report["notes"]))

    def test_saves_tables_and_notes(self):
        import csv

        report = analyze_bias_performance(self.make_results())
        with tempfile.TemporaryDirectory() as directory:
            save_bias_performance(report, directory)
            root = Path(directory)
            with (root / "models.csv").open() as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 6)
            self.assertTrue((root / "associations.csv").is_file())
            self.assertTrue((root / "normalization_pairs.csv").is_file())
            self.assertIn("Training step: 20", (root / "notes.txt").read_text())
