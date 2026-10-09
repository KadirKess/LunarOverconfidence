import copy
import csv
import tempfile
import unittest
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from lunar_overconfidence.analysis import (
    analyze_bias_performance,
    load_evaluations,
    save_bias_performance,
)
from lunar_overconfidence.experiments import plot_diagnostics, plot_experiment
from lunar_overconfidence.plotting.learning_curves import draw_seed_curves
from tests.fixtures import make_results


class ReportTests(unittest.TestCase):
    def test_report_exports_figures_and_tables_without_changing_inputs(self):
        results = make_results()
        before = copy.deepcopy(results)
        with tempfile.TemporaryDirectory() as directory:
            report = analyze_bias_performance(results)
            save_bias_performance(report, Path(directory) / "tables")
            figures = plot_experiment(results, report, directory)
            figures.update(plot_diagnostics(results, directory))
            self.assertEqual(report["step"], 20)
            self.assertEqual(len(figures), 5)
            for formats in figures.values():
                for path in formats.values():
                    self.assertGreater(path.stat().st_size, 1000)
            self.assertTrue((Path(directory) / "tables/models.csv").is_file())
            self.assertFalse((Path(directory) / "README.md").exists())
        self.assertEqual(results, before)
        self.assertFalse(plt.get_fignums())

    def test_missing_bias_is_a_gap_instead_of_a_zero(self):
        rows = make_results()[("TD3", "baseline", 1)]
        rows[3]["bias"] = float("nan")
        fig, ax = plt.subplots()
        try:
            draw_seed_curves(ax, [rows], "bias", "blue")
            mean = ax.lines[-1].get_ydata()
            self.assertEqual(mean[0], 11)
            self.assertTrue(np.isnan(mean[1]))
        finally:
            plt.close(fig)

    def test_csv_loader_restores_numeric_metadata(self):
        results = make_results()
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for index, rows in enumerate(results.values()):
                root = Path(directory) / str(index)
                (root / "analysis").mkdir(parents=True)
                with (root / "analysis/bias.csv").open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
                paths.append(root)
            loaded = load_evaluations(paths)
            expected = {
                (algo.lower(), variant, seed): rows
                for (algo, variant, seed), rows in results.items()
            }
            self.assertEqual(loaded, expected)
            with self.assertRaisesRegex(ValueError, "Duplicate training run"):
                load_evaluations([paths[0], paths[0]])
