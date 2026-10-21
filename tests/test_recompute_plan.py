"""Targeted corrections must preserve usable archival estimates and provenance."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

MODULE_PATH = Path(__file__).resolve().parents[1] / "tools/run_recompute_plan.py"
spec = importlib.util.spec_from_file_location("run_recompute_plan", MODULE_PATH)
plan_runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plan_runner)

PAPER_LAMBDA = "loss_average + (lambda/n) * sum(g(beta_j))"


class QuadratureEscalationTests(unittest.TestCase):
    def setUp(self):
        self.job = {'kind': 'quadrature', 'job_id': 'test_quadrature', 'df': 2., 'delta': 1.001,
                    'settings': {'orders': [300, 600, 1200, 2400, 4800],
                                 'relative_tolerance': 5e-4}}

    def solve(self, values):
        def fake_solve(df, delta, n_quad):
            value = values[n_quad]
            return {'df': df, 'delta': delta, 'alpha2_rank': value, 'alpha2_LAD': value}
        with mock.patch('rank_repro.student.solve_student_t', side_effect=fake_solve) as solve:
            result = plan_runner.worker(self.job)
        return result, [call.kwargs['n_quad'] for call in solve.call_args_list]

    def test_escalates_without_relaxing_tolerance_and_stops_at_first_pass(self):
        result, orders = self.solve({300: 1.1, 600: 1.01, 1200: 1.001, 2400: 1.0004, 4800: 1.})
        self.assertTrue(result['complete'])
        self.assertEqual(orders, [300, 600, 1200, 2400, 4800])
        self.assertGreater(result['trace'][-2]['max_relative_change'], 5e-4)
        self.assertLessEqual(result['trace'][-1]['max_relative_change'], 5e-4)
        result, orders = self.solve({300: 1., 600: 1.0001})
        self.assertTrue(result['complete'])
        self.assertEqual(orders, [300, 600])

    def test_exhausted_orders_and_nonfinite_results_remain_failures(self):
        result, _ = self.solve({300: 1.1, 600: 1.08, 1200: 1.06, 2400: 1.04, 4800: 1.02})
        self.assertFalse(result['complete'])
        result, _ = self.solve({300: np.inf})
        self.assertFalse(result['complete'])
        self.assertIn('nonpositive or nonfinite risk', result['error'])


class RecomputeMergeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.output = self.root / "output"
        self.output.mkdir()
        (self.output / "checkpoints").mkdir()
        self.data = self.root / "data"
        recovered = self.data / "recovered"
        (recovered / "empirical").mkdir(parents=True)
        (recovered / "theory").mkdir()
        self.plan = {"data_root": "data/recovered", "mode": "paper"}
        phase_rows = []
        empirical_rows = []
        theoretical_rows = []
        for df in (3., 8.):
            for delta in (2., 4.):
                phase_rows.append({"df": df, "delta": delta, "alpha2_LS": 3., "alpha2_LAD": 2., "alpha2_rank": 1.,
                                   "alpha_LS": np.sqrt(3), "alpha_LAD": np.sqrt(2), "alpha_rank": 1.})
                empirical_rows.append({"df": df, "delta": delta, "n": 100, "n_rep": 2,
                                       "mse_LS_mean": 3., "mse_LAD_mean": 2., "mse_rank_mean": 4.,
                                       "mse_LS_median": 2.8, "mse_LAD_median": 1.8, "mse_rank_median": 3.8,
                                       "success_rate_LS": 1., "success_rate_LAD": 1., "success_rate_rank": 1.})
                for penalty in ("lasso", "ridge"):
                    theoretical_rows.append({"df": df, "delta": delta, "penalty": penalty,
                                             "alpha2_ls_opt": 3., "alpha2_lad_opt": 2., "alpha2_rank_opt": 1.,
                                             "lambda_ls_opt": .6, "lambda_lad_opt": .4, "lambda_rank_opt": .8})
        penalized_rows = [{**row, "penalty": penalty, "lambda_LS_opt": .6, "lambda_LAD_opt": .4,
                           "lambda_rank_opt": .8, "lambda_eff_LS": .006, "lambda_eff_LAD": .004,
                           "lambda_eff_rank": .008, "winner_lambda": .4}
                          for row in empirical_rows for penalty in ("lasso", "ridge")]
        self.write_table(self.data / "unpenalized_student_t.csv", phase_rows, penalized=False)
        self.write_table(recovered / "empirical/unpenalized_phase.csv", empirical_rows, penalized=False)
        self.write_table(recovered / "empirical/penalized_phase.csv", penalized_rows, penalized=True)
        self.write_table(recovered / "theory/penalized_phase.csv", theoretical_rows, penalized=True, theoretical=True)
        for name in ("gaussian_scores.csv", "gaussian_parameters.csv"):
            (self.data / name).write_text("delta,value\n2,1\n")
        (self.data / "paper/are").mkdir(parents=True)
        (self.data / "paper/are/metadata.json").write_text('{"figure_numbers": {"cauchy": "S3", "laplace": "S4"}}\n')

    def tearDown(self):
        self.temporary.cleanup()

    def write_table(self, path, rows, *, penalized, theoretical=False):
        frame = pd.DataFrame(rows)
        frame.to_csv(path, index=False)
        meta = {"complete": True, "noise": "standard_student_t", "n": 100, "n_rep": 2,
                "grid": {"df": [3., 8.], "delta": [2., 4.]},
                "lambda_convention": PAPER_LAMBDA if penalized else "not_applicable",
                "signal": {"kind": "spike_slab", "nonzero_probability": .1 if theoretical else .2,
                           "nonzero_variance": 10. if theoretical else 5., "force_nonzero": not theoretical},
                "selection": {"criterion": "mse_median", "evaluation": "same_repetitions", "display_statistic": "mean"}}
        if penalized:
            meta["grid"]["penalty"] = ["ridge", "lasso"]
        path.with_suffix(".metadata.json").write_text(json.dumps(meta))

    def merge(self, records):
        jobs = []
        for index, record in enumerate(records):
            job_id = f"{index:06d}_{record['kind']}"
            jobs.append({"job_id": job_id, "kind": record["kind"]})
            (self.output / "checkpoints" / (job_id + ".json")).write_text(json.dumps(record))
        with mock.patch.object(plan_runner, "ROOT", self.root):
            return plan_runner.write_outputs(self.plan, self.output, jobs, "test-fingerprint")

    def test_incomplete_checkpoint_never_enters_published_table(self):
        before = pd.read_csv(self.data / "recovered/empirical/penalized_phase.csv")
        failed, updated = self.merge([{"kind": "penalized_repair", "complete": False,
            "updates": {"df": 3., "delta": 2., "penalty": "lasso", "mse_rank_mean": 0.00001},
            "raw": [], "trace": []}])
        path = self.output / "data/empirical/penalized_phase.csv"
        after = pd.read_csv(path)
        np.testing.assert_array_equal(before.mse_rank_mean, after.mse_rank_mean)
        self.assertEqual(failed["penalized_repair"], 1)
        self.assertEqual(updated["penalized_repair"], 0)
        self.assertFalse(json.loads(path.with_suffix(".metadata.json").read_text())["complete"])
        from rank_repro.figures_heavy import _load, FigureDataError
        with self.assertRaisesRegex(FigureDataError, "incomplete or failed"):
            _load(path, penalized=True)

    def test_rank_only_update_preserves_unrecomputed_ls_lad_and_lambda_units(self):
        source = pd.read_csv(self.data / "recovered/empirical/penalized_phase.csv")
        failed, updated = self.merge([{"kind": "penalized_repair", "complete": True,
            "updates": {"df": 3., "delta": 2., "penalty": "lasso", "mse_rank_mean": .5,
                        "mse_rank_median": .4, "lambda_rank_opt": 4., "lambda_eff_rank": .04},
            "raw": [{"rep": 0, "mse": .5}], "trace": []}])
        path = self.output / "data/empirical/penalized_phase.csv"
        after = pd.read_csv(path)
        for column in ("mse_LS_mean", "mse_LS_median", "mse_LAD_mean", "mse_LAD_median",
                       "lambda_LS_opt", "lambda_LAD_opt", "lambda_eff_LS", "lambda_eff_LAD"):
            np.testing.assert_array_equal(source[column], after[column])
        selected = after[(after.df == 3) & (after.delta == 2) & (after.penalty == "lasso")].iloc[0]
        self.assertEqual(selected.mse_rank_mean, .5)
        self.assertEqual(selected.winner, "rank")
        self.assertEqual(selected.winner_lambda, 4.)
        self.assertEqual(selected.lambda_rank_opt, 4.)  # Already paper lambda; no extra delta factor.
        self.assertEqual(selected.lambda_eff_rank, selected.lambda_rank_opt / selected.n)
        self.assertAlmostEqual(selected.ratio_rank_to_LS, .5 / 3.)
        unchanged = ~((source.df == 3) & (source.delta == 2) & (source.penalty == "lasso"))
        np.testing.assert_array_equal(source.loc[unchanged, "mse_rank_mean"], after.loc[unchanged, "mse_rank_mean"])
        self.assertEqual(updated["penalized_repair"], 1)
        self.assertEqual(failed["penalized_repair"], 0)
        meta = json.loads(path.with_suffix(".metadata.json").read_text())
        self.assertEqual(meta["lambda_convention"], PAPER_LAMBDA)

    def test_corrected_theory_and_empirical_signal_metadata_match_for_overlay(self):
        self.merge([{"kind": "theory_lasso", "complete": True,
            "updates": {"df": 3., "delta": 2., "penalty": "lasso", "alpha2_rank_opt": .75}, "raw": [], "trace": []},
            {"kind": "penalized_repair", "complete": True,
             "updates": {"df": 3., "delta": 2., "penalty": "lasso", "mse_rank_mean": .75}, "raw": [], "trace": []}])
        theory_path = self.output / "data/theory/penalized_phase.csv"
        empirical_path = self.output / "data/empirical/penalized_phase.csv"
        from rank_repro.figures_heavy import _load, _matched_overlay
        theory, theory_meta = _load(theory_path, penalized=True)
        _, empirical_meta = _load(empirical_path, penalized=True)
        self.assertEqual(theory_meta["signal"], empirical_meta["signal"])
        self.assertEqual(theory_meta["signal"], {"kind": "spike_slab", "nonzero_probability": .2,
                                               "nonzero_variance": 5., "force_nonzero": False})
        self.assertEqual(theory_meta["lambda_convention"], empirical_meta["lambda_convention"])
        self.assertEqual(empirical_meta["selection"], {"criterion": "mse_median", "evaluation": "same_repetitions", "display_statistic": "mean"})
        ax = mock.Mock()
        ax.get_xlim.return_value = (3., 8.)
        ax.get_ylim.return_value = (2., 4.)
        with mock.patch("rank_repro.figures_heavy._boundaries") as boundaries:
            _matched_overlay(ax, theory, theory_meta, empirical_meta,
                             ["alpha2_ls_opt", "alpha2_lad_opt", "alpha2_rank_opt"], penalized=True)
            boundaries.assert_called_once()

    def test_lasso_recalculation_keeps_all_recovered_ridge_values(self):
        before = pd.read_csv(self.data / "recovered/theory/penalized_phase.csv")
        self.merge([{"kind": "theory_lasso", "complete": True,
                     "updates": {"df": 3., "delta": 2., "penalty": "lasso", "alpha2_ls_opt": 1.7,
                                 "alpha2_lad_opt": 1.5, "alpha2_rank_opt": 1.2}, "raw": [], "trace": []}])
        after = pd.read_csv(self.output / "data/theory/penalized_phase.csv")
        pd.testing.assert_frame_equal(before[before.penalty == "ridge"].reset_index(drop=True),
                                      after[after.penalty == "ridge"].reset_index(drop=True))


if __name__ == "__main__":
    unittest.main()
