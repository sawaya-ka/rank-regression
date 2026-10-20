"""Numerical conventions and restart behavior, without a production grid run."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from rank_repro import _empirical, _theory_penalized, _unpenalized, experiments


class ExperimentTests(unittest.TestCase):
    def test_rank_pairwise_definition_and_directional_derivative(self):
        rng = np.random.default_rng(12)
        x = rng.normal(size=(13, 3))
        y = rng.normal(size=13)
        beta = rng.normal(size=3)
        residual = y - x @ beta
        i, j = np.triu_indices(len(y), 1)
        pair_sum = np.abs(residual[i] - residual[j]).sum()
        value, gradient = _empirical.rank_loss_and_grad(beta, x, y)
        self.assertAlmostEqual(value, 2 * pair_sum / (13 * 12))
        direction, step = rng.normal(size=3), 1e-7
        plus = _empirical.rank_loss_and_grad(beta + step * direction, x, y)[0]
        minus = _empirical.rank_loss_and_grad(beta - step * direction, x, y)[0]
        self.assertAlmostEqual((plus - minus) / (2 * step), gradient @ direction, places=7)
        unpen_value = _unpenalized.rank_loss_and_grad(beta, x, y)[0]
        self.assertAlmostEqual(unpen_value, np.sqrt(3) * pair_sum / (13 * 14))

    def test_ridge_uses_paper_lambda_over_n(self):
        for n, p in [(23, 7), (7, 23)]:
            with self.subTest(n=n, p=p):
                rng = np.random.default_rng(3)
                x = rng.normal(size=(n, p)) / np.sqrt(p)
                y = rng.normal(size=n)
                lam = 2.3
                expected = np.linalg.solve(x.T @ x + lam * np.eye(p), x.T @ y)
                np.testing.assert_allclose(_empirical.fit_ls_ridge(x, y, lam), expected)
                old_lambda = 1.7
                old_solution = np.linalg.solve(x.T @ x + n * old_lambda / p * np.eye(p), x.T @ y)
                np.testing.assert_allclose(_empirical.fit_ls_ridge(x, y, n / p * old_lambda), old_solution)

    def test_theory_ridge_calibration_and_bayes_optimum(self):
        for delta in [0.5, 1.0, 2.0, 10.0]:
            with self.subTest(delta=delta):
                variance, probability, slab_variance = 3.0, 0.2, 5.0
                optimal = _theory_penalized.analytic_ls_ridge_lambda_seed(delta, variance, probability, slab_variance, 0.01, 100)
                self.assertAlmostEqual(optimal, variance / (probability * slab_variance))
                gamma = _theory_penalized._ls_ridge_gamma_from_lambda(delta, optimal)
                self.assertAlmostEqual(_theory_penalized._ls_ridge_lambda_from_gamma(delta, gamma), optimal)
                risk = _theory_penalized._alpha2_ls_ridge_at_lambda
                center = risk(delta, optimal, variance, probability, slab_variance)
                self.assertLessEqual(center, risk(delta, optimal * 0.8, variance, probability, slab_variance))
                self.assertLessEqual(center, risk(delta, optimal * 1.2, variance, probability, slab_variance))

    def test_seed_order_and_unconditioned_signal(self):
        forward = [_empirical.make_seed(42, 1, 3, 5, r) for r in range(5)]
        reverse = [_empirical.make_seed(42, 1, 3, 5, r) for r in reversed(range(5))]
        self.assertEqual(forward, reverse[::-1])
        x, _, beta, _, _ = _unpenalized.sample_problem(n=100, delta=5, df=4, seed=10)
        generator = np.random.default_rng(10)
        np.testing.assert_array_equal(x, generator.standard_normal(size=(100, 20)))
        np.testing.assert_array_equal(beta, generator.standard_normal(size=20))
        generator = np.random.default_rng(8)
        samples = [_empirical.generate_beta_true(4, generator, ensure_nonzero=False) for _ in range(100)]
        self.assertTrue(any(np.count_nonzero(beta) == 0 for beta in samples))

    def test_resume_requires_complete_matching_checkpoint(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.json"
            payload = {"experiment": "empirical_unpenalized", "preset": "test", "settings": {"df_values": [4.0], "delta_values": [2.0], "n": 20, "n_rep": 1, "base_seed": 1}}
            config.write_text(json.dumps(payload))
            calls = []

            def worker(job):
                calls.append(job)
                return {"key": [0, 0, 0], "complete": True, "summary": {"df": 4.0, "delta": 2.0, "test": 1}, "trace": [], "raw": []}

            with patch.object(experiments, "_worker", worker):
                result = experiments.run_experiment("empirical_unpenalized", config, root / "data", 1)
                self.assertTrue(result["complete"])
                self.assertEqual(len(calls), 1)
                experiments.run_experiment("empirical_unpenalized", config, root / "data", 1)
                self.assertEqual(len(calls), 1)
                checkpoint = next((root / "data/checkpoints/empirical_unpenalized").glob("*.json"))
                checkpoint.write_text("incomplete write")
                experiments.run_experiment("empirical_unpenalized", config, root / "data", 1)
                self.assertEqual(len(calls), 2)
                payload["settings"]["n_rep"] = 2
                config.write_text(json.dumps(payload))
                with self.assertRaisesRegex(ValueError, "Settings or engine source changed"):
                    experiments.run_experiment("empirical_unpenalized", config, root / "data", 1)

    def test_failed_lambda_search_has_no_silent_fallback(self):
        lam, row = _empirical.choose_best_lambda({1.0: {"mse_mean": 0.2, "success_rate": 0.0}}, min_success_rate=1.0)
        self.assertTrue(np.isnan(lam))
        self.assertEqual(row, {})

    def test_nan_theory_residual_is_rejected(self):
        self.assertFalse(_theory_penalized.good_solution({"alpha2": 1.0, "cost": np.nan}))

    def test_lad_quadrature_preserves_fractional_df(self):
        from scipy.stats import t

        _theory_penalized._T_MIXTURE_QUAD_CACHE.clear()
        first = _theory_penalized.lad_AC_fast_t(0, 1, 4.2, False, 1, 160)
        second = _theory_penalized.lad_AC_fast_t(0, 1, 4.8, False, 1, 160)
        self.assertNotEqual(first["C"], second["C"])
        self.assertAlmostEqual(first["C"], 2 * t.cdf(1, 4.2) - 1, delta=1e-5)
        self.assertAlmostEqual(second["C"], 2 * t.cdf(1, 4.8) - 1, delta=1e-5)
        self.assertEqual(len(_theory_penalized._T_MIXTURE_QUAD_CACHE), 2)

    def test_active_rank_cutoff_not_lost_on_coarse_grid(self):
        from scipy.stats import t

        _theory_penalized.init_gh(20)
        kappa = (1 + 1e-5) / (4 * t.pdf(0, 4))
        stats = _theory_penalized.rank_AC(0, kappa, 4, False, 21)
        self.assertGreater(stats["c"], 0)
        residual = 4 * kappa * (t.cdf(stats["c"], 4) - 0.5) - stats["c"]
        self.assertAlmostEqual(residual, 0, delta=1e-10)


if __name__ == "__main__":
    unittest.main()
