"""Independent small LP/QP references for the nonsmooth rank estimator."""

import unittest

import numpy as np
from scipy.optimize import linprog, minimize, LinearConstraint

from rank_repro.rank_solver import fit_rank_convex, rank_loss, rank_prox


def epigraph_reference(X, y, penalty="none", lam=0.0):
    n, p = X.shape
    left, right = np.triu_indices(n, 1)
    A = X[left] - X[right]
    b = y[left] - y[right]
    m = len(b)
    extra = p if penalty == "lasso" else 0
    size = p + m + extra
    inequalities = np.zeros((2 * m + 2 * extra, size))
    inequalities[:m, :p] = -A
    inequalities[:m, p:p + m] = -np.eye(m)
    inequalities[m:2 * m, :p] = A
    inequalities[m:2 * m, p:p + m] = -np.eye(m)
    upper = np.concatenate([-b, b, np.zeros(2 * extra)])
    if extra:
        inequalities[2 * m:2 * m + p, :p] = np.eye(p)
        inequalities[2 * m:2 * m + p, p + m:] = -np.eye(p)
        inequalities[2 * m + p:, :p] = -np.eye(p)
        inequalities[2 * m + p:, p + m:] = -np.eye(p)
    linear = np.zeros(size)
    linear[p:p + m] = 2.0 / (n * (n - 1))
    linear[p + m:] = lam / n
    bounds = [(None, None)] * p + [(0, None)] * (m + extra)
    if penalty != "ridge":
        result = linprog(linear, A_ub=inequalities, b_ub=upper, bounds=bounds, method="highs")
    else:
        def objective(z):
            return linear @ z + lam / (2 * n) * np.dot(z[:p], z[:p])

        def gradient(z):
            grad = linear.copy()
            grad[:p] = lam / n * z[:p]
            return grad

        z0 = np.concatenate([np.zeros(p), np.abs(b) + 1.0])
        result = minimize(objective, z0, jac=gradient, bounds=bounds,
                          constraints=LinearConstraint(inequalities, -np.inf, upper),
                          method="SLSQP", options={"ftol": 1e-12, "maxiter": 2000})
    if not result.success:
        raise AssertionError(result.message)
    return result.x[:p], result.fun


class RankConvexSolverTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(923)
        self.X = rng.normal(size=(12, 3))
        self.y = self.X @ np.array([0.8, -0.5, 0.1]) + rng.standard_t(3, size=12)

    def assert_converged(self, info):
        self.assertTrue(info["success"], info)
        self.assertLessEqual(info["primal_residual"], info["primal_tolerance"])
        self.assertLessEqual(info["dual_residual"], info["dual_tolerance"])
        self.assertLessEqual(info["duality_gap"], info["gap_tolerance"])

    def test_loss_is_pairwise_mean(self):
        r = np.array([2.0, -4.0, 2.0, 8.0, -1.0])
        left, right = np.triu_indices(len(r), 1)
        self.assertAlmostEqual(rank_loss(r), np.abs(r[left] - r[right]).mean(), places=13)
        self.assertAlmostEqual(rank_loss(r + 100.0), rank_loss(r), places=13)

    def test_proximal_optimality_against_pairwise_subgradient(self):
        values = np.array([2.0, -1.1, 0.4, 0.5, 1.8, 2.1])
        step = 1.3
        prox = rank_prox(values, step)
        d = (values - prox) / step
        n = len(values)
        prefix = np.cumsum(np.sort(d)[::-1])[:-1]
        k = np.arange(1, n)
        self.assertLessEqual(np.max(prefix - 2 * k * (n - k) / (n * (n - 1))), 1e-12)
        self.assertAlmostEqual(d.sum(), 0.0, places=12)
        self.assertAlmostEqual(float(d @ prox), rank_loss(prox), places=12)

    def test_unpenalized_matches_independent_linear_program(self):
        _, expected = epigraph_reference(self.X, self.y)
        beta, info = fit_rank_convex(self.X, self.y, atol=1e-9, rtol=1e-8)
        self.assert_converged(info)
        self.assertAlmostEqual(info["objective"], expected, delta=2e-8)
        self.assertAlmostEqual(rank_loss(self.y - self.X @ beta), expected, delta=2e-8)
        self.assertLessEqual(info["dual_objective"], expected + 1e-10)

    def test_lasso_matches_independent_linear_program(self):
        _, expected = epigraph_reference(self.X, self.y, "lasso", 1.7)
        _, info = fit_rank_convex(self.X, self.y, penalty="lasso", lam=1.7, atol=1e-9, rtol=1e-8)
        self.assert_converged(info)
        self.assertAlmostEqual(info["objective"], expected, delta=2e-8)
        self.assertLessEqual(info["dual_objective"], expected + 1e-10)

    def test_ridge_matches_independent_epigraph_quadratic_program(self):
        expected_beta, expected = epigraph_reference(self.X, self.y, "ridge", 2.3)
        beta, info = fit_rank_convex(self.X, self.y, penalty="ridge", lam=2.3, atol=1e-9, rtol=1e-8)
        self.assert_converged(info)
        self.assertAlmostEqual(info["objective"], expected, delta=2e-8)
        np.testing.assert_allclose(beta, expected_beta, atol=2e-6, rtol=2e-6)
        self.assertLessEqual(info["dual_objective"], expected + 1e-10)

    def test_residual_translation_invariance(self):
        beta, info = fit_rank_convex(self.X, self.y, penalty="ridge", lam=0.7)
        shifted, shifted_info = fit_rank_convex(self.X + 9.0, self.y + 50.0, penalty="ridge", lam=0.7)
        self.assert_converged(info)
        self.assert_converged(shifted_info)
        np.testing.assert_allclose(shifted, beta, atol=1e-9, rtol=1e-9)

    def test_p_equals_n_minus_one_has_zero_rank_loss(self):
        rng = np.random.default_rng(64)
        X = rng.normal(size=(16, 15))
        y = rng.standard_t(1, size=16)
        beta, info = fit_rank_convex(X, y)
        self.assert_converged(info)
        self.assertEqual(info["status"], "zero_loss_solution")
        self.assertLess(rank_loss(y - X @ beta), 1e-10)

    def test_max_iteration_is_not_reported_as_convergence(self):
        _, info = fit_rank_convex(self.X, self.y, penalty="lasso", lam=0.3, maxiter=1)
        self.assertFalse(info["success"])
        self.assertEqual(info["status"], "max_iterations")

    def test_overparameterized_penalized_design(self):
        rng = np.random.default_rng(2)
        X = rng.normal(size=(8, 12)) / np.sqrt(8)
        y = rng.normal(size=8)
        _, expected = epigraph_reference(X, y, "lasso", 0.9)
        _, info = fit_rank_convex(X, y, penalty="lasso", lam=0.9, atol=1e-9, rtol=1e-8)
        self.assert_converged(info)
        self.assertAlmostEqual(info["objective"], expected, delta=2e-8)

    def test_rank_deficient_unpenalized_design_rejected(self):
        X = self.X.copy()
        X[:, 1] = X[:, 0]
        with self.assertRaisesRegex(ValueError, "rank deficient"):
            fit_rank_convex(X, self.y)

    def test_small_lambda_shortcut_is_certified_and_cached(self):
        rng = np.random.default_rng(78)
        X = rng.normal(size=(8, 12)) / np.sqrt(8)
        y = rng.standard_t(1, size=8)
        for lam in (0.01, 0.005):
            _, expected = epigraph_reference(X, y, "lasso", lam)
            _, info = fit_rank_convex(X, y, penalty="lasso", lam=lam, atol=1e-9, rtol=1e-8)
            self.assert_converged(info)
            self.assertEqual(info["method"], "basis_pursuit_with_rank_dual_certificate")
            self.assertAlmostEqual(info["objective"], expected, delta=1e-8)
        self.assertTrue(info["lp_cache_hit"])

    def test_interpolation_shortcut_rejected_when_dual_not_feasible(self):
        rng = np.random.default_rng(79)
        X = rng.normal(size=(8, 12)) * 1e-4
        y = rng.normal(size=8)
        _, expected = epigraph_reference(X, y, "lasso", 0.01)
        _, info = fit_rank_convex(X, y, penalty="lasso", lam=0.01, atol=1e-9, rtol=1e-8)
        self.assert_converged(info)
        self.assertEqual(info["method"], "two_block_admm_isotonic_prox")
        self.assertAlmostEqual(info["objective"], expected, delta=1e-8)


if __name__ == "__main__":
    unittest.main()
