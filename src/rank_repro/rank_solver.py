"""Convex rank regression with an exact isotonic proximal step.

The objective is 2/[n(n-1)] sum_{i<j}|r_i-r_j| + lam/n * P(beta),
where r=y-X beta and P is ||beta||_1 or ||beta||_2^2/2.
Internally the objective is multiplied by n; this does not alter its minimizer.
The two ADMM blocks are beta and (r,z), not three sequential variable blocks.
Residual criteria follow Boyd et al. (2011), section 3.3.1, and are supplemented
by a primal-dual objective gap.  A max-iteration exit is never success.
For overparameterized, weakly penalized lasso, a basis-pursuit LP is tried first
and accepted only when its dual certifies optimality for the rank objective.
"""

from __future__ import annotations

from collections import OrderedDict
import hashlib
import warnings

import numpy as np
from scipy.linalg import cho_factor, cho_solve, qr, solve_triangular
from scipy.optimize import isotonic_regression, linprog, OptimizeWarning


_BASIS_CACHE: OrderedDict = OrderedDict()


def _basis_pursuit(X, y):
    """Share LP candidates across lambda paths, retaining the 50-repetition batch."""
    key = hashlib.blake2b(digest_size=20)
    key.update(np.asarray(X.shape, dtype=np.int64).tobytes())
    key.update(X.tobytes())
    key.update(y.tobytes())
    digest = key.digest()
    if digest in _BASIS_CACHE:
        _BASIS_CACHE.move_to_end(digest)
        return _BASIS_CACHE[digest], True
    n, p = X.shape
    # An unpenalized intercept avoids a redundant centered equality row.
    with warnings.catch_warnings():
        # SciPy forwards this native HiGHS option but warns that it is not a
        # scipy-level option.  One solver thread prevents nested VM parallelism.
        warnings.filterwarnings("ignore", message="Unrecognized options detected.*(threads|parallel).*", category=OptimizeWarning)
        options = {"threads": 1, "parallel": False, "primal_feasibility_tolerance": 1e-9, "dual_feasibility_tolerance": 1e-9}
        result = linprog(
            np.r_[np.ones(2 * p), 0.0],
            A_eq=np.column_stack([X, -X, np.ones(n)]), b_eq=y,
            bounds=[(0.0, None)] * (2 * p) + [(None, None)], method="highs",
            options=options,
        )
        if result.status == 4:
            # HiGHS has one process-wide scheduler.  A previous third-party
            # linprog call may have initialized it with another thread count.
            # Preserve that scheduler but still disable parallel LP solving.
            options.pop("threads")
            result = linprog(
                np.r_[np.ones(2 * p), 0.0],
                A_eq=np.column_stack([X, -X, np.ones(n)]), b_eq=y,
                bounds=[(0.0, None)] * (2 * p) + [(None, None)], method="highs", options=options,
            )
    candidate = None
    if result.success:
        candidate = (result.x[:p] - result.x[p:2 * p], result.eqlin.marginals, int(result.nit))
    _BASIS_CACHE[digest] = candidate
    if len(_BASIS_CACHE) > 64:
        _BASIS_CACHE.popitem(last=False)
    return candidate, False


def rank_loss(residual: np.ndarray) -> float:
    """Average absolute difference over unordered pairs, in O(n log n)."""
    r = np.asarray(residual, dtype=float)
    if r.ndim != 1 or r.size < 2:
        raise ValueError("At least two one-dimensional residuals are required.")
    centered = r - np.mean(r)
    weights = 2.0 * (2.0 * np.arange(1, r.size + 1) - r.size - 1.0)
    weights /= r.size * (r.size - 1.0)
    return float(np.dot(weights, np.sort(centered)))


def rank_prox(values: np.ndarray, step: float, *, scaled_by_n: bool = False) -> np.ndarray:
    """Return prox_{step * rank_loss}(values) exactly up to floating point."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size < 2 or not np.all(np.isfinite(values)):
        raise ValueError("values must be a finite vector with at least two entries.")
    if not np.isfinite(step) or step < 0:
        raise ValueError("step must be finite and nonnegative.")
    n = values.size
    order = np.argsort(values, kind="stable")
    mean = float(np.mean(values))
    weights = 2.0 * (2.0 * np.arange(1, n + 1) - n - 1.0) / (n - 1.0)
    if not scaled_by_n:
        weights /= n
    out = np.empty(n)
    out[order] = isotonic_regression(values[order] - mean - step * weights).x + mean
    return out


def _dual_rank_feasible(d: np.ndarray) -> np.ndarray:
    """Scale a zero-mean vector into the dual ball of n * rank_loss."""
    d = d - np.mean(d)
    n = d.size
    prefix = np.cumsum(np.sort(d)[::-1])[:-1]
    k = np.arange(1, n, dtype=float)
    bounds = 2.0 * k * (n - k) / (n - 1.0)
    ratio = float(np.max(prefix / bounds))
    # A small inward margin prevents a floating-point excess at a face.
    return d / max(1.0, ratio * (1.0 + 1e-12))


def fit_rank_convex(
    X: np.ndarray,
    y: np.ndarray,
    *,
    penalty: str = "none",
    lam: float = 0.0,
    x0: np.ndarray | None = None,
    maxiter: int = 20_000,
    atol: float = 1e-6,
    rtol: float = 1e-5,
    rho: float | None = None,
    adaptive_rho: bool = True,
) -> tuple[np.ndarray, dict]:
    """Fit rank regression, returning coefficients and explicit diagnostics.

    ``rho`` is measured in the original, unmultiplied objective (default 1/n).
    Its value is adjusted every 25 iterations by balancing residuals normalized
    by their respective stopping tolerances, unless ``adaptive_rho=False``.
    The latter holds rho at its initial value without changing stopping tests.
    ``success`` requires both ADMM residual tests and an objective-gap test.
    Centering X and y exploits residual-shift invariance and is exact for this
    objective; it does not add an estimated intercept to the returned vector.
    Objective accuracy does not by itself bound coefficient error in an
    ill-conditioned design.  Unpenalized fits require centered full column rank.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim != 2 or y.shape != (X.shape[0],):
        raise ValueError("X and y have incompatible shapes.")
    n, p = X.shape
    if n < 2 or p < 1 or not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
        raise ValueError("X and y must be finite, with n >= 2 and p >= 1.")
    if penalty not in {"none", "ridge", "lasso"}:
        raise ValueError("penalty must be none, ridge, or lasso.")
    if not np.isfinite(lam) or lam < 0 or (penalty == "none" and lam != 0):
        raise ValueError("lam must be nonnegative and must be zero without a penalty.")
    if maxiter < 1 or not np.isfinite(atol) or not np.isfinite(rtol) or atol <= 0 or rtol <= 0:
        raise ValueError("maxiter, atol, and rtol must be positive.")
    rho_work = 1.0 if rho is None else float(rho) * n
    if not np.isfinite(rho_work) or rho_work <= 0:
        raise ValueError("rho must be finite and positive.")
    if lam == 0:
        penalty = "none"
    Xc = np.asfortranarray(X - X.mean(axis=0))
    yc = y - y.mean()
    beta = np.zeros(p) if x0 is None else np.asarray(x0, dtype=float).copy()
    if beta.shape != (p,) or not np.all(np.isfinite(beta)):
        raise ValueError("x0 must be a finite coefficient vector of length p.")
    gap = float("inf")
    dual_objective = float("nan")
    objective = float("nan")
    gap_tolerance = float("nan")
    success = False

    if penalty == "lasso" and p >= n - 1 and lam <= 0.1:
        candidate, cached = _basis_pursuit(Xc, yc)
        if candidate is not None:
            bp_beta, bp_dual, bp_iterations = candidate
            d = _dual_rank_feasible(lam * bp_dual)
            d /= max(1.0, np.max(np.abs(Xc.T @ d)) / lam) * (1.0 + 1e-12)
            res = yc - Xc @ bp_beta
            obj = n * rank_loss(res) + lam * np.abs(bp_beta).sum()
            dual = float(yc @ d)
            bp_gap = max(0.0, float(obj - dual))
            bp_gap_tolerance = atol + rtol * max(1.0, abs(obj), abs(dual))
            bp_primal = float(np.linalg.norm(res - res.mean()))
            bp_primal_tolerance = np.sqrt(n) * atol + rtol * np.linalg.norm(yc)
            xtd = Xc.T @ d
            active = np.abs(bp_beta) > 1e-10
            kkt = np.maximum(np.abs(xtd) - lam, 0.0)
            kkt[active] = xtd[active] - lam * np.sign(bp_beta[active])
            bp_dual_residual = float(np.linalg.norm(kkt))
            bp_dual_tolerance = np.sqrt(p) * atol + rtol * max(lam * np.sqrt(p), np.linalg.norm(xtd))
            if bp_gap <= bp_gap_tolerance and bp_primal <= bp_primal_tolerance and bp_dual_residual <= bp_dual_tolerance:
                return bp_beta.copy(), {
                    "success": True, "status": "certified_zero_rank_loss_lasso",
                    "iterations": bp_iterations, "objective": float(obj / n),
                    "dual_objective": dual / n, "duality_gap": bp_gap / n,
                    "gap_tolerance": float(bp_gap_tolerance / n),
                    "primal_residual": bp_primal, "dual_residual": bp_dual_residual / n,
                    "primal_tolerance": float(bp_primal_tolerance),
                    "dual_tolerance": float(bp_dual_tolerance / n), "rho": rho_work / n,
                    "method": "basis_pursuit_with_rank_dual_certificate", "lp_cache_hit": cached,
                }

    if penalty == "none":
        if p >= n:
            raise ValueError("Unpenalized fits require p < n and centered full column rank.")
        # Pivoting makes rank detection reliable.  For p near n, the orthogonal
        # complement makes each projection O(n(n-p)) rather than O(np).
        Q, R, piv = qr(Xc, mode="full" if p > n // 2 else "economic", pivoting=True)
        Rtop = R[:p, :]
        diag = np.abs(np.diag(Rtop))
        threshold = np.finfo(float).eps * max(n, p) * max(float(diag.max()), 1.0)
        if np.any(diag <= threshold):
            raise ValueError("The centered design is numerically rank deficient.")
        Qp = Q[:, :p]
        if p > n // 2:
            Qperp = Q[:, p:]

            def project(v):
                return v - Qperp @ (Qperp.T @ v)
        else:

            def project(v):
                return Qp @ (Qp.T @ v)

        def recover_beta(w):
            result = np.empty(p)
            result[piv] = solve_triangular(Rtop, Qp.T @ w)
            return result

        if p == n - 1:
            # The centered column space is the whole mean-zero subspace.
            beta = recover_beta(yc)
            res = yc - Xc @ beta
            obj = rank_loss(res)
            threshold_res = np.sqrt(n) * atol + rtol * np.linalg.norm(yc)
            success = bool(np.linalg.norm(res) <= threshold_res and obj <= (atol + rtol) / n)
            return beta, {
                "success": success, "status": "zero_loss_solution" if success else "linear_solve_residual",
                "iterations": 0, "objective": obj, "dual_objective": 0.0,
                "duality_gap": obj, "gap_tolerance": (atol + rtol) / n,
                "primal_residual": float(np.linalg.norm(res)), "dual_residual": 0.0,
                "primal_tolerance": float(threshold_res), "dual_tolerance": float(np.sqrt(n) * atol),
                "rho": rho_work / n, "method": "centered_qr_zero_rank_loss",
            }

        w = Xc @ beta
        r = yc - w
        u = np.zeros(n)
        for iteration in range(1, maxiter + 1):
            w = project(yc - r - u)
            previous_r = r
            r = rank_prox(yc - w - u, 1.0 / rho_work, scaled_by_n=True)
            residual = w + r - yc
            u += residual
            primal_norm = float(np.linalg.norm(residual))
            dual_norm = float(rho_work * np.linalg.norm(r - previous_r))
            eps_primal = np.sqrt(n) * atol + rtol * max(np.linalg.norm(w), np.linalg.norm(r), np.linalg.norm(yc))
            eps_dual = np.sqrt(n) * atol + rtol * rho_work * np.linalg.norm(u)
            if iteration % 10 == 0 or (primal_norm <= eps_primal and dual_norm <= eps_dual) or iteration == maxiter:
                d = -rho_work * u
                d = _dual_rank_feasible(d - project(d))
                objective = n * rank_loss(yc - w)
                dual_objective = float(yc @ d)
                gap = max(0.0, objective - dual_objective)
                gap_tolerance = atol + rtol * max(1.0, abs(objective), abs(dual_objective))
                if primal_norm <= eps_primal and dual_norm <= eps_dual and gap <= gap_tolerance:
                    success = True
                    break
            if adaptive_rho and iteration % 25 == 0:
                old_rho = rho_work
                if primal_norm / eps_primal > 10.0 * dual_norm / eps_dual:
                    rho_work = min(2.0 * rho_work, 1e6)
                elif dual_norm / eps_dual > 10.0 * primal_norm / eps_primal:
                    rho_work = max(rho_work / 2.0, 1e-6)
                u *= old_rho / rho_work
        beta = recover_beta(w)
    else:
        # X'X+I is independent of rho.  Use the smaller Cholesky factor.
        if p <= n:
            factor = cho_factor(Xc.T @ Xc + np.eye(p), lower=True, check_finite=False)

            def solve_beta(rhs):
                return cho_solve(factor, rhs, check_finite=False)
        else:
            factor = cho_factor(Xc @ Xc.T + np.eye(n), lower=True, check_finite=False)

            def solve_beta(rhs):
                return rhs - Xc.T @ cho_solve(factor, Xc @ rhs, check_finite=False)

        r = yc - Xc @ beta
        z = beta.copy()
        u = np.zeros(n)
        v = np.zeros(p)
        for iteration in range(1, maxiter + 1):
            beta = solve_beta(Xc.T @ (yc - r - u) + z - v)
            w = Xc @ beta
            previous_r, previous_z = r, z
            r = rank_prox(yc - w - u, 1.0 / rho_work, scaled_by_n=True)
            if penalty == "ridge":
                z = (beta + v) / (1.0 + lam / rho_work)
            else:
                z = np.sign(beta + v) * np.maximum(np.abs(beta + v) - lam / rho_work, 0.0)
            residual_r, residual_z = w + r - yc, beta - z
            u += residual_r
            v += residual_z
            primal_norm = float(np.hypot(np.linalg.norm(residual_r), np.linalg.norm(residual_z)))
            dual_norm = float(rho_work * np.linalg.norm(Xc.T @ (r - previous_r) - (z - previous_z)))
            eps_primal = np.sqrt(n + p) * atol + rtol * max(np.hypot(np.linalg.norm(w), np.linalg.norm(beta)), np.hypot(np.linalg.norm(r), np.linalg.norm(z)), np.linalg.norm(yc))
            eps_dual = np.sqrt(p) * atol + rtol * rho_work * np.linalg.norm(Xc.T @ u + v)
            if iteration % 10 == 0 or (primal_norm <= eps_primal and dual_norm <= eps_dual) or iteration == maxiter:
                d = _dual_rank_feasible(-rho_work * u)
                xtd = Xc.T @ d
                if penalty == "lasso":
                    dual_scale = max(1.0, float(np.max(np.abs(xtd))) / lam)
                    d /= dual_scale * (1.0 + 1e-12)
                    dual_objective = float(yc @ d)
                    regularizer = lam * np.sum(np.abs(beta))
                else:
                    dual_objective = float(yc @ d - np.dot(xtd, xtd) / (2.0 * lam))
                    regularizer = lam * np.dot(beta, beta) / 2.0
                objective = n * rank_loss(yc - w) + regularizer
                gap = max(0.0, objective - dual_objective)
                gap_tolerance = atol + rtol * max(1.0, abs(objective), abs(dual_objective))
                if primal_norm <= eps_primal and dual_norm <= eps_dual and gap <= gap_tolerance:
                    success = True
                    break
            if adaptive_rho and iteration % 25 == 0:
                old_rho = rho_work
                if primal_norm / eps_primal > 10.0 * dual_norm / eps_dual:
                    rho_work = min(2.0 * rho_work, 1e6)
                elif dual_norm / eps_dual > 10.0 * primal_norm / eps_primal:
                    rho_work = max(rho_work / 2.0, 1e-6)
                u *= old_rho / rho_work
                v *= old_rho / rho_work

    return beta, {
        "success": success, "status": "converged" if success else "max_iterations",
        "iterations": iteration, "objective": float(objective / n),
        "dual_objective": float(dual_objective / n), "duality_gap": float(gap / n),
        "gap_tolerance": float(gap_tolerance / n), "primal_residual": primal_norm,
        "dual_residual": dual_norm / n, "primal_tolerance": float(eps_primal),
        "dual_tolerance": float(eps_dual / n), "rho": rho_work / n,
        "method": "two_block_admm_isotonic_prox",
        "rho_schedule": "residual_balancing" if adaptive_rho else "constant",
    }
