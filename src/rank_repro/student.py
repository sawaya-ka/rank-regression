"""Unpenalized risk under standard Student-t noise, with delta = n/p.

The scale is one (not variance-standardized). Quadrature represents the t
law as a Gaussian scale mixture. The order controls numerical integration,
not Monte Carlo precision. Near delta=1 and small degrees of freedom,
quadrature convergence can be slow; increase the order and compare results.
The archived 100-node phase grid is historical output, not certified accuracy.
"""

import numpy as np
from scipy.optimize import brentq
from functools import lru_cache
from scipy.special import erf, roots_legendre
from scipy.stats import norm, chi2, t as student_t


def _m0_standard_normal(t):
    """
    m0(t) = E[min{X^2, t^2}], X ~ N(0,1).

    Formula:
        m0(t) = 2 Phi(t) - 1 - 2 t phi(t) + 2 t^2 Phi_bar(t)

    A small-t expansion is used to avoid cancellation.
    """
    t = np.asarray(t, dtype=float)

    val = (
        2.0 * (norm.cdf(t) - 0.5)
        - 2.0 * t * norm.pdf(t)
        + 2.0 * t**2 * norm.sf(t)
    )

    # Small-t expansion:
    # m0(t) = t^2 - (4/3) phi(0) t^3 + (2/15) phi(0) t^5 + O(t^7)
    phi0 = 1.0 / np.sqrt(2.0 * np.pi)
    series = t**2 - (4.0 / 3.0) * phi0 * t**3 + (2.0 / 15.0) * phi0 * t**5

    return np.where(t < 1e-5, series, val)


def alpha2_lad_standard_t(
    delta,
    df,
    *,
    n_quad=400,
    xtol=1e-10,
    rtol=1e-10,
    return_kappa=False,
):
    """
    Compute alpha^2 for LAD regression under standard Student-t error.

    Model:
        Z ~ t_df, standard Student-t with location 0 and scale 1.
        V_alpha = alpha G + Z, G ~ N(0,1).

    Self-consistency:
        P(|V_alpha| <= kappa) = 1 / delta
        alpha^2 = delta * E[min{V_alpha^2, kappa^2}]

    Parameters
    ----------
    delta : float
        Aspect ratio. Must satisfy delta > 1.

    df : float
        Degrees of freedom of the standard Student-t distribution.
        Must satisfy df > 0.

    n_quad : int, optional
        Number of Gauss-Legendre quadrature nodes used for integrating over
        U ~ chi-square(df). Larger values are more accurate but slower.
        For heavy-tailed cases such as df <= 1, n_quad=400 or larger is safer.

    xtol, rtol : float, optional
        Tolerances for Brent root finding.

    return_kappa : bool, optional
        If True, return (alpha^2, kappa). Otherwise return alpha^2.

    Returns
    -------
    alpha2 : float
        The asymptotic squared error alpha^2.

    kappa : float, optional
        The LAD threshold kappa, returned only if return_kappa=True.
    """
    if delta <= 1:
        raise ValueError("delta must be larger than 1.")
    if df <= 0:
        raise ValueError("df must be positive.")
    if n_quad < 20:
        raise ValueError("n_quad should be at least 20.")

    q = 1.0 / delta

    # We compute E[h(U)] for U ~ chi-square(df) by
    # E[h(U)] = integral_0^1 h(chi2.ppf(p; df)) dp.
    #
    # Gauss-Legendre quadrature on [0,1].
    weights, u_nodes = _student_chi2_nodes(float(df), int(n_quad))

    sqrt2 = np.sqrt(2.0)

    def H_abs(kappa, alpha):
        """
        H(kappa; alpha) = P(|V_alpha| <= kappa).
        Conditional on U,
            V_alpha | U ~ N(0, alpha^2 + df / U).
        """
        s = np.sqrt(alpha**2 + df / u_nodes)

        # 2 Phi(kappa/s) - 1 = erf(kappa / (sqrt(2) s))
        return np.sum(weights * erf(kappa / (sqrt2 * s)))

    def M_trunc(kappa, alpha):
        """
        M(kappa; alpha) = E[min{V_alpha^2, kappa^2}].
        Conditional on U, V_alpha = s X with X ~ N(0,1).
        Hence
            E[min{V_alpha^2, kappa^2} | U]
            = s^2 m0(kappa / s).
        """
        s2 = alpha**2 + df / u_nodes
        s = np.sqrt(s2)
        t = kappa / s
        return np.sum(weights * s2 * _m0_standard_normal(t))

    def kappa_for_alpha(alpha):
        """
        Solve H_abs(kappa, alpha) = 1 / delta.
        """
        p0 = (1.0 + q) / 2.0

        # A useful initial upper bound based on the pure t quantile.
        # We enlarge it dynamically if needed.
        hi = student_t.ppf(p0, df=df) + norm.ppf(p0) * alpha + 1e-12
        if not np.isfinite(hi) or hi <= 0:
            hi = 1.0 + alpha

        while H_abs(hi, alpha) < q:
            hi *= 2.0
            if hi > 1e100:
                raise RuntimeError("Failed to bracket kappa.")

        return brentq(
            lambda k: H_abs(k, alpha) - q,
            0.0,
            hi,
            xtol=xtol,
            rtol=rtol,
        )

    def residual(alpha):
        """
        R(alpha) = delta M(kappa(alpha); alpha) - alpha^2.
        """
        kappa = kappa_for_alpha(alpha)
        return delta * M_trunc(kappa, alpha) - alpha**2

    # R(0) > 0 for non-degenerate continuous noise.
    # For large alpha, R(alpha) < 0, so we bracket by doubling.
    lo = 0.0
    hi = 1.0

    while residual(hi) > 0:
        hi *= 2.0
        if hi > 1e100:
            raise RuntimeError("Failed to bracket alpha.")

    alpha = brentq(
        residual,
        lo,
        hi,
        xtol=xtol,
        rtol=rtol,
    )

    alpha2 = alpha**2

    if return_kappa:
        kappa = kappa_for_alpha(alpha)
        return alpha2, kappa

    return alpha2

from math import sqrt, pi
from scipy.special import roots_legendre, ndtr, erf
from scipy.stats import gamma as gamma_dist

_INV_SQRT_2PI = 1.0 / sqrt(2.0 * pi)
_SQRT2 = sqrt(2.0)

def _Phi_minus_half(x):
    """
    Stable computation of Phi(x) - 1/2.
    """
    return 0.5 * erf(x / _SQRT2)

def _phi(x):
    """
    Standard normal density.
    """
    return _INV_SQRT_2PI * np.exp(-0.5 * x * x)

def has_valid_c0_branch(solver, delta):
    """
    Return True iff the c=0 branch exists and is consistent at this delta.
    """
    cand = solver.c0_candidate(delta)
    if cand is None:
        return False
    _, _, valid, _ = cand
    return bool(valid)

def find_c0_boundary_delta(
    solver,
    delta_lo=1.0001,
    delta_hi=10.0,
    tol=1e-6,
    maxiter=80,
):
    """
    Find the boundary delta_* where the solution changes from c>0 to c=0.

    Returns
    -------
    delta_star : float or None
        Approximate boundary. Returns None if no valid c=0 branch appears
        in [delta_lo, delta_hi].
    """
    # If even at delta_hi the c=0 branch is not valid, no boundary in range.
    if not has_valid_c0_branch(solver, delta_hi):
        return None

    # If already valid at delta_lo, boundary is at or below delta_lo.
    if has_valid_c0_branch(solver, delta_lo):
        return delta_lo

    lo, hi = delta_lo, delta_hi
    for _ in range(maxiter):
        mid = 0.5 * (lo + hi)
        if has_valid_c0_branch(solver, mid):
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            break

    return hi

class RankRegressionStudentT:
    """
    Solver for the scalar fixed point of rank regression
    under standard Student-t noise.

    Noise model:
        Z ~ t_df with scale 1.

    Proportional regime:
        n / p -> delta > 1.
    """

    def __init__(self, df, n_quad=180):
        if df <= 0:
            raise ValueError("df must be positive.")
        self.df = float(df)
        self.n_quad = int(n_quad)

        # Student t as Gaussian scale mixture:
        #
        #   Z | Lambda=lambda ~ N(0, 1/lambda),
        #   Lambda ~ Gamma(df/2, rate=df/2).
        #
        shape = self.df / 2.0
        rate = self.df / 2.0

        # Integrate E[h(Lambda)] using Gauss-Legendre over u in (0,1):
        #
        #   Lambda = Gamma^{-1}(u).
        #
        # This is often more stable for small df than ordinary Gauss-Laguerre
        # quadrature, because small df puts substantial mass in the tail.
        x, w = roots_legendre(self.n_quad)
        u = 0.5 * (x + 1.0)
        self.w = 0.5 * w
        self.lam = gamma_dist.ppf(u, a=shape, scale=1.0 / rate)
        self.W2 = self.w[:, None] * self.w[None, :]

    def _s2(self, alpha):
        """
        Conditional variance:
            s_lambda^2 = alpha^2 + 1/lambda.
        """
        return alpha * alpha + 1.0 / self.lam

    def A(self, alpha, c):
        """
        A_alpha(c) = F_V(c) - 1/2.
        """
        s = np.sqrt(self._s2(alpha))
        return float(np.dot(self.w, _Phi_minus_half(c / s)))

    def f0(self, alpha):
        """
        f_V(0).
        """
        s = np.sqrt(self._s2(alpha))
        return float(np.dot(self.w, _INV_SQRT_2PI / s))

    def K(self, alpha, c):
        """
        K_alpha(c) = int_0^c v^2 f_V(v) dv.
        """
        s2 = self._s2(alpha)
        s = np.sqrt(s2)
        a = c / s
        vals = s2 * (_Phi_minus_half(a) - a * _phi(a))
        return float(np.dot(self.w, vals))

    def I(self, alpha):
        """
        I_alpha = int f_V(v)^2 dv.
        """
        s2 = self._s2(alpha)
        sum_s2 = s2[:, None] + s2[None, :]
        vals = 1.0 / np.sqrt(2.0 * pi * sum_s2)
        return float(np.sum(self.W2 * vals))

    def J(self, alpha, c):
        """
        J_alpha(c) = int_c^infty f_V(v)^2 dv.
        """
        if c == 0:
            return 0.5 * self.I(alpha)

        s2 = self._s2(alpha)
        sum_s2 = s2[:, None] + s2[None, :]
        const = 1.0 / np.sqrt(2.0 * pi * sum_s2)

        r = np.sqrt((s2[:, None] * s2[None, :]) / sum_s2)
        vals = const * ndtr(-c / r)
        return float(np.sum(self.W2 * vals))

    def c0_candidate(self, delta):
        """
        Solve the c=0 branch candidate if it exists.
        If it does not exist, return None.
        """
        def g(alpha):
            I = self.I(alpha)
            return alpha * alpha - 1.0 / (12.0 * delta * I * I)

        hi = 1.0
        while g(hi) <= 0.0:
            hi *= 2.0

            # For small delta, the c=0 candidate equation may have no solution.
            # In that case, give up and move to the c>0 branch.
            if hi > 1e6:
                return None

        alpha = brentq(g, 0.0, hi, xtol=1e-11, rtol=1e-11, maxiter=200)

        I = self.I(alpha)
        f0 = self.f0(alpha)
        kappa = 1.0 / (4.0 * delta * I)

        threshold_value = f0 / I
        valid = delta >= threshold_value * (1.0 - 1e-10)

        return alpha, kappa, valid, threshold_value

    def _h_for_c_equation(self, alpha, c):
        """
        h_alpha(c) = 2 A + 2 c J/A.

        The c-equation is:
            h_alpha(c) = 1/delta.
        """
        if c == 0.0:
            return self.I(alpha) / self.f0(alpha)

        A = self.A(alpha, c)
        return 2.0 * A + 2.0 * c * self.J(alpha, c) / A

    def _solve_c_given_alpha(self, alpha, delta):
        """
        For fixed alpha, solve

            1/delta = 2 A_alpha(c) + 2 c J_alpha(c)/A_alpha(c)

        for c > 0.
        """
        target = 1.0 / delta
        h0 = self.I(alpha) / self.f0(alpha)

        if target <= h0:
            raise RuntimeError(
                "For this alpha, the positive-c equation has no positive root."
            )

        def eq(c):
            return self._h_for_c_equation(alpha, c) - target

        hi = max(1.0, alpha + 1.0)
        while eq(hi) <= 0.0:
            hi *= 2.0
            if hi > 1e8:
                raise RuntimeError("Failed to bracket c.")

        return brentq(eq, 0.0, hi, xtol=1e-10, rtol=1e-10, maxiter=200)

    def _alpha_residual_c_positive(self, alpha, delta):
        """
        Residual for the second equation after eliminating c by the first equation.
        """
        c = self._solve_c_given_alpha(alpha, delta)

        A = self.A(alpha, c)
        K = self.K(alpha, c)

        rhs = (
            2.0 * K
            + c * c / (12.0 * A * A)
            - (2.0 / 3.0) * c * c * A
        )

        return alpha * alpha / delta - rhs

    def solve(self, delta):
        """
        Safer solver:
        first attempts c=0 branch;
        if unavailable or inconsistent, solves the c>0 branch.
        """
        if delta <= 1.0:
            raise ValueError("This proportional-limit formula assumes delta > 1.")

        # Try c=0 branch.
        c0 = self.c0_candidate(delta)

        if c0 is not None:
            alpha0, kappa0, valid_c0, threshold_value = c0

            if valid_c0:
                return {
                    "alpha": alpha0,
                    "alpha2": alpha0 * alpha0,
                    "kappa": kappa0,
                    "c": 0.0,
                    "branch": "c=0",
                    "c0_consistency_threshold": threshold_value,
                }

            alpha_start = max(1.0, alpha0)
        else:
            threshold_value = np.nan
            alpha_start = 1.0

        # Otherwise solve c>0 branch.
        def res(alpha):
            return self._alpha_residual_c_positive(alpha, delta)

        # Find a lower point where residual is finite and negative.
        lo = 0.0
        f_lo = res(lo)

        # In rare numerical cases, move lo slightly.
        if not np.isfinite(f_lo):
            lo = 1e-8
            f_lo = res(lo)

        # Find upper bracket.
        hi = alpha_start
        f_hi = res(hi)

        while (not np.isfinite(f_hi)) or f_hi <= 0.0:
            hi *= 2.0
            f_hi = res(hi)

            if hi > 1e8:
                raise RuntimeError("Failed to bracket alpha on the c>0 branch.")

        alpha = brentq(res, lo, hi, xtol=1e-9, rtol=1e-9, maxiter=200)

        c = self._solve_c_given_alpha(alpha, delta)
        A = self.A(alpha, c)
        kappa = c / (4.0 * A)

        return {
            "alpha": alpha,
            "alpha2": alpha * alpha,
            "kappa": kappa,
            "c": c,
            "branch": "c>0",
            "c0_consistency_threshold": threshold_value,
        }

def alpha2_rank_student_t(delta, df, n_quad=180, return_full=False):
    """
    Convenience wrapper.

    Parameters
    ----------
    delta : float
        Aspect ratio n/p. Must be > 1.
    df : float
        Degrees of freedom of the standard Student-t noise.
    n_quad : int
        Quadrature order. Increase to 250--400 when df is very small
        or delta is very close to 1.
    return_full : bool
        If True, return full solution dictionary.

    Returns
    -------
    float or dict
        alpha^2, or full solution dictionary.
    """
    solver = RankRegressionStudentT(df=df, n_quad=n_quad)
    sol = solver.solve(delta)
    return sol if return_full else sol["alpha2"]


@lru_cache(maxsize=64)
def _student_chi2_nodes(df, order):
    x, weights = roots_legendre(order)
    return 0.5 * weights, chi2.ppf(0.5 * (x + 1.0), df=df)


def ls_risk(delta, df):
    """Normalized limiting squared error; infinity when the t variance is infinite."""
    if not np.isfinite(delta) or delta <= 1 or not np.isfinite(df) or df <= 0:
        raise ValueError("Require finite delta>1 and df>0.")
    return np.inf if df <= 2 else df / ((df - 2) * (delta - 1))


def risk_curve(df, deltas, order=300):
    """Return LS, LAD and rank squared errors on a strictly supercritical grid."""
    solver = RankRegressionStudentT(df, n_quad=order)
    rows = []
    for delta in np.asarray(deltas, dtype=float):
        rank = solver.solve(float(delta))
        rows.append((float(delta), ls_risk(float(delta), df),
                     alpha2_lad_standard_t(float(delta), df, n_quad=order),
                     rank["alpha2"], rank["c"]))
    return np.asarray(rows)


import pandas as pd
from scipy.special import betaln
from scipy.interpolate import RegularGridInterpolator

def classical_risks(nu):
    """Return (LS, LAD, Rank) classical constants, not their reciprocals."""
    nu = np.asarray(nu, dtype=float)
    if np.any(~np.isfinite(nu)) or np.any(nu <= 0):
        raise ValueError("Degrees of freedom must be finite and positive.")
    log_b = betaln(0.5, nu / 2)
    # f_nu(0) = 1 / [sqrt(nu) B(1/2, nu/2)].
    # I_nu = integral f_nu^2 = B(1/2, nu+1/2) / [sqrt(nu) B(1/2, nu/2)^2].
    log_i = betaln(0.5, nu + 0.5) - 0.5 * np.log(nu) - 2 * log_b
    ls = np.full_like(nu, np.inf)
    np.divide(nu, nu - 2, out=ls, where=nu > 2)
    lad = np.exp(np.log(nu) + 2 * log_b - np.log(4.0))
    rank = np.exp(-np.log(12.0) - 2 * log_i)
    return np.stack([ls, lad, rank])


def _classical_regions(xmin, xmax):
    """Locate changes of the lowest classical risk without fixed cutoffs."""
    nu = np.geomspace(xmin, xmax, 4000)
    winner = np.argmin(classical_risks(nu), axis=0)
    changes = np.flatnonzero(np.diff(winner))
    edges = [xmin]
    methods = [int(winner[0])]
    for k in changes:
        a, b = int(winner[k]), int(winner[k + 1])
        def difference(z):
            risk = classical_risks(z)
            return np.log(risk[a]) - np.log(risk[b])
        edges.append(brentq(difference, nu[k], nu[k + 1]))
        methods.append(b)
    edges.append(xmax)
    return list(zip(edges[:-1], edges[1:], methods))


def load_interpolated_risks(csv_path, fine_n_df=500, fine_n_delta=500):
    """Interpolate scaled log risks; never turn failed solves into large risks.

    Multiply all risks by delta-1 before interpolation to remove the known
    LS inflation factor. Undo this scaling afterwards. At any fixed delta
    the common positive scaling leaves the winner unchanged.
    """
    data = pd.read_csv(csv_path)
    required = {"df", "delta", "alpha_LAD", "alpha_rank"}
    if not required.issubset(data.columns):
        raise ValueError(f"Missing CSV columns: {sorted(required - set(data.columns))}")
    if data.duplicated(["df", "delta"]).any():
        raise ValueError("Duplicate (df, delta) coordinates in phase grid.")
    coords = data[["df", "delta"]].to_numpy(dtype=float)
    if not np.isfinite(coords).all() or np.any(coords[:, 0] <= 0) or np.any(coords[:, 1] <= 1):
        raise ValueError("This unpenalized plot requires finite df>0 and delta>1.")
    nu = np.sort(data["df"].unique().astype(float))
    delta = np.sort(data["delta"].unique().astype(float))
    if min(len(nu), len(delta), fine_n_df, fine_n_delta) < 2:
        raise ValueError("At least two points are required along each axis.")
    log_nu, log_delta = np.log(nu), np.log(delta)
    xx, yy = np.meshgrid(
        np.linspace(log_nu[0], log_nu[-1], fine_n_df),
        np.linspace(log_delta[0], log_delta[-1], fine_n_delta),
    )
    points = np.column_stack([yy.ravel(), xx.ravel()])
    x, y = np.exp(xx), np.exp(yy)
    log_risks = []
    for column in ("alpha_LAD", "alpha_rank"):
        a = (data.pivot(index="delta", columns="df", values=column)
             .reindex(index=delta, columns=nu).to_numpy(dtype=float))
        bad = ~np.isfinite(a) | (a <= 0)
        if bad.any():
            examples = [(float(nu[j]), float(delta[i]))
                        for i, j in np.argwhere(bad)[:5]]
            raise ValueError(
                f"{column}: {bad.sum()} missing/nonpositive/nonfinite grid values. "
                f"Example (df, delta): {examples}. Recompute these CSV entries."
            )
        interp = RegularGridInterpolator(
            (log_delta, log_nu), 2 * np.log(a) + np.log(delta[:, None] - 1),
            method="linear", bounds_error=True,
        )
        log_risks.append(interp(points).reshape(x.shape) - np.log(y - 1))
    # Exact LS risk: nu / [(nu-2)(delta-1)], for nu>2 only.
    # Its infinite-variance domain is not a missing-data placeholder.
    log_ls = np.full_like(x, np.inf)
    finite_ls = x > 2
    log_ls[finite_ls] = (np.log(x[finite_ls]) - np.log(x[finite_ls] - 2)
                         - np.log(y[finite_ls] - 1))
    return x, y, np.stack([log_ls, *log_risks])



def solve_student_t(df, delta, n_quad=300):
    """One row in the archived phase-grid schema; no Monte Carlo is used."""
    rank = RankRegressionStudentT(df, n_quad=n_quad).solve(delta)
    risks = np.asarray([ls_risk(delta, df),
                        alpha2_lad_standard_t(delta, df, n_quad=n_quad),
                        rank["alpha2"]])
    winner = int(np.argmin(risks))
    row = {"df": float(df), "delta": float(delta), "quadrature_order": int(n_quad),
           "winner": ("LS", "LAD", "rank")[winner], "winner_code": winner,
           "rank_branch": rank["branch"], "rank_cutoff": rank["c"]}
    for name, risk in zip(("LS", "LAD", "rank"), risks):
        row[f"alpha2_{name}"] = float(risk)
        row[f"alpha_{name}"] = float(np.sqrt(risk))
    return row
