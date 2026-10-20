"""Penalized scalar-risk equations, fixed-penalty solvers, and bounded searches."""

from __future__ import annotations
from typing import Optional, Any
import numpy as np
from scipy.optimize import brentq, least_squares, minimize_scalar
from scipy.stats import norm, t as student_t
from scipy.special import roots_genlaguerre, gamma, erf
from numpy.polynomial.hermite import hermgauss


DEFAULT_GH_N = 40


DEFAULT_LAD_QUAD_MULTIPLIER = 1


DEFAULT_LAD_QUAD_MIN = 40


GH_Z = None


GH_WEIGHTS = None


def good_solution(sol: Optional[dict[str, Any]], cost_tol: float = 1e-5) -> bool:
    if sol is None:
        return False
    a2 = sol.get('alpha2', np.nan)
    if not (np.isfinite(a2) and a2 > 0.0):
        return False
    cost = sol.get('cost', 0.0)
    if not np.isfinite(cost) or cost > cost_tol:
        return False
    return True


def output_df_value(df: int | float) -> int | float:
    """Keep integer dfs as ints, but allow non-integer df grids in CSV output."""
    df_float = float(df)
    nearest = round(df_float)
    if abs(df_float - nearest) < 1e-12:
        return int(nearest)
    return df_float


def finite_positive(x: Any) -> bool:
    try:
        return bool(np.isfinite(float(x)) and float(x) > 0.0)
    except Exception:
        return False


def solution_lambda(sol: Optional[dict[str, Any]]) -> Optional[float]:
    if sol is None:
        return None
    lam = sol.get('lambda', sol.get('lambda_opt', np.nan))
    if finite_positive(lam):
        return float(lam)
    return None


def make_focused_log_lambda_grid(lambda_min: float, lambda_max: float, n_grid: int,
                                 lambda_hint: Optional[float] = None) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Global + local log-lambda grid.

    The grid always includes the boundaries and a global log-spaced grid.  If a
    previous neighboring optimum lambda is available, additional points are
    concentrated around it.  This gives global coverage while exploiting the
    smoothness of lambda*(delta,df).
    """
    lo = float(np.log(lambda_min))
    hi = float(np.log(lambda_max))
    n_grid = int(max(9, n_grid))

    if lambda_hint is not None and finite_positive(lambda_hint):
        hint_log = float(np.clip(np.log(float(lambda_hint)), lo, hi))
        n_local = max(5, n_grid // 2)
        n_base = max(7, n_grid - n_local)
        base = np.linspace(lo, hi, n_base)
        half_width = min(4.0, 0.5 * (hi - lo))
        local = hint_log + np.linspace(-half_width, half_width, n_local)
        logs = np.concatenate([base, local, [lo, hi, hint_log]])
        target_log = hint_log
    else:
        logs = np.linspace(lo, hi, n_grid)
        target_log = float(np.clip(0.0, lo, hi))

    logs = logs[np.isfinite(logs)]
    logs = np.clip(logs, lo, hi)
    logs = np.unique(np.round(logs, 14))
    logs.sort()
    return logs.astype(float), np.exp(logs).astype(float), float(target_log)


def t_scale(df: int | float, standardize: bool = False) -> float:
    if standardize:
        if df <= 2:
            raise ValueError("Cannot variance-standardize Student-t with df <= 2.")
        return float(np.sqrt((df - 2.0) / df))
    return 1.0


def t_pdf(z: np.ndarray, df: int | float, standardize: bool = False) -> np.ndarray:
    s = t_scale(df, standardize)
    return student_t.pdf(z / s, df) / s


def t_cdf(z: np.ndarray, df: int | float, standardize: bool = False) -> np.ndarray:
    s = t_scale(df, standardize)
    return student_t.cdf(z / s, df)


def robust_xmax(alpha: float, df: int | float, kappa: float = 0.0, standardize: bool = False) -> float:
    """A conservative positive grid endpoint that also works for df=1,2."""
    s = t_scale(df, standardize)
    try:
        q995 = float(student_t.ppf(0.995, df)) * s
        if not np.isfinite(q995):
            q995 = 80.0 * s
    except Exception:
        q995 = 80.0 * s

    return float(max(
        40.0 * s,
        2.5 * q995 + 10.0,
        12.0 * max(alpha, 1.0) + 10.0,
        2.5 * max(kappa, 0.0) + 10.0,
    ))


def init_gh(gh_n: int) -> None:
    global GH_Z, GH_WEIGHTS
    x, w = hermgauss(gh_n)
    GH_Z = np.sqrt(2.0) * x
    GH_WEIGHTS = w / np.sqrt(np.pi)


def fV_FV_on_grid(alpha: float, xgrid: np.ndarray, df: int, standardize_t: bool) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes f_V(x), F_V(x) for V = alpha G + Z.
    Uses
        f_V(x) = E_G f_Z(x - alpha G),
        F_V(x) = E_G F_Z(x - alpha G).
    """
    xgrid = np.asarray(xgrid, dtype=float)
    if GH_Z is None or GH_WEIGHTS is None:
        raise RuntimeError("Gauss-Hermite nodes are not initialized. Call init_gh().")

    if alpha < 1e-12:
        return t_pdf(xgrid, df, standardize_t), t_cdf(xgrid, df, standardize_t)

    arg = xgrid[None, :] - alpha * GH_Z[:, None]
    f = np.sum(GH_WEIGHTS[:, None] * t_pdf(arg, df, standardize_t), axis=0)
    F = np.sum(GH_WEIGHTS[:, None] * t_cdf(arg, df, standardize_t), axis=0)
    return f, F


def _m0_standard_normal(t: np.ndarray | float) -> np.ndarray:
    """
    m0(t) = E[min{X^2, t^2}], X ~ N(0,1).

    m0(t) = 2 Phi(t)-1 - 2 t phi(t) + 2 t^2 Phi_bar(t).
    """
    t = np.asarray(t, dtype=float)
    val = erf(t / np.sqrt(2.0)) - 2.0 * t * norm.pdf(t) + 2.0 * t**2 * norm.sf(t)

    # Small-t expansion avoids cancellation in erf(t/sqrt(2)) - 2t phi(t).
    phi0 = 1.0 / np.sqrt(2.0 * np.pi)
    series = t**2 - (4.0 / 3.0) * phi0 * t**3 + (2.0 / 15.0) * phi0 * t**5
    return np.where(t < 1e-5, series, val)


_T_MIXTURE_QUAD_CACHE: dict[tuple[float, bool, int], tuple[np.ndarray, np.ndarray]] = {}


def _t_mixture_quad_cached(df: float, standardize_t: bool, n_quad: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Quadrature for U ~ chi-square(df).

    If X = U/2, then X ~ Gamma(df/2, 1), so
        E[h(U)] = 1/Gamma(a) int_0^inf x^{a-1} e^{-x} h(2x) dx,
    with a = df/2.
    """
    key = (float(df), bool(standardize_t), int(n_quad))
    cached = _T_MIXTURE_QUAD_CACHE.get(key)
    if cached is not None:
        return cached

    df_float = float(df)
    if standardize_t and df_float <= 2.0:
        raise ValueError("Cannot variance-standardize Student-t with df <= 2.")

    a = 0.5 * df_float
    x, w = roots_genlaguerre(int(n_quad), a - 1.0)
    u = 2.0 * x
    weights = w / gamma(a)

    if standardize_t:
        t_scale2 = (df_float - 2.0) / df_float
    else:
        t_scale2 = 1.0

    # Conditional variance of t-noise given U is t_scale2 * df / U.
    noise_var_base = t_scale2 * df_float / u
    out = (noise_var_base.astype(float), weights.astype(float))
    _T_MIXTURE_QUAD_CACHE[key] = out
    return out


def lad_quad_n_from_gh(gh_n: int) -> int:
    return int(max(DEFAULT_LAD_QUAD_MIN, DEFAULT_LAD_QUAD_MULTIPLIER * int(gh_n)))


def lad_AC_fast_t(alpha: float, kappa: float, df: float, standardize_t: bool,
                  score_scale: float, n_quad: int) -> dict[str, float]:
    """
    Fast LAD score side for V = alpha G + Student-t noise.

    psi_kappa(v) = clip(v, [-c,c]), c = score_scale*kappa.
    A = E[min{V^2,c^2}]
    C = P(|V| <= c)

    Conditional on U ~ chi-square(df), V | U ~ N(0, alpha^2 + scale^2*df/U).
    """
    c = float(max(score_scale * kappa, 0.0))
    if c <= 0.0:
        return {'A': 0.0, 'C': 0.0, 'c': c, 'f0': np.nan}

    noise_var_base, weights = _t_mixture_quad_cached(float(df), bool(standardize_t), int(n_quad))
    var = alpha**2 + noise_var_base
    sd = np.sqrt(var)
    r = c / sd

    C_cond = erf(r / np.sqrt(2.0))
    A_cond = var * _m0_standard_normal(r)

    C = float(np.sum(weights * C_cond))
    A = float(np.sum(weights * A_cond))
    return {'A': float(max(A, 0.0)), 'C': float(min(max(C, 0.0), 1.0)), 'c': c, 'f0': np.nan}


def total_gamma_b2(eps_b: float, gamma_b_nz2: float) -> float:
    return float(eps_b * gamma_b_nz2)


def risk_div_ridge(s: float, tau: float, eps_b: float, gamma_b_nz2: float) -> tuple[float, float]:
    gamma_b2 = total_gamma_b2(eps_b, gamma_b_nz2)
    den = 1.0 + tau
    R = (tau / den) ** 2 * gamma_b2 + (s / den) ** 2
    D = 1.0 / den
    return float(max(R, 0.0)), float(D)


def risk_div_lasso_gaussian_component(s: float, tau: float, gamma_b_component: float) -> tuple[float, float]:
    gb2 = gamma_b_component ** 2
    q = gb2 + s ** 2
    sd = np.sqrt(max(q, 1e-300))

    u = tau / sd
    ph = norm.pdf(u)
    sf = norm.sf(u)

    D = 2.0 * sf

    # X = B + sH ~ N(0,q)
    M1 = 2.0 * sd * ph
    M2 = 2.0 * q * (u * ph + sf)

    # E[B|X] = m X, including m=0 for point mass B=0.
    m = gb2 / q

    # S_tau(X)=sign(X)(|X|-tau)_+.
    EXS = M2 - tau * M1
    ES2 = M2 - 2.0 * tau * M1 + tau ** 2 * D

    R = gb2 - 2.0 * m * EXS + ES2
    return float(max(R, 0.0)), float(D)


def risk_div_lasso_spikeslab(s: float, tau: float, eps_b: float, gamma_b_nz2: float) -> tuple[float, float]:
    R0, D0 = risk_div_lasso_gaussian_component(s, tau, 0.0)
    R1, D1 = risk_div_lasso_gaussian_component(s, tau, np.sqrt(gamma_b_nz2))
    R = (1.0 - eps_b) * R0 + eps_b * R1
    D = (1.0 - eps_b) * D0 + eps_b * D1
    return float(max(R, 0.0)), float(D)


def risk_div(penalty: str, s: float, tau: float, eps_b: float, gamma_b_nz2: float) -> tuple[float, float]:
    if penalty == 'ridge':
        return risk_div_ridge(s, tau, eps_b, gamma_b_nz2)
    if penalty == 'lasso':
        return risk_div_lasso_spikeslab(s, tau, eps_b, gamma_b_nz2)
    raise ValueError("penalty must be 'ridge' or 'lasso'")


def rank_density_stats(alpha: float, df: int, standardize_t: bool, grid_size: int) -> dict[str, float]:
    xmax = robust_xmax(alpha, df, kappa=0.0, standardize=standardize_t)
    x = np.linspace(0.0, xmax, grid_size)
    f, _ = fV_FV_on_grid(alpha, x, df, standardize_t)
    Ialpha = 2.0 * np.trapezoid(f ** 2, x)
    return {'Ialpha': float(Ialpha), 'f0': float(f[0]), 'xmax': float(xmax), 'tail_f': float(f[-1])}


def rank_AC(alpha: float, kappa: float, df: int, standardize_t: bool, grid_size: int) -> dict[str, float]:
    xmax = robust_xmax(alpha, df, kappa=kappa, standardize=standardize_t)
    # Make sure the crossing bound c <= 2 kappa is covered comfortably.
    xmax = max(xmax, 2.2 * kappa + 10.0)
    x = np.linspace(0.0, xmax, grid_size)
    f, F = fV_FV_on_grid(alpha, x, df, standardize_t)

    Ialpha = 2.0 * np.trapezoid(f ** 2, x)
    f0 = float(f[0])

    if 4.0 * kappa * f0 <= 1.0 + 1e-10:
        A = 4.0 * kappa ** 2 / 3.0
        C = 4.0 * kappa * Ialpha
        return {'A': float(A), 'C': float(C), 'c': 0.0, 'Ialpha': float(Ialpha), 'f0': f0}

    # The positive root can lie entirely before the first integration-grid point.
    # Divide the crossing equation by c and use its positive limit at zero;
    # never reinterpret a missed crossing as an inactive isotonic constraint.
    local_nodes, local_weights = np.polynomial.legendre.leggauss(16)

    def crossing_ratio(value: float) -> float:
        if value == 0:
            return 4.0 * kappa * f0 - 1.0
        _, distribution = fV_FV_on_grid(alpha, np.array([value]), df, standardize_t)
        difference = float(distribution[0] - 0.5)
        if abs(difference) < 1e-6:
            # Integral_0^c f(v)dv / c avoids CDF cancellation near zero.
            nodes = value * (local_nodes + 1.0) / 2.0
            densities, _ = fV_FV_on_grid(alpha, nodes, df, standardize_t)
            average_density = float(local_weights @ densities) / 2.0
        else:
            average_density = difference / value
        return 4.0 * kappa * average_density - 1.0

    c = float(brentq(crossing_ratio, 0.0, 2.0 * kappa, xtol=1e-11, rtol=1e-10))
    if c <= 0 or not np.isfinite(c):
        raise FloatingPointError("Failed to resolve the positive rank-score cutoff")
    f_c = float(fV_FV_on_grid(alpha, np.array([c]), df, standardize_t)[0][0])
    ic = int(np.searchsorted(x, c, side='right'))

    x_left = np.concatenate([x[:ic], [c]])
    f_left = np.concatenate([f[:ic], [f_c]])
    I2 = np.trapezoid((x_left ** 2) * f_left, x_left)

    x_right = np.concatenate([[c], x[ic:]])
    f_right = np.concatenate([[f_c], f[ic:]])
    J2 = np.trapezoid(f_right ** 2, x_right)

    A = 2.0 * I2 + 4.0 * kappa ** 2 / 3.0 - c ** 3 / (6.0 * kappa)
    C = c / (2.0 * kappa) + 8.0 * kappa * J2
    return {'A': float(max(A, 0.0)), 'C': float(max(C, 1e-300)), 'c': c, 'Ialpha': float(Ialpha), 'f0': f0}


def solve_rank_c0_t(delta: float, lam: float, penalty: str, df: int, standardize_t: bool,
                    eps_b: float, gamma_b_nz2: float, grid_size: int) -> Optional[dict[str, Any]]:
    def Froot(alpha: float) -> float:
        stats = rank_density_stats(alpha, df, standardize_t, grid_size)
        Ialpha = stats['Ialpha']
        s = 1.0 / (2.0 * np.sqrt(3.0 * delta) * Ialpha)
        tau = lam / (4.0 * delta * Ialpha)
        R, _ = risk_div(penalty, s, tau, eps_b, gamma_b_nz2)
        return alpha ** 2 - R

    lo = 1e-10
    flo = Froot(lo)
    hi = max(1.0, np.sqrt(total_gamma_b2(eps_b, gamma_b_nz2)))
    fhi = Froot(hi)

    for _ in range(70):
        if flo * fhi <= 0:
            break
        hi *= 2.0
        fhi = Froot(hi)
        if hi > 1e8:
            return None

    if flo * fhi > 0:
        return None

    try:
        alpha = brentq(Froot, lo, hi, xtol=1e-9, rtol=1e-9, maxiter=200)
    except Exception:
        return None

    stats = rank_density_stats(alpha, df, standardize_t, grid_size)
    Ialpha = stats['Ialpha']
    f0 = stats['f0']
    s = 1.0 / (2.0 * np.sqrt(3.0 * delta) * Ialpha)
    tau = lam / (4.0 * delta * Ialpha)
    _, D = risk_div(penalty, s, tau, eps_b, gamma_b_nz2)
    kappa = D / (4.0 * delta * Ialpha)
    valid = (D * f0 <= delta * Ialpha * (1.0 + 1e-7))

    return {
        'alpha': float(alpha), 'alpha2': float(alpha ** 2), 'kappa': float(kappa),
        'c': 0.0, 'regime': 'c=0', 'valid': bool(valid),
        'cost': float((Froot(alpha) / max(alpha * alpha, 1e-12)) ** 2)
    }


def _least_squares_from_guess(residual, ag: float, kg: float, max_nfev: int,
                              lb: np.ndarray, ub: np.ndarray):
    return least_squares(
        residual,
        np.log([max(ag, 1e-10), max(kg, 1e-10)]),
        bounds=(lb, ub),
        xtol=1e-8,
        ftol=1e-8,
        gtol=1e-8,
        max_nfev=max_nfev,
    )


def solve_rank_cpos_t(delta: float, lam: float, penalty: str, df: int, standardize_t: bool,
                      eps_b: float, gamma_b_nz2: float, grid_size: int,
                      prev: Optional[dict[str, Any]] = None) -> Optional[dict[str, Any]]:
    eps = 1e-300

    def residual(z: np.ndarray) -> np.ndarray:
        alpha = float(np.exp(z[0]))
        kappa = float(np.exp(z[1]))
        stats = rank_AC(alpha, kappa, df, standardize_t, grid_size)
        A, C = stats['A'], stats['C']
        den = delta * C
        s = np.sqrt(delta * A) / den
        tau = lam * kappa / den
        R, D = risk_div(penalty, s, tau, eps_b, gamma_b_nz2)
        return np.array([
            np.log(alpha ** 2 + eps) - np.log(R + eps),
            np.log(den + eps) - np.log(D + eps),
        ])

    lb = np.log([1e-10, 1e-10])
    ub = np.log([1e7, 1e7])

    # Continuation first.
    if prev is not None and np.isfinite(prev.get('alpha', np.nan)):
        try:
            res = _least_squares_from_guess(
                residual,
                max(float(prev['alpha']), 1e-8),
                max(float(prev.get('kappa', 0.1)), 1e-8),
                90,
                lb,
                ub,
            )
            cost = float(np.sum(res.fun ** 2))
            if cost < 1e-8:
                alpha = float(np.exp(res.x[0]))
                kappa = float(np.exp(res.x[1]))
                stats = rank_AC(alpha, kappa, df, standardize_t, grid_size)
                return {
                    'alpha': alpha, 'alpha2': alpha ** 2, 'kappa': kappa, 'c': stats['c'],
                    'regime': 'c>0', 'cost': cost
                }
        except Exception:
            pass

    guesses: list[tuple[float, float]] = []
    if prev is not None and np.isfinite(prev.get('alpha', np.nan)):
        guesses.append((max(float(prev['alpha']), 1e-8), max(float(prev.get('kappa', 0.1)), 1e-8)))
    for ag in [0.05, 0.2, 0.8, 3.0, 10.0]:
        for kg in [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]:
            guesses.append((ag, kg))

    best = None
    for ag, kg in guesses:
        try:
            res = _least_squares_from_guess(residual, ag, kg, 220, lb, ub)
            cost = float(np.sum(res.fun ** 2))
            if best is None or cost < best[0]:
                best = (cost, res)
        except Exception:
            continue

    if best is None:
        return None
    cost, res = best
    alpha = float(np.exp(res.x[0]))
    kappa = float(np.exp(res.x[1]))
    stats = rank_AC(alpha, kappa, df, standardize_t, grid_size)
    return {
        'alpha': alpha, 'alpha2': alpha ** 2, 'kappa': kappa, 'c': stats['c'],
        'regime': 'c>0', 'cost': cost
    }


def solve_rank_point(delta: float, lam: float, penalty: str, df: int, standardize_t: bool,
                     eps_b: float, gamma_b_nz2: float, grid_size: int,
                     prev: Optional[dict[str, Any]] = None) -> Optional[dict[str, Any]]:
    sol0 = solve_rank_c0_t(delta, lam, penalty, df, standardize_t, eps_b, gamma_b_nz2, grid_size)
    if sol0 is not None and sol0['valid']:
        return sol0

    init = sol0 if sol0 is not None else prev
    sol = solve_rank_cpos_t(delta, lam, penalty, df, standardize_t, eps_b, gamma_b_nz2, grid_size, prev=init)
    if sol is not None and sol.get('cost', 0.0) > 1e-4:
        sol_alt = solve_rank_cpos_t(delta, lam, penalty, df, standardize_t, eps_b, gamma_b_nz2, grid_size, prev=prev)
        if sol_alt is not None and sol_alt.get('cost', np.inf) < sol.get('cost', np.inf):
            sol = sol_alt
    return sol


def solve_lad_ridge_fast_lambda(delta: float, lam: float, df: int, standardize_t: bool,
                                eps_b: float, gamma_b_nz2: float, score_scale: float,
                                n_quad: int,
                                prev: Optional[dict[str, Any]] = None) -> Optional[dict[str, Any]]:
    """
    Fixed-lambda LAD + ridge solver using the simplified two equations
        delta*C(alpha,kappa) + lambda*kappa = 1,
        alpha^2 = delta*A(alpha,kappa) + (lambda*kappa)^2 E[B^2].

    This is solved as a two-variable least-squares problem, not by nested
    alpha/kappa scalar roots.  In practice this is much faster because each
    residual evaluation needs only one LAD A,C evaluation.
    """
    gamma_b2 = total_gamma_b2(eps_b, gamma_b_nz2)
    eps = 1e-300

    def residual(z: np.ndarray) -> np.ndarray:
        alpha = float(np.exp(z[0]))
        kappa = float(np.exp(z[1]))
        stats = lad_AC_fast_t(alpha, kappa, df, standardize_t, score_scale, n_quad)
        A = stats['A']
        C = stats['C']
        rhs = delta * A + (lam * kappa) ** 2 * gamma_b2
        return np.array([
            np.log(alpha ** 2 + eps) - np.log(rhs + eps),
            delta * C + lam * kappa - 1.0,
        ])

    lb = np.log([1e-10, 1e-10])
    ub = np.log([1e8, 1e8])

    # Continuation-first.
    if prev is not None and np.isfinite(prev.get('alpha', np.nan)):
        try:
            res = _least_squares_from_guess(
                residual,
                max(float(prev['alpha']), 1e-8),
                max(float(prev.get('kappa', 0.5)), 1e-8),
                80,
                lb,
                ub,
            )
            cost = float(np.sum(res.fun ** 2))
            if cost < 1e-8:
                alpha = float(np.exp(res.x[0]))
                kappa = float(np.exp(res.x[1]))
                stats = lad_AC_fast_t(alpha, kappa, df, standardize_t, score_scale, n_quad)
                return {
                    'alpha': alpha, 'alpha2': alpha ** 2, 'kappa': kappa,
                    'c': stats['c'], 'regime': 'lad', 'cost': cost, 'lambda': float(lam)
                }
        except Exception:
            pass

    guesses: list[tuple[float, float]] = []
    if prev is not None and np.isfinite(prev.get('alpha', np.nan)):
        guesses.append((max(float(prev['alpha']), 1e-8), max(float(prev.get('kappa', 0.5)), 1e-8)))

    # A small deterministic multi-start set is enough for the simplified ridge equations.
    for ag in [0.04, 0.15, 0.5, 1.5, 5.0, 15.0]:
        for kg in [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]:
            guesses.append((ag, kg))

    best = None
    for ag, kg in guesses:
        try:
            res = _least_squares_from_guess(residual, ag, kg, 180, lb, ub)
            cost = float(np.sum(res.fun ** 2))
            if best is None or cost < best[0]:
                best = (cost, res)
        except Exception:
            continue

    if best is None:
        return None

    cost, res = best
    alpha = float(np.exp(res.x[0]))
    kappa = float(np.exp(res.x[1]))
    stats = lad_AC_fast_t(alpha, kappa, df, standardize_t, score_scale, n_quad)
    return {
        'alpha': alpha, 'alpha2': alpha ** 2, 'kappa': kappa,
        'c': stats['c'], 'regime': 'lad', 'cost': cost, 'lambda': float(lam)
    }


def solve_lad_lasso_fast_lambda(delta: float, lam: float, df: int, standardize_t: bool,
                                eps_b: float, gamma_b_nz2: float, score_scale: float,
                                n_quad: int,
                                prev: Optional[dict[str, Any]] = None) -> Optional[dict[str, Any]]:
    eps = 1e-300

    def residual(z: np.ndarray) -> np.ndarray:
        alpha = float(np.exp(z[0]))
        kappa = float(np.exp(z[1]))
        stats = lad_AC_fast_t(alpha, kappa, df, standardize_t, score_scale, n_quad)
        A, C = stats['A'], max(stats['C'], 1e-300)
        den = delta * C
        s = np.sqrt(delta * A) / den
        tau = lam * kappa / den
        R, D = risk_div_lasso_spikeslab(s, tau, eps_b, gamma_b_nz2)
        return np.array([
            np.log(alpha ** 2 + eps) - np.log(R + eps),
            np.log(den + eps) - np.log(D + eps),
        ])

    lb = np.log([1e-10, 1e-10])
    ub = np.log([1e8, 1e8])

    if prev is not None and np.isfinite(prev.get('alpha', np.nan)):
        try:
            res = _least_squares_from_guess(
                residual,
                max(float(prev['alpha']), 1e-8),
                max(float(prev.get('kappa', 0.5)), 1e-8),
                90,
                lb,
                ub,
            )
            cost = float(np.sum(res.fun ** 2))
            if cost < 1e-8:
                alpha = float(np.exp(res.x[0]))
                kappa = float(np.exp(res.x[1]))
                stats = lad_AC_fast_t(alpha, kappa, df, standardize_t, score_scale, n_quad)
                return {
                    'alpha': alpha, 'alpha2': alpha ** 2, 'kappa': kappa,
                    'c': stats['c'], 'regime': 'lad', 'cost': cost, 'lambda': float(lam)
                }
        except Exception:
            pass

    guesses: list[tuple[float, float]] = []
    if prev is not None and np.isfinite(prev.get('alpha', np.nan)):
        guesses.append((max(float(prev['alpha']), 1e-8), max(float(prev.get('kappa', 0.5)), 1e-8)))
    for ag in [0.05, 0.2, 0.8, 3.0, 10.0]:
        for kg in [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]:
            guesses.append((ag, kg))

    best = None
    for ag, kg in guesses:
        try:
            res = _least_squares_from_guess(residual, ag, kg, 220, lb, ub)
            cost = float(np.sum(res.fun ** 2))
            if best is None or cost < best[0]:
                best = (cost, res)
        except Exception:
            continue

    if best is None:
        return None

    cost, res = best
    if cost > 1e-4:
        return None
    alpha = float(np.exp(res.x[0]))
    kappa = float(np.exp(res.x[1]))
    stats = lad_AC_fast_t(alpha, kappa, df, standardize_t, score_scale, n_quad)
    return {
        'alpha': alpha, 'alpha2': alpha ** 2, 'kappa': kappa,
        'c': stats['c'], 'regime': 'lad', 'cost': cost, 'lambda': float(lam)
    }


def solve_lad_point(delta: float, lam: float, penalty: str, df: int, standardize_t: bool,
                    eps_b: float, gamma_b_nz2: float, grid_size: int, score_scale: float,
                    prev: Optional[dict[str, Any]] = None,
                    lad_quad_n: Optional[int] = None) -> Optional[dict[str, Any]]:
    # grid_size is retained in the signature for compatibility with the original script.
    n_quad = int(lad_quad_n if lad_quad_n is not None else lad_quad_n_from_gh(DEFAULT_GH_N))
    if penalty == 'ridge':
        return solve_lad_ridge_fast_lambda(delta, lam, df, standardize_t,
                                           eps_b, gamma_b_nz2, score_scale, n_quad, prev=prev)
    if penalty == 'lasso':
        return solve_lad_lasso_fast_lambda(delta, lam, df, standardize_t,
                                           eps_b, gamma_b_nz2, score_scale, n_quad, prev=prev)
    raise ValueError("penalty must be 'ridge' or 'lasso'")


def solve_ls_lasso_point(delta: float, lam: float, sigma2: float,
                         eps_b: float, gamma_b_nz2: float,
                         prev: Optional[dict[str, Any]] = None) -> Optional[dict[str, Any]]:
    """Fixed-lambda squared-loss Lasso solver used by lambda-space search."""
    if not (np.isfinite(sigma2) and sigma2 >= 0.0 and lam > 0.0):
        return None

    eps = 1e-300

    def residual(z: np.ndarray) -> np.ndarray:
        alpha = float(np.exp(z[0]))
        theta = float(np.exp(z[1]))
        s_eff = np.sqrt((sigma2 + alpha * alpha) / delta)
        R, D = risk_div_lasso_spikeslab(s_eff, theta, eps_b, gamma_b_nz2)
        lam_cal = theta * (delta - D)
        return np.array([
            np.log(alpha * alpha + eps) - np.log(R + eps),
            (lam_cal - lam) / (1.0 + abs(lam)),
        ])

    lb = np.log([1e-10, 1e-10])
    ub = np.log([1e8, 1e8])

    def run_guess(alpha0: float, theta0: float, max_nfev: int):
        return least_squares(
            residual,
            np.log([max(alpha0, 1e-10), max(theta0, 1e-10)]),
            bounds=(lb, ub),
            xtol=1e-8,
            ftol=1e-8,
            gtol=1e-8,
            max_nfev=max_nfev,
        )

    # Continuation first: previous neighboring delta or neighboring lambda.
    if prev is not None:
        alpha0 = prev.get('alpha', None)
        theta0 = prev.get('theta', None)
        if finite_positive(alpha0) and finite_positive(theta0):
            try:
                res = run_guess(float(alpha0), float(theta0), 90)
                cost = float(np.sum(res.fun ** 2))
                if cost < 1e-8:
                    alpha = float(np.exp(res.x[0]))
                    theta = float(np.exp(res.x[1]))
                    return {
                        'alpha': alpha,
                        'alpha2': alpha * alpha,
                        'theta': theta,
                        'lambda': float(lam),
                        'cost': cost,
                        'regime': 'ls-lasso',
                    }
            except Exception:
                pass

    guesses: list[tuple[float, float]] = []
    if prev is not None:
        alpha0 = prev.get('alpha', None)
        theta0 = prev.get('theta', None)
        if finite_positive(alpha0) and finite_positive(theta0):
            guesses.append((float(alpha0), float(theta0)))

    for ag in [0.05, 0.2, 0.8, 3.0, 10.0]:
        for tg in [max(lam, 1e-4), 0.05, 0.2, 0.8, 3.0, 10.0, 30.0]:
            guesses.append((ag, tg))

    best: Optional[tuple[float, Any]] = None
    for ag, tg in guesses:
        try:
            res = run_guess(ag, tg, 180)
            cost = float(np.sum(res.fun ** 2))
            if best is None or cost < best[0]:
                best = (cost, res)
        except Exception:
            continue

    if best is None:
        return None

    cost, res = best
    if cost > 1e-4:
        return None
    alpha = float(np.exp(res.x[0]))
    theta = float(np.exp(res.x[1]))
    return {
        'alpha': alpha,
        'alpha2': alpha * alpha,
        'theta': theta,
        'lambda': float(lam),
        'cost': cost,
        'regime': 'ls-lasso',
    }


def solve_ls_point(delta: float, lam: float, penalty: str, sigma2: float,
                   eps_b: float, gamma_b_nz2: float,
                   prev: Optional[dict[str, Any]] = None) -> Optional[dict[str, Any]]:
    """Fixed-lambda LS solver used by the unified lambda optimizer."""
    if penalty == 'ridge':
        a2 = _alpha2_ls_ridge_at_lambda(delta, lam, sigma2, eps_b, gamma_b_nz2)
        if not (np.isfinite(a2) and a2 > 0.0):
            return None
        return {
            'alpha': float(np.sqrt(a2)),
            'alpha2': float(a2),
            'lambda': float(lam),
            'cost': 0.0,
            'regime': 'ls-ridge',
        }
    if penalty == 'lasso':
        return solve_ls_lasso_point(delta, lam, sigma2, eps_b, gamma_b_nz2, prev=prev)
    raise ValueError("penalty must be 'ridge' or 'lasso'")


def analytic_ls_ridge_lambda_seed(delta: float, sigma2: float, eps_b: float,
                                  gamma_b_nz2: float,
                                  lambda_min: float, lambda_max: float) -> Optional[float]:
    """Closed-form stationary LS+ridge lambda, used as a seed in lambda search."""
    if not np.isfinite(sigma2):
        return None
    gamma_b2 = total_gamma_b2(eps_b, gamma_b_nz2)
    if gamma_b2 <= 0.0 or sigma2 <= 0.0:
        return None
    a = 1.0 - 1.0 / delta
    B = gamma_b2
    S = sigma2
    bb = B * a - S / delta
    cc = -S / delta
    disc = bb * bb - 4.0 * B * cc
    if disc < 0.0:
        return None
    gamma_star = (-bb + np.sqrt(disc)) / (2.0 * B)
    if not (np.isfinite(gamma_star) and gamma_star > 0.0):
        return None
    lam_star = _ls_ridge_lambda_from_gamma(delta, gamma_star)
    if np.isfinite(lam_star) and lambda_min <= lam_star <= lambda_max:
        return float(lam_star)
    return None


def optimize_lambda_direct(estimator: str, penalty: str, delta: float, df: int | float,
                           standardize_t: bool, eps_b: float, gamma_b_nz2: float,
                           lambda_min: float, lambda_max: float, n_grid: int,
                           grid_size: int, score_scale: float, gh_n: int,
                           lambda_hint: Optional[float] = None,
                           state_hint: Optional[dict[str, Any]] = None,
                           sigma2: Optional[float] = None) -> dict[str, Any]:
    """
    Coarse-to-refined optimization over log(lambda).

    Steps:
      1. Build a global log-lambda grid, plus a local cluster around lambda_hint.
      2. Evaluate grid points starting near the hint / lambda=1.
      3. Each fixed-lambda solve uses the nearest solved point as continuation.
      4. Refine the best neighboring log-lambda interval with minimize_scalar.
         If the coarse best is at a boundary, refine the adjacent edge interval
         too, so the result is not determined by a single endpoint evaluation.
    """
    logs, lambdas, target_log = make_focused_log_lambda_grid(
        lambda_min, lambda_max, max(9, int(n_grid)), lambda_hint=lambda_hint
    )
    alpha2 = np.full_like(lambdas, np.nan, dtype=float)
    sols: list[Optional[dict[str, Any]]] = [None] * len(lambdas)
    lad_quad_n = lad_quad_n_from_gh(gh_n)

    evaluation_trace = []
    solved_pairs: list[tuple[float, dict[str, Any]]] = []
    if state_hint is not None:
        lam_state = solution_lambda(state_hint)
        if lam_state is None and finite_positive(lambda_hint):
            lam_state = float(lambda_hint)
        if lam_state is not None and lambda_min <= lam_state <= lambda_max:
            solved_pairs.append((float(np.log(lam_state)), state_hint))

    order = np.argsort(np.abs(logs - target_log))

    def coarse_prev(loglam: float) -> Optional[dict[str, Any]]:
        if not solved_pairs:
            return None
        j = int(np.argmin([abs(ll - loglam) for ll, _ in solved_pairs]))
        return solved_pairs[j][1]

    def solve_raw(lam: float, prev: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
        if estimator == 'rank':
            return solve_rank_point(delta, lam, penalty, df, standardize_t,
                                    eps_b, gamma_b_nz2, grid_size, prev=prev)
        if estimator == 'lad':
            return solve_lad_point(delta, lam, penalty, df, standardize_t,
                                   eps_b, gamma_b_nz2, grid_size, score_scale,
                                   prev=prev, lad_quad_n=lad_quad_n)
        if estimator == 'ls':
            if sigma2 is None:
                raise ValueError('sigma2 is required for LS lambda search')
            return solve_ls_point(delta, lam, penalty, sigma2,
                                  eps_b, gamma_b_nz2, prev=prev)
        raise ValueError("estimator must be 'rank', 'lad', or 'ls'")

    def solve_at_lambda(lam: float, prev: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
        sol = solve_raw(lam, prev)
        evaluation_trace.append({"lambda": float(lam), "alpha2": float(sol.get("alpha2", np.nan)) if sol else np.nan, "residual_squared": float(sol.get("cost", np.nan)) if sol else np.nan, "valid": good_solution(sol)})
        return sol

    for i in order:
        lam = float(lambdas[i])
        prev = coarse_prev(float(logs[i]))
        sol = solve_at_lambda(lam, prev)
        sols[i] = sol
        if good_solution(sol):
            alpha2[i] = sol['alpha2']
            solved_pairs.append((float(logs[i]), sol))

    mask = np.isfinite(alpha2) & (alpha2 > 0.0)
    if not np.any(mask):
        return {
            'df': output_df_value(df), 'penalty': penalty, 'estimator': estimator, 'delta': float(delta),
            'alpha2_opt': np.nan, 'lambda_opt': np.nan, 'opt_on_boundary': False,
            'state': None, 'trace': evaluation_trace,
        }

    valid_idx = np.where(mask)[0]
    best_idx = int(valid_idx[np.argmin(alpha2[mask])])
    best_alpha2 = float(alpha2[best_idx])
    best_lambda = float(lambdas[best_idx])
    best_state = sols[best_idx]
    best_boundary = bool(best_idx == valid_idx[0] or best_idx == valid_idx[-1])

    # Refine the best interval.  Edge intervals are also refined.
    if len(lambdas) >= 2:
        if best_idx <= 0:
            lo_log, hi_log = float(logs[0]), float(logs[1])
        elif best_idx >= len(lambdas) - 1:
            lo_log, hi_log = float(logs[-2]), float(logs[-1])
        else:
            lo_log, hi_log = float(logs[best_idx - 1]), float(logs[best_idx + 1])

        eval_cache: dict[float, tuple[float, Optional[dict[str, Any]]]] = {}

        def nearest_prev(loglam: float) -> Optional[dict[str, Any]]:
            if eval_cache:
                key = min(eval_cache.keys(), key=lambda k: abs(k - loglam))
                sol = eval_cache[key][1]
                if sol is not None:
                    return sol
            order_local = np.argsort(np.abs(logs - loglam))
            for j in order_local:
                sol = sols[int(j)]
                if sol is not None:
                    return sol
            return coarse_prev(loglam)

        def objective(loglam: float) -> float:
            key = round(float(loglam), 12)
            if key in eval_cache:
                return eval_cache[key][0]
            lam = float(np.exp(loglam))
            prev_hint = nearest_prev(float(loglam))
            sol = solve_at_lambda(lam, prev_hint)
            if not good_solution(sol):
                val = 1e100
            else:
                val = float(sol['alpha2'])
            eval_cache[key] = (val, sol)
            return val

        try:
            res = minimize_scalar(objective, bounds=(lo_log, hi_log), method='bounded',
                                  options={'xatol': 1e-3, 'maxiter': 40})
            val = objective(float(res.x))
            sol_refined = eval_cache[round(float(res.x), 12)][1]
            if sol_refined is not None and np.isfinite(val) and val <= best_alpha2 * (1.0 + 1e-7):
                best_alpha2 = float(val)
                best_lambda = float(np.exp(res.x))
                best_state = sol_refined
                best_boundary = best_boundary or abs(float(res.x) - logs[0]) < 2e-3 or abs(float(res.x) - logs[-1]) < 2e-3
        except Exception:
            pass

    best_boundary = bool(
        best_boundary or
        best_lambda <= lambda_min * (1.0 + 1e-4) or
        best_lambda >= lambda_max / (1.0 + 1e-4)
    )

    return {
        'df': output_df_value(df),
        'penalty': penalty,
        'estimator': estimator,
        'delta': float(delta),
        'alpha2_opt': best_alpha2,
        'lambda_opt': best_lambda,
        'opt_on_boundary': best_boundary,
        'state': best_state,
        'trace': evaluation_trace,
    }


def t_variance_for_ls(df: int, standardize_t: bool) -> float:
    """Variance of Z. Returns inf for raw t_df with df<=2."""
    if standardize_t:
        if df <= 2:
            raise ValueError("--standardize-t is incompatible with df<=2.")
        return 1.0
    if df <= 2:
        return np.inf
    return float(df / (df - 2.0))


def _ls_ridge_gamma_from_lambda(delta: float, lam: float) -> float:
    """Inverse of lambda = gamma * {delta - 1/(1+gamma)}."""
    b = delta - 1.0 - lam
    disc = b * b + 4.0 * lam * delta
    return float((-b + np.sqrt(max(disc, 0.0))) / (2.0 * delta))


def _ls_ridge_lambda_from_gamma(delta: float, gamma_val: float) -> float:
    return float(gamma_val * (delta - 1.0 / (1.0 + gamma_val)))


def _alpha2_ls_ridge_at_lambda(delta: float, lam: float, sigma2: float,
                                eps_b: float, gamma_b_nz2: float) -> float:
    if not np.isfinite(sigma2) or delta <= 0.0 or lam <= 0.0:
        return np.nan
    gamma_val = _ls_ridge_gamma_from_lambda(delta, lam)
    gamma_b2 = total_gamma_b2(eps_b, gamma_b_nz2)
    denom = (1.0 + gamma_val) ** 2 - 1.0 / delta
    if denom <= 0.0:
        return np.nan
    return float((gamma_val * gamma_val * gamma_b2 + sigma2 / delta) / denom)


def make_phase_row(df: int | float, delta: float, penalty: str,
                   ls_row: dict[str, Any], lad_row: dict[str, Any], rank_row: dict[str, Any]) -> dict[str, Any]:
    ls_opt = float(ls_row.get('alpha2_opt', np.nan))
    lad_opt = float(lad_row.get('alpha2_opt', np.nan))
    rank_opt = float(rank_row.get('alpha2_opt', np.nan))
    ls_lam = float(ls_row.get('lambda_opt', np.nan))
    lad_lam = float(lad_row.get('lambda_opt', np.nan))
    rank_lam = float(rank_row.get('lambda_opt', np.nan))

    vals = np.array([ls_opt, lad_opt, rank_opt], dtype=float)
    labels = np.array(['ls', 'lad', 'rank'], dtype=object)
    lams = np.array([ls_lam, lad_lam, rank_lam], dtype=float)
    finite = np.isfinite(vals) & (vals > 0.0)
    winner = 'none'
    winner_code = -1
    winner_alpha2 = np.nan
    winner_lambda = np.nan
    if np.any(finite):
        j = np.where(finite)[0][np.argmin(vals[finite])]
        winner = str(labels[j])
        winner_code = int(j)  # ls=0, lad=1, rank=2
        winner_alpha2 = float(vals[j])
        winner_lambda = float(lams[j])

    return {
        'df': output_df_value(df),
        'delta': float(delta),
        'penalty': penalty,
        'alpha2_ls_opt': ls_opt,
        'lambda_ls_opt': ls_lam,
        'ls_opt_on_boundary': bool(ls_row.get('opt_on_boundary', False)),
        'alpha2_lad_opt': lad_opt,
        'lambda_lad_opt': lad_lam,
        'lad_opt_on_boundary': bool(lad_row.get('opt_on_boundary', False)),
        'alpha2_rank_opt': rank_opt,
        'lambda_rank_opt': rank_lam,
        'rank_opt_on_boundary': bool(rank_row.get('opt_on_boundary', False)),
        'winner': winner,
        'winner_code': winner_code,
        'winner_alpha2': winner_alpha2,
        'winner_lambda': winner_lambda,
        'ls_available': bool(np.isfinite(ls_opt) and ls_opt > 0.0),
    }
