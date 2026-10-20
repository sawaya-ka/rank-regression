#!/usr/bin/env python3
"""Gaussian self-consistency solutions and effective scores for LS, LAD, and rank.

The noise is Z~N(0,1), delta=n/p>1, and V=alpha*G+Z. All three scores
satisfy alpha**2=delta*E[psi(V)**2] and alpha=delta*E[G*psi(V)].
The Gaussian reduction uses closed forms plus a scalar root for the active
rank branch. ``verify_self_consistency`` checks both equations independently
by one-dimensional quadrature. No Monte Carlo is used.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import erf, erfc
from scipy.stats import norm


SQRT2 = math.sqrt(2.0)
SQRT_PI = math.sqrt(math.pi)
RANK_CRITICAL_DELTA = SQRT2


@dataclass(frozen=True)
class ScoreSolution:
    delta: float
    estimator: str
    alpha: float
    alpha2: float
    kappa: float
    sigma_v: float
    c: float
    standardized_threshold: float
    kappa_over_sigma_v: float
    branch: str
    residual_alpha2: float = math.nan
    residual_alpha: float = math.nan


def _phi(x: float | np.ndarray) -> float | np.ndarray:
    return np.exp(-0.5 * np.asarray(x) ** 2) / math.sqrt(2.0 * math.pi)


def _Phi_minus_half(x: float | np.ndarray) -> float | np.ndarray:
    """Stable evaluation of Phi(x)-1/2."""
    return 0.5 * erf(np.asarray(x) / SQRT2)


def _m0_standard_normal(t: float) -> float:
    """E[min(X^2,t^2)] for X~N(0,1)."""
    return float(
        2.0 * (norm.cdf(t) - 0.5)
        - 2.0 * t * norm.pdf(t)
        + 2.0 * t * t * norm.sf(t)
    )


def solve_ls_gaussian(delta: float) -> ScoreSolution:
    """Exact LS solution under Z~N(0,1)."""
    if not math.isfinite(delta) or delta <= 1.0:
        raise ValueError("delta must be greater than 1.")

    alpha2 = 1.0 / (delta - 1.0)
    alpha = math.sqrt(alpha2)
    kappa = 1.0 / (delta - 1.0)
    sigma_v = math.sqrt(1.0 + alpha2)

    return ScoreSolution(
        delta=delta,
        estimator="LS",
        alpha=alpha,
        alpha2=alpha2,
        kappa=kappa,
        sigma_v=sigma_v,
        c=math.nan,
        standardized_threshold=math.nan,
        kappa_over_sigma_v=kappa / sigma_v,
        branch="linear",
    )


def solve_lad_gaussian(delta: float) -> ScoreSolution:
    """Exact one-dimensional reduction of the LAD self-consistency system."""
    if not math.isfinite(delta) or delta <= 1.0:
        raise ValueError("delta must be greater than 1.")

    # From E[rho'(V)]=1/delta:
    # P(|V|<=kappa)=1/delta.  With V=sigma_v X, t=kappa/sigma_v.
    t = float(norm.ppf(0.5 * (1.0 + 1.0 / delta)))
    m0 = _m0_standard_normal(t)
    denominator = 1.0 - delta * m0
    if denominator <= 0.0:
        raise RuntimeError("Invalid LAD denominator; check the derivation/grid.")

    alpha2 = delta * m0 / denominator
    alpha = math.sqrt(alpha2)
    sigma_v = math.sqrt(1.0 + alpha2)
    kappa = t * sigma_v

    return ScoreSolution(
        delta=delta,
        estimator="LAD",
        alpha=alpha,
        alpha2=alpha2,
        kappa=kappa,
        sigma_v=sigma_v,
        c=math.nan,
        standardized_threshold=t,
        kappa_over_sigma_v=kappa / sigma_v,
        branch="Huber/clipped",
    )


def _rank_derivative_moment_from_t(t: float) -> float:
    """
    E[rho'(V)] on the positive-c Gaussian rank branch, as a function of
    t=c/sigma_v only.
    """
    if t <= 1e-10:
        return 1.0 / SQRT2

    A = float(_Phi_minus_half(t))
    return 2.0 * A + t * erfc(t) / (2.0 * A * SQRT_PI)


def solve_rank_gaussian(delta: float) -> ScoreSolution:
    """
    Solve the Gaussian rank self-consistency system.

    For delta >= sqrt(2), c_*=0 and the solution is explicit.
    For 1 < delta < sqrt(2), solve one scalar equation for
    t=c_*/sigma_v, then recover alpha_*, kappa_*, and c_*.
    """
    if not math.isfinite(delta) or delta <= 1.0:
        raise ValueError("delta must be greater than 1.")

    if delta >= RANK_CRITICAL_DELTA:
        # Inactive isotonic constraint.
        r = SQRT_PI / (2.0 * delta)  # r = kappa/sigma_v
        score_second_moment = 4.0 * r * r / 3.0
        denominator = 1.0 - delta * score_second_moment
        if denominator <= 0.0:
            raise RuntimeError("Invalid rank c=0 denominator.")

        alpha2 = delta * score_second_moment / denominator
        alpha = math.sqrt(alpha2)
        sigma_v = math.sqrt(1.0 + alpha2)
        kappa = r * sigma_v

        return ScoreSolution(
            delta=delta,
            estimator="rank",
            alpha=alpha,
            alpha2=alpha2,
            kappa=kappa,
            sigma_v=sigma_v,
            c=0.0,
            standardized_threshold=0.0,
            kappa_over_sigma_v=r,
            branch="inactive",
        )

    target = 1.0 / delta
    t = brentq(
        lambda u: _rank_derivative_moment_from_t(u) - target,
        0.0,
        12.0,
        xtol=1e-13,
        rtol=1e-13,
        maxiter=200,
    )

    A = float(_Phi_minus_half(t))
    r = t / (4.0 * A)  # r = kappa/sigma_v from F_V(c)-1/2=c/(4kappa)

    # E[h(X)^2], where rho(sigma_v X)=sigma_v h(X).
    score_second_moment = (
        2.0 * A
        - 2.0 * t * float(_phi(t))
        + 4.0 * r * r / 3.0
        - t**3 / (6.0 * r)
    )
    denominator = 1.0 - delta * score_second_moment
    if denominator <= 0.0:
        raise RuntimeError("Invalid rank c>0 denominator.")

    alpha2 = delta * score_second_moment / denominator
    alpha = math.sqrt(alpha2)
    sigma_v = math.sqrt(1.0 + alpha2)
    kappa = r * sigma_v
    c = t * sigma_v

    return ScoreSolution(
        delta=delta,
        estimator="rank",
        alpha=alpha,
        alpha2=alpha2,
        kappa=kappa,
        sigma_v=sigma_v,
        c=c,
        standardized_threshold=t,
        kappa_over_sigma_v=r,
        branch="active",
    )


def rho_ls(x: np.ndarray | float, solution: ScoreSolution) -> np.ndarray:
    x_arr = np.asarray(x, dtype=float)
    return solution.kappa * x_arr / (solution.kappa + 1.0)


def rho_lad(x: np.ndarray | float, solution: ScoreSolution) -> np.ndarray:
    x_arr = np.asarray(x, dtype=float)
    return np.clip(x_arr, -solution.kappa, solution.kappa)


def rho_rank(x: np.ndarray | float, solution: ScoreSolution) -> np.ndarray:
    x_arr = np.asarray(x, dtype=float)
    tail = 4.0 * solution.kappa * _Phi_minus_half(x_arr / solution.sigma_v)
    if solution.c <= 0.0:
        return np.asarray(tail, dtype=float)
    return np.where(np.abs(x_arr) < solution.c, x_arr, tail)


def score_function(solution: ScoreSolution) -> Callable[[np.ndarray | float], np.ndarray]:
    if solution.estimator == "LS":
        return lambda x: rho_ls(x, solution)
    if solution.estimator == "LAD":
        return lambda x: rho_lad(x, solution)
    if solution.estimator == "rank":
        return lambda x: rho_rank(x, solution)
    raise ValueError(f"Unknown estimator: {solution.estimator}")


def verify_self_consistency(solution: ScoreSolution) -> ScoreSolution:
    """Independent one-dimensional quadrature check of both state equations."""
    rho = score_function(solution)
    s = solution.sigma_v

    if solution.estimator == "LAD":
        split = solution.kappa / s
    elif solution.estimator == "rank" and solution.c > 0.0:
        split = solution.c / s
    else:
        split = 0.0

    def integrate_even(func: Callable[[float], float]) -> float:
        if split > 0.0:
            left = quad(func, 0.0, split, epsabs=2e-11, epsrel=2e-11, limit=300)[0]
            right = quad(func, split, np.inf, epsabs=2e-11, epsrel=2e-11, limit=300)[0]
            return 2.0 * (left + right)
        return 2.0 * quad(func, 0.0, np.inf, epsabs=2e-11, epsrel=2e-11, limit=300)[0]

    def normal_density(x: float) -> float:
        return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

    e_rho2 = integrate_even(
        lambda x: float(rho(s * x)) ** 2 * normal_density(x)
    )
    e_xrho = integrate_even(
        lambda x: x * float(rho(s * x)) * normal_density(x)
    )
    e_grho = solution.alpha / s * e_xrho

    return ScoreSolution(
        **{
            **asdict(solution),
            "residual_alpha2": solution.alpha2 - solution.delta * e_rho2,
            "residual_alpha": solution.alpha - solution.delta * e_grho,
        }
    )



def rank_atom_mass(delta: float) -> float:
    """Mass at zero of the population rank proximal output, at its risk solution."""
    solution = solve_rank_gaussian(delta)
    return float(erf(solution.standardized_threshold / SQRT2))
