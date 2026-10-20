"""Penalized finite-sample experiment kernels.
All exported lambda values multiply the penalty sum by 1/n.
"""

from __future__ import annotations
import math
import traceback
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np
from scipy.optimize import OptimizeWarning, minimize
from scipy.stats import rankdata
from sklearn.linear_model import Lasso, QuantileRegressor
from sklearn.exceptions import ConvergenceWarning
import warnings


ESTIMATORS = ("LS", "LAD", "rank")


WINNER_CODE = {"LS": 0, "LAD": 1, "rank": 2}


def make_seed(base_seed: int, i_penalty: int, i_df: int, i_delta: int, rep: int) -> int:
    ss = np.random.SeedSequence([base_seed, i_penalty, i_df, i_delta, rep])
    return int(ss.generate_state(1, dtype=np.uint32)[0])


def compute_p_from_delta(n: int, delta: float) -> Tuple[int, float]:
    p = int(round(n / delta))
    p = max(1, p)
    return p, float(n / p)


def generate_beta_true(
    p: int,
    rng: np.random.Generator,
    *,
    zero_prob: float = 0.8,
    slab_var: float = 5.0,
    ensure_nonzero: bool = False,
) -> np.ndarray:
    if p <= 0:
        raise ValueError("p must be positive.")
    if not (0.0 <= zero_prob < 1.0):
        raise ValueError("zero_prob must be in [0, 1).")
    if slab_var <= 0.0:
        raise ValueError("slab_var must be positive.")

    beta = np.zeros(p, dtype=float)
    nonzero = rng.random(p) >= zero_prob
    if ensure_nonzero and not np.any(nonzero):
        nonzero[int(rng.integers(p))] = True
    beta[nonzero] = rng.normal(0.0, math.sqrt(slab_var), size=int(nonzero.sum()))
    return beta


def sample_problem(
    *,
    n: int,
    p: int,
    df: float,
    seed: int,
    beta_zero_prob: float,
    beta_slab_var: float,
    ensure_nonzero_beta: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    if n <= 1:
        raise ValueError("n must be larger than 1.")
    if p <= 0:
        raise ValueError("p must be positive.")
    if df <= 0:
        raise ValueError("df must be positive.")

    rng = np.random.default_rng(seed)
    X = rng.standard_normal(size=(n, p)) / math.sqrt(p)
    beta_true = generate_beta_true(
        p,
        rng,
        zero_prob=beta_zero_prob,
        slab_var=beta_slab_var,
        ensure_nonzero=ensure_nonzero_beta,
    )
    eps = rng.standard_t(df=df, size=n)
    signal = X @ beta_true
    y = signal + eps

    diag = {
        "beta_norm2": float(beta_true @ beta_true),
        "beta_norm2_over_p": float((beta_true @ beta_true) / p),
        "beta_nonzero_frac": float(np.count_nonzero(beta_true) / p),
        "signal_mean": float(np.mean(signal)),
        "signal_var": float(np.var(signal)),
        "noise_mean": float(np.mean(eps)),
        "noise_var": float(np.var(eps)),
        "y_var": float(np.var(y)),
        "x_entry_var_emp": float(np.var(X)),
    }
    return X, y, beta_true, diag


def sqerr(beta_hat: np.ndarray, beta_true: np.ndarray) -> float:
    d = np.asarray(beta_hat, dtype=float) - np.asarray(beta_true, dtype=float)
    return float(d @ d)


def mse_per_coordinate(beta_hat: np.ndarray, beta_true: np.ndarray) -> float:
    return float(sqerr(beta_hat, beta_true) / beta_true.size)


def lambda_eff(lam: float, n: int) -> float:
    if not np.isfinite(lam):
        return np.nan
    return max(float(lam), 0.0) / float(n)


def fit_ls_ridge(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """Ridge for 0.5/n||y-Xb||^2 + 0.5(lambda/n)||b||^2."""
    n, p = X.shape
    leff = lambda_eff(lam, n)
    if not np.isfinite(leff):
        raise ValueError("lambda for LS ridge is not finite.")
    if leff == 0.0:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        return np.asarray(beta, dtype=float)

    ridge = n * leff
    if p <= n:
        A = X.T @ X + ridge * np.eye(p)
        b = X.T @ y
        return np.linalg.solve(A, b)
    A = X @ X.T + ridge * np.eye(n)
    u = np.linalg.solve(A, y)
    return X.T @ u


def fit_ls_lasso_path(
    X: np.ndarray,
    y: np.ndarray,
    lambdas: Sequence[float],
    *,
    max_iter: int,
    tol: float,
) -> Dict[float, np.ndarray]:
    """Evaluate sklearn Lasso along a lambda path with warm_start=True."""
    n, p = X.shape
    # Large lambda first is usually easier for lasso.
    path = sorted([float(l) for l in lambdas], reverse=True)
    out: Dict[float, np.ndarray] = {}
    model = Lasso(
        alpha=1.0,
        fit_intercept=False,
        max_iter=int(max_iter),
        tol=float(tol),
        selection="cyclic",
        warm_start=True,
    )
    for lam in path:
        alpha = lambda_eff(lam, n)
        if not np.isfinite(alpha):
            raise ValueError("lambda for LS lasso is not finite.")
        model.alpha = max(alpha, 1e-15)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            model.fit(X, y)
        out[lam] = np.asarray(model.coef_, dtype=float).copy()
    return out


def smooth_abs_and_grad(z: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    denom = np.sqrt(z * z + eps * eps)
    return denom, z / denom


def fit_lad_smooth(
    X: np.ndarray,
    y: np.ndarray,
    lam: float,
    *,
    penalty: str,
    x0: Optional[np.ndarray],
    eps_abs: float,
    eps_l1: float,
    maxiter: int,
    gtol: float,
    ftol: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Smooth LAD with lambda/n regularization.

    The target objective is mean smooth_abs(res) + (lambda/n)R(b).  We minimize
    the equivalent p*mean smooth_abs(res) + (p/n)*lambda R(b) for better scaling.
    """
    n, p = X.shape
    beta0 = np.zeros(p, dtype=float) if x0 is None else np.asarray(x0, dtype=float)
    lam = max(float(lam), 0.0) * X.shape[1] / X.shape[0]
    penalty = penalty.lower()

    def fun_grad(beta: np.ndarray) -> Tuple[float, np.ndarray]:
        res = y - X @ beta
        vals, d_abs = smooth_abs_and_grad(res, eps_abs)
        obj = float(p * np.mean(vals))
        grad = -p * (X.T @ d_abs) / n

        if penalty == "ridge":
            obj += 0.5 * lam * float(beta @ beta)
            grad = grad + lam * beta
        elif penalty == "lasso":
            bvals, db = smooth_abs_and_grad(beta, eps_l1)
            obj += lam * float(np.sum(bvals))
            grad = grad + lam * db
        else:
            raise ValueError("penalty must be 'ridge' or 'lasso'.")
        return float(obj), np.asarray(grad, dtype=float)

    opt = minimize(
        fun_grad,
        beta0,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": int(maxiter), "gtol": float(gtol), "ftol": float(ftol)},
    )
    info = {
        "success": bool(opt.success),
        "status": int(opt.status),
        "message": str(opt.message),
        "nit": int(getattr(opt, "nit", -1)),
        "fun": float(opt.fun) if np.isfinite(opt.fun) else np.nan,
    }
    return np.asarray(opt.x, dtype=float), info


def fit_lad_lasso_quantile(
    X: np.ndarray,
    y: np.ndarray,
    lam: float,
    *,
    alpha_factor: float,
    solver: str,
    max_iter: Optional[int],
    solver_threads: Optional[int] = None,
) -> np.ndarray:
    n, p = X.shape
    alpha = lambda_eff(lam, n)
    if not np.isfinite(alpha):
        raise ValueError("lambda for LAD lasso is not finite.")
    alpha = max(float(alpha_factor) * alpha, 0.0)

    solver_options = {} if max_iter is None else {"maxiter": int(max_iter)}
    if solver_threads is not None:
        solver_options.update(threads=int(solver_threads), parallel=False)
    model = QuantileRegressor(
        quantile=0.5,
        alpha=alpha,
        fit_intercept=False,
        solver=solver,
        solver_options=solver_options,
    )
    with warnings.catch_warnings():
        if solver_threads is not None:
            # SciPy forwards these supported native HiGHS options verbatim.
            warnings.filterwarnings("ignore", message="Unrecognized options detected:.*'threads'.*", category=OptimizeWarning)
        model.fit(X, y)
    return np.asarray(model.coef_, dtype=float)


def rank_loss_and_grad(beta: np.ndarray, X: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray]:
    """2/(n(n-1)) * sum_{i<j}|r_i-r_j| and a subgradient."""
    n = X.shape[0]
    if n <= 1:
        raise ValueError("n must be larger than 1 for rank loss.")
    res = y - X @ beta
    ranks = rankdata(res, method="average")
    scores = 2.0 * ranks - (n + 1.0)
    L = float(scores @ res)
    coef = 2.0 / (n * (n - 1.0))
    obj = coef * L
    grad = -coef * (X.T @ scores)
    return float(obj), np.asarray(grad, dtype=float)


def fit_rank_penalized(
    X: np.ndarray,
    y: np.ndarray,
    lam: float,
    *,
    penalty: str,
    x0: Optional[np.ndarray],
    eps_l1: float,
    maxiter: int,
    gtol: float,
    ftol: float,
    maxls: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Rank objective with lambda/n regularization.

    The target is J_rank(b) + (lambda/n)R(b).  We minimize the equivalent
    p*J_rank(b) + (p/n)*lambda R(b).
    """
    p = X.shape[1]
    beta0 = np.zeros(p, dtype=float) if x0 is None else np.asarray(x0, dtype=float)
    lam = max(float(lam), 0.0) * X.shape[1] / X.shape[0]
    penalty = penalty.lower()

    def fun_grad(beta: np.ndarray) -> Tuple[float, np.ndarray]:
        obj, grad = rank_loss_and_grad(beta, X, y)
        obj = p * obj
        grad = p * grad
        if penalty == "ridge":
            obj += 0.5 * lam * float(beta @ beta)
            grad = grad + lam * beta
        elif penalty == "lasso":
            bvals, db = smooth_abs_and_grad(beta, eps_l1)
            obj += lam * float(np.sum(bvals))
            grad = grad + lam * db
        else:
            raise ValueError("penalty must be 'ridge' or 'lasso'.")
        return float(obj), np.asarray(grad, dtype=float)

    opt = minimize(
        fun_grad,
        beta0,
        method="L-BFGS-B",
        jac=True,
        options={
            "maxiter": int(maxiter),
            "gtol": float(gtol),
            "ftol": float(ftol),
            "maxls": int(maxls),
        },
    )
    info = {
        "success": bool(opt.success),
        "status": int(opt.status),
        "message": str(opt.message),
        "nit": int(getattr(opt, "nit", -1)),
        "fun": float(opt.fun) if np.isfinite(opt.fun) else np.nan,
    }
    return np.asarray(opt.x, dtype=float), info


def _lambda_key(lam: float) -> float:
    # Stable enough for dictionary keys and CSV display.
    return float(f"{float(lam):.12g}")


def unique_lambdas(lambdas: Iterable[float], lam_min: float, lam_max: float) -> List[float]:
    vals = []
    for lam in lambdas:
        if not np.isfinite(lam):
            continue
        lam = min(max(float(lam), lam_min), lam_max)
        vals.append(_lambda_key(lam))
    return sorted(set(vals))


def lambda_grid_around(
    center: float,
    *,
    span: float,
    n: int,
    lam_min: float,
    lam_max: float,
    lam_min_pos: float,
    include_zero: bool,
) -> List[float]:
    if n < 1:
        return []
    if not np.isfinite(center) or center <= 0:
        center = math.sqrt(max(lam_min_pos, 1e-300) * lam_max)
    center = min(max(float(center), lam_min_pos), lam_max)
    lo = max(lam_min_pos, center / max(span, 1.0))
    hi = min(lam_max, center * max(span, 1.0))
    if hi < lo:
        hi = lo
    if n == 1:
        vals = [center]
    elif abs(math.log(hi) - math.log(lo)) < 1e-15:
        vals = [lo]
    else:
        vals = list(np.geomspace(lo, hi, n))
    if include_zero and lam_min <= 0.0:
        vals.append(0.0)
    return unique_lambdas(vals, lam_min, lam_max)


def fit_one_estimator(
    estimator: str,
    penalty: str,
    X: np.ndarray,
    y: np.ndarray,
    lam: float,
    *,
    x0: Optional[np.ndarray],
    cfg: Dict[str, Any],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    estimator = estimator.upper() if estimator != "rank" else "rank"
    penalty = penalty.lower()

    if estimator == "LS":
        if penalty == "ridge":
            return fit_ls_ridge(X, y, lam), {"success": True, "status": 0, "nit": np.nan, "fun": np.nan, "message": "closed_form"}
        if penalty == "lasso":
            # Path warm-start is handled separately. This single fit is a fallback.
            out = fit_ls_lasso_path(X, y, [lam], max_iter=cfg["ls_lasso_maxiter"], tol=cfg["ls_lasso_tol"])
            return out[_lambda_key(lam)], {"success": True, "status": 0, "nit": np.nan, "fun": np.nan, "message": "sklearn_lasso"}
        raise ValueError("penalty must be ridge or lasso")

    if estimator == "LAD":
        if penalty == "lasso" and cfg["lad_lasso_backend"] == "quantile":
            beta = fit_lad_lasso_quantile(
                X,
                y,
                lam,
                alpha_factor=cfg["lad_lasso_alpha_factor"],
                solver=cfg["lad_quantile_solver"],
                max_iter=cfg["lad_quantile_maxiter"],
                solver_threads=cfg.get("lad_quantile_threads"),
            )
            return beta, {"success": True, "status": 0, "nit": np.nan, "fun": np.nan, "message": "QuantileRegressor"}
        beta, info = fit_lad_smooth(
            X,
            y,
            lam,
            penalty=penalty,
            x0=x0,
            eps_abs=cfg["eps_abs"],
            eps_l1=cfg["eps_l1"],
            maxiter=cfg["lad_maxiter"],
            gtol=cfg["lad_gtol"],
            ftol=cfg["lad_ftol"],
        )
        return beta, info

    if estimator == "rank":
        if cfg.get("rank_backend", "legacy") == "admm":
            from .rank_solver import fit_rank_convex

            beta, info = fit_rank_convex(
                X, y, penalty=penalty, lam=lam, x0=x0,
                maxiter=int(cfg.get("rank_admm_maxiter", 20000)),
                atol=float(cfg.get("rank_admm_atol", 1e-6)),
                rtol=float(cfg.get("rank_admm_rtol", 1e-5)),
                rho=cfg.get("rank_admm_rho"),
            )
            info = dict(info)
            info["termination"] = str(info.get("status", ""))
            info["status"] = 0 if info["success"] else 1
            info["nit"] = info.get("iterations", 0)
            info["fun"] = info.get("objective", np.nan)
            info.setdefault("message", info.get("method", "rank_admm"))
            return beta, info
        beta, info = fit_rank_penalized(
            X,
            y,
            lam,
            penalty=penalty,
            x0=x0,
            eps_l1=cfg["eps_l1"],
            maxiter=cfg["rank_maxiter"],
            gtol=cfg["rank_gtol"],
            ftol=cfg["rank_ftol"],
            maxls=cfg["rank_maxls"],
        )
        return beta, info

    raise ValueError(f"Unknown estimator: {estimator}")


def evaluate_lambdas_for_rep(task: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Worker function.  It generates one data set and evaluates all lambdas for
    one estimator at one grid point.  Data generation is done once per rep.
    """
    estimator = task["estimator"]
    penalty = task["penalty"]
    lambdas = [float(x) for x in task["lambdas"]]
    rep = int(task["rep"])
    seed = make_seed(task["base_seed"], task["i_penalty"], task["i_df"], task["i_delta"], rep)

    rows: List[Dict[str, Any]] = []
    try:
        X, y, beta_true, diag = sample_problem(
            n=task["n"],
            p=task["p"],
            df=task["df"],
            seed=seed,
            beta_zero_prob=task["beta_zero_prob"],
            beta_slab_var=task["beta_slab_var"],
            ensure_nonzero_beta=task["ensure_nonzero_beta"],
        )

        # Sort path for warm-starts.  Large -> small works well for lasso; for
        # ridge/smooth objectives it is still a reasonable continuation path.
        path = sorted(lambdas, reverse=True)
        beta_prev: Optional[np.ndarray] = None
        info_prev: Dict[str, Any] = {}

        if estimator == "LS" and penalty == "lasso":
            try:
                beta_path = fit_ls_lasso_path(
                    X,
                    y,
                    path,
                    max_iter=task["ls_lasso_maxiter"],
                    tol=task["ls_lasso_tol"],
                )
                for lam in path:
                    beta = beta_path[_lambda_key(lam)]
                    row = {
                        **_base_eval_row(task, estimator, lam, rep, seed),
                        **diag,
                        "sqerr": sqerr(beta, beta_true),
                        "mse": mse_per_coordinate(beta, beta_true),
                        "success": True,
                        "status": 0,
                        "message": "sklearn_lasso_path",
                        "nit": np.nan,
                        "fun": np.nan,
                        "error": "",
                    }
                    rows.append(row)
                return rows
            except Exception as e:
                # Fall through to per-lambda exception rows below.
                err = repr(e)
                for lam in path:
                    rows.append({**_base_eval_row(task, estimator, lam, rep, seed), **diag, "sqerr": np.nan, "mse": np.nan, "success": False, "status": -999, "message": "exception", "nit": -1, "fun": np.nan, "error": err})
                return rows

        for lam in path:
            try:
                x0 = beta_prev
                beta, info = fit_one_estimator(
                    estimator,
                    penalty,
                    X,
                    y,
                    lam,
                    x0=x0,
                    cfg=task,
                )
                beta_prev = beta
                info_prev = info
                rows.append({
                    **_base_eval_row(task, estimator, lam, rep, seed),
                    **diag,
                    "sqerr": sqerr(beta, beta_true),
                    "mse": mse_per_coordinate(beta, beta_true),
                    "success": bool(info.get("success", True)),
                    "status": int(info.get("status", 0)) if np.isfinite(info.get("status", 0)) else 0,
                    "message": str(info.get("message", "")),
                    "nit": info.get("nit", np.nan),
                    "fun": info.get("fun", np.nan),
                    "error": "",
                    **{f"solver_{key}": value for key, value in info.items()
                       if key not in {"success", "status", "message", "nit", "fun"}},
                })
            except Exception as e:
                rows.append({
                    **_base_eval_row(task, estimator, lam, rep, seed),
                    **diag,
                    "sqerr": np.nan,
                    "mse": np.nan,
                    "success": False,
                    "status": -999,
                    "message": "exception",
                    "nit": -1,
                    "fun": np.nan,
                    "error": repr(e),
                })
    except Exception as e:
        for lam in lambdas:
            rows.append({
                **_base_eval_row(task, estimator, lam, rep, seed),
                "sqerr": np.nan,
                "mse": np.nan,
                "success": False,
                "status": -999,
                "message": "outer_exception",
                "nit": -1,
                "fun": np.nan,
                "error": traceback.format_exc(limit=3),
            })
    return rows


def _base_eval_row(task: Dict[str, Any], estimator: str, lam: float, rep: int, seed: int) -> Dict[str, Any]:
    p = int(task["p"])
    n = int(task["n"])
    return {
        "penalty": task["penalty"],
        "estimator": estimator,
        "df": float(task["df"]),
        "delta": float(task["delta"]),
        "actual_delta": float(task["actual_delta"]),
        "i_penalty": int(task["i_penalty"]),
        "i_df": int(task["i_df"]),
        "i_delta": int(task["i_delta"]),
        "rep": int(rep),
        "seed": int(seed),
        "n": int(task["n"]),
        "p": p,
        "lambda": float(lam),
        "lambda_eff": lambda_eff(lam, n),
    }


def aggregate_eval_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        raise ValueError("Cannot aggregate empty rows.")
    first = rows[0]
    out: Dict[str, Any] = {
        "penalty": first["penalty"],
        "estimator": first["estimator"],
        "df": first["df"],
        "delta": first["delta"],
        "actual_delta": first["actual_delta"],
        "i_penalty": first["i_penalty"],
        "i_df": first["i_df"],
        "i_delta": first["i_delta"],
        "n": first["n"],
        "p": first["p"],
        "lambda": first["lambda"],
        "lambda_eff": first["lambda_eff"],
        "n_rep": len(rows),
    }

    numeric_cols = [
        "mse", "sqerr", "beta_norm2", "beta_norm2_over_p", "beta_nonzero_frac",
        "signal_mean", "signal_var", "noise_mean", "noise_var", "y_var", "x_entry_var_emp",
        "nit", "fun",
    ]
    for col in numeric_cols:
        vals = np.array([r.get(col, np.nan) for r in rows], dtype=float)
        finite = np.isfinite(vals)
        out[f"{col}_mean"] = float(np.nanmean(vals)) if finite.any() else np.nan
        out[f"{col}_median"] = float(np.nanmedian(vals)) if finite.any() else np.nan
        out[f"{col}_std"] = float(np.nanstd(vals, ddof=1)) if finite.sum() >= 2 else np.nan
    out["success_rate"] = float(np.mean([bool(r.get("success", False)) and np.isfinite(r.get("mse", np.nan)) for r in rows]))
    out["finite_rate"] = float(np.mean([np.isfinite(r.get("mse", np.nan)) for r in rows]))
    errors = [str(r.get("error", "")) for r in rows if str(r.get("error", ""))]
    out["n_errors"] = len(errors)
    failed = [r for r in rows if not bool(r.get("success", False))]
    out["n_nonconverged"] = len(failed)
    out["first_numerical_message"] = str(failed[0].get("message", "")) if failed else ""
    out["first_error"] = errors[0] if errors else ""
    return out


def choose_best_lambda(
    cache,
    *,
    criterion="mse_mean",
    min_success_rate=0.8,
):
    best_lam = np.nan
    best_row = {}
    best_val = np.inf

    for lam, row in cache.items():
        success_rate = float(row.get("success_rate", 0.0))

        if success_rate < min_success_rate:
            continue

        val = float(row.get(criterion, np.nan))

        if np.isfinite(val) and val < best_val:
            best_val = val
            best_lam = lam
            best_row = row

    return best_lam, best_row


def merge_best_rows_to_summary(
    base_task: Dict[str, Any],
    best_by_est: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "penalty": base_task["penalty"],
        "df": float(base_task["df"]),
        "delta": float(base_task["delta"]),
        "actual_delta": float(base_task["actual_delta"]),
        "i_penalty": int(base_task["i_penalty"]),
        "i_df": int(base_task["i_df"]),
        "i_delta": int(base_task["i_delta"]),
        "n": int(base_task["n"]),
        "p": int(base_task["p"]),
        "n_rep": int(base_task["n_rep"]),
    }

    diagnostic_cols = [
        "beta_norm2", "beta_norm2_over_p", "beta_nonzero_frac",
        "signal_mean", "signal_var", "noise_mean", "noise_var", "y_var", "x_entry_var_emp",
    ]
    # Use diagnostics from the first available estimator; common random numbers
    # make them identical up to repeated evaluation noise.
    first_best = next((r for r in best_by_est.values() if r), {})
    for col in diagnostic_cols:
        out[f"{col}_mean"] = first_best.get(f"{col}_mean", np.nan)
        out[f"{col}_std"] = first_best.get(f"{col}_std", np.nan)

    for est in ESTIMATORS:
        row = best_by_est.get(est, {})
        out[f"lambda_{est}_opt"] = row.get("lambda", np.nan)
        out[f"lambda_eff_{est}_opt"] = row.get("lambda_eff", np.nan)
        out[f"lambda_{est}_init"] = row.get("lambda_init", np.nan)
        out[f"lambda_{est}_n_evaluated"] = row.get("lambda_n_evaluated", 0)
        out[f"mse_{est}_mean"] = row.get("mse_mean", np.nan)
        out[f"mse_{est}_median"] = row.get("mse_median", np.nan)
        out[f"mse_{est}_std"] = row.get("mse_std", np.nan)
        out[f"sqerr_{est}_mean"] = row.get("sqerr_mean", np.nan)
        out[f"sqerr_{est}_median"] = row.get("sqerr_median", np.nan)
        out[f"sqerr_{est}_std"] = row.get("sqerr_std", np.nan)
        out[f"success_rate_{est}"] = row.get("success_rate", 0.0)
        out[f"first_error_{est}"] = row.get("first_error", "")

    means = np.array([out["mse_LS_mean"], out["mse_LAD_mean"], out["mse_rank_mean"]], dtype=float)
    means_arg = np.where(np.isfinite(means), means, np.inf)
    if np.all(np.isinf(means_arg)):
        out["winner"] = "none"
        out["winner_code"] = np.nan
        out["min_mse_mean"] = np.nan
        out["winner_lambda"] = np.nan
    else:
        idx = int(np.argmin(means_arg))
        winner = ESTIMATORS[idx]
        out["winner"] = winner
        out["winner_code"] = WINNER_CODE[winner]
        out["min_mse_mean"] = float(means_arg[idx])
        out["winner_lambda"] = out[f"lambda_{winner}_opt"]

    def ratio(a: float, b: float) -> float:
        if not (np.isfinite(a) and np.isfinite(b)) or b == 0:
            return np.nan
        return float(a / b)

    out["ratio_rank_to_LAD"] = ratio(out["mse_rank_mean"], out["mse_LAD_mean"])
    out["ratio_rank_to_LS"] = ratio(out["mse_rank_mean"], out["mse_LS_mean"])
    out["ratio_LAD_to_LS"] = ratio(out["mse_LAD_mean"], out["mse_LS_mean"])
    return out


def eval_lambdas_serial(base_task: Dict[str, Any], estimator: str, lambdas: Sequence[float]) -> Tuple[Dict[float, Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Evaluate a lambda set for one grid point/estimator inside a single process.
    This is intentionally serial over reps because we parallelize over grid points.
    """
    lambdas = unique_lambdas(lambdas, base_task["lambda_min"], base_task["lambda_max"])
    if not lambdas:
        return {}, []

    all_rows: List[Dict[str, Any]] = []
    for rep in range(int(base_task["n_rep"])):
        task = dict(base_task)
        task["estimator"] = estimator
        task["lambdas"] = lambdas
        task["rep"] = rep
        all_rows.extend(evaluate_lambdas_for_rep(task))

    if base_task.get("raw_sink") is not None:
        base_task["raw_sink"].extend(all_rows)

    by_lambda: Dict[float, List[Dict[str, Any]]] = {_lambda_key(lam): [] for lam in lambdas}
    for row in all_rows:
        by_lambda[_lambda_key(row["lambda"])].append(row)

    agg: Dict[float, Dict[str, Any]] = {}
    for lam, rows in by_lambda.items():
        if rows:
            agg[lam] = aggregate_eval_rows(rows)
    return agg, list(agg.values())


def tune_lambda_local_serial(base_task: Dict[str, Any], estimator: str, lambda_init: float) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Local adaptive lambda search for one grid point/estimator.
    No inter-grid dependency; suitable for full grid-point parallelism.
    """
    lam_min = float(base_task["lambda_min"])
    lam_max = float(base_task["lambda_max"])
    lam_min_pos = float(base_task["lambda_min_pos"])
    include_zero = bool(base_task["include_zero_lambda"])

    cache: Dict[float, Dict[str, Any]] = {}
    trace_rows: List[Dict[str, Any]] = []

    def eval_candidates(cands: Sequence[float]) -> None:
        new = [c for c in unique_lambdas(cands, lam_min, lam_max) if c not in cache]
        if not new:
            return
        agg, trace = eval_lambdas_serial(base_task, estimator, new)
        cache.update(agg)
        trace_rows.extend(trace)

    center = float(lambda_init) if np.isfinite(lambda_init) and lambda_init > 0 else float(base_task[f"lambda_init_{estimator}"])
    if not np.isfinite(center) or center <= 0:
        center = float(base_task["lambda_init"])
    if not np.isfinite(center) or center <= 0:
        center = math.sqrt(lam_min_pos * lam_max)

    span = float(base_task["lambda_initial_span"])
    n_initial = int(base_task["lambda_n_initial"])
    n_refine = int(base_task["lambda_n_refine"])

    current_cands = lambda_grid_around(
        center,
        span=span,
        n=n_initial,
        lam_min=lam_min,
        lam_max=lam_max,
        lam_min_pos=lam_min_pos,
        include_zero=include_zero,
    )

    for _ in range(int(base_task["lambda_expand_rounds"]) + 1):
        eval_candidates(current_cands)
        best_lam, _ = choose_best_lambda(cache,
            criterion=base_task.get("lambda_criterion", "mse_mean"),
            min_success_rate=base_task.get("min_lambda_success_rate", 0.8),)
        finite_cands = [c for c in current_cands if c > 0]
        if not np.isfinite(best_lam) or not finite_cands:
            break
        lo_c, hi_c = min(finite_cands), max(finite_cands)
        at_low = abs(math.log(max(best_lam, lam_min_pos)) - math.log(lo_c)) < 1e-10 and lo_c > lam_min_pos
        at_high = abs(math.log(max(best_lam, lam_min_pos)) - math.log(hi_c)) < 1e-10 and hi_c < lam_max
        if at_low:
            center = max(lam_min_pos, best_lam / span)
        elif at_high:
            center = min(lam_max, best_lam * span)
        else:
            break
        current_cands = lambda_grid_around(
            center,
            span=span,
            n=n_initial,
            lam_min=lam_min,
            lam_max=lam_max,
            lam_min_pos=lam_min_pos,
            include_zero=include_zero,
        )

    refine_span = float(base_task["lambda_refine_span"])
    for _ in range(int(base_task["lambda_refine_rounds"])):
        best_lam, _ = choose_best_lambda(cache,
            criterion=base_task.get("lambda_criterion", "mse_mean"),
            min_success_rate=base_task.get("min_lambda_success_rate", 0.8),)
        if not np.isfinite(best_lam):
            break
        cands = lambda_grid_around(
            best_lam if best_lam > 0 else lam_min_pos,
            span=refine_span,
            n=n_refine,
            lam_min=lam_min,
            lam_max=lam_max,
            lam_min_pos=lam_min_pos,
            include_zero=include_zero,
        )
        eval_candidates(cands)

    best_lam, best_row = choose_best_lambda(cache,
        criterion=base_task.get("lambda_criterion", "mse_mean"),
        min_success_rate=base_task.get("min_lambda_success_rate", 0.8),)
    if not best_row:
        best_row = {
            "penalty": base_task["penalty"],
            "estimator": estimator,
            "df": base_task["df"],
            "delta": base_task["delta"],
            "actual_delta": base_task["actual_delta"],
            "i_penalty": base_task["i_penalty"],
            "i_df": base_task["i_df"],
            "i_delta": base_task["i_delta"],
            "n": base_task["n"],
            "p": base_task["p"],
            "lambda": np.nan,
            "lambda_eff": np.nan,
            "n_rep": base_task["n_rep"],
            "mse_mean": np.nan,
            "sqerr_mean": np.nan,
            "success_rate": 0.0,
            "n_errors": base_task["n_rep"],
            "first_error": "all lambda evaluations failed",
        }
    best_row = dict(best_row)
    best_row["lambda_init"] = float(lambda_init) if np.isfinite(lambda_init) else np.nan
    best_row["lambda_n_evaluated"] = len(cache)
    best_row["lambda_search_status"] = "ok" if np.isfinite(best_row.get("mse_mean", np.nan)) else "failed"
    return best_row, trace_rows


def run_grid_point_parallel(job: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Worker: tune all estimators for one independent grid point."""
    try:
        best_by_est: Dict[str, Dict[str, Any]] = {}
        trace_all: List[Dict[str, Any]] = []
        for est in ESTIMATORS:
            if est not in job["estimators"]:
                best_by_est[est] = {}
                continue
            lam0 = float(job[f"lambda_init_value_{est}"])
            best, trace = tune_lambda_local_serial(job, est, lam0)
            best["lambda_init_source"] = job.get(f"lambda_init_source_{est}", "unknown")
            best_by_est[est] = best
            trace_all.extend(trace)
        summary = merge_best_rows_to_summary(job, best_by_est)
        for est in ESTIMATORS:
            summary[f"lambda_{est}_init_source"] = job.get(f"lambda_init_source_{est}", "")
        return summary, trace_all
    except Exception:
        # Keep the whole run alive and write a diagnostic row.
        summary = {
            "penalty": job.get("penalty", ""),
            "df": job.get("df", np.nan),
            "delta": job.get("delta", np.nan),
            "actual_delta": job.get("actual_delta", np.nan),
            "i_penalty": job.get("i_penalty", -1),
            "i_df": job.get("i_df", -1),
            "i_delta": job.get("i_delta", -1),
            "n": job.get("n", np.nan),
            "p": job.get("p", np.nan),
            "n_rep": job.get("n_rep", np.nan),
            "winner": "none",
            "winner_code": np.nan,
            "min_mse_mean": np.nan,
            "outer_error": traceback.format_exc(limit=5),
        }
        return summary, []
