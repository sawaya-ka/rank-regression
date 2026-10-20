"""Unpenalized finite-sample experiment kernels, derived from the supplied Rank2 code.
Production orchestration and checkpoints are in experiments.py.
"""

from __future__ import annotations
import math
import traceback
import warnings
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from scipy.optimize import OptimizeWarning, minimize
from scipy.stats import rankdata
from sklearn.linear_model import QuantileRegressor


ESTIMATORS = ("LS", "LAD", "rank")


WINNER_CODE = {"LS": 0, "LAD": 1, "rank": 2}


def make_seed(base_seed: int, i_df: int, i_delta: int, rep: int) -> int:
    """
    Reproducible seed that is stable under multiprocessing scheduling.
    """
    ss = np.random.SeedSequence([base_seed, i_df, i_delta, rep])
    return int(ss.generate_state(1, dtype=np.uint32)[0])


def sample_problem(
    *,
    n: int,
    delta: float,
    df: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, float]:
    """
    Generate synthetic data:
        X_ij ~ N(0,1), beta_j iid N(0,1),
        y = X beta_true + eps, eps_i ~ t_df.

    Returns
    -------
    X, y, beta_true, p, actual_delta
    """
    if delta <= 1:
        raise ValueError("delta must be larger than 1.")
    if df <= 0:
        raise ValueError("df must be positive.")

    p = int(np.floor(n / delta))
    p = max(1, min(p, n - 1))
    actual_delta = n / p

    rng = np.random.default_rng(seed)

    X = rng.standard_normal(size=(n, p))

    beta_true = rng.standard_normal(size=p)
    eps = rng.standard_t(df=df, size=n)
    y = X @ beta_true + eps

    return X, y, beta_true, p, actual_delta


def mse(beta_hat: np.ndarray, beta_true: np.ndarray) -> float:
    diff = beta_hat - beta_true
    return float(diff @ diff)


def fit_ls(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Least squares. Use lstsq rather than inv(X.T @ X) for stability.
    """
    beta_hat, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta_hat


def fit_lad(
    X: np.ndarray,
    y: np.ndarray,
    *,
    solver: str = "highs",
    fit_intercept: bool = False,
    max_iter: Optional[int] = None,
    solver_threads: Optional[int] = None,
) -> np.ndarray:
    """
    LAD via sklearn QuantileRegressor at quantile=0.5.

    We set fit_intercept=False because the data-generating model here has no
    intercept. If you want an intercept, set --lad-fit-intercept.
    """
    solver_options = {}
    if max_iter is not None:
        # scipy.optimize.linprog's HiGHS options use "maxiter".
        solver_options = {"maxiter": int(max_iter)}
    if solver_threads is not None:
        solver_options.update(threads=int(solver_threads), parallel=False)

    model = QuantileRegressor(
        quantile=0.5,
        alpha=0.0,
        fit_intercept=fit_intercept,
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
    """
    Rank objective and gradient used in the user's reference code, evaluated
    together to avoid computing rankdata twice per optimizer iteration.

    res = y - X beta.

    loss = sqrt(12)/n * { rank(res)^T res/(n+1) - sum(res)/2 }
    grad = -sqrt(12)/(n(n+1)) * X^T { rank(res) - (n+1)/2 }.

    Average ranks give a valid subgradient at residual ties.
    Ties can occur at the fitted optimum even with continuous noise.
    """
    n = X.shape[0]
    res = y - X @ beta
    ranks = rankdata(res, method="average")

    loss = math.sqrt(12.0) / n * (
        float(ranks @ res) / (n + 1.0) - 0.5 * float(np.sum(res))
    )

    centered_ranks = ranks - 0.5 * (n + 1.0)
    grad = -math.sqrt(12.0) / (n * (n + 1.0)) * (X.T @ centered_ranks)

    return float(loss), np.asarray(grad, dtype=float)


def fit_rank(
    X: np.ndarray,
    y: np.ndarray,
    *,
    x0: Optional[np.ndarray] = None,
    maxiter: int = 500,
    gtol: float = 1e-6,
    ftol: float = 1e-9,
    maxls: int = 50,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Rank regression via L-BFGS-B on the rank objective.

    Returns beta_hat and optimizer metadata. The returned beta is used even if
    scipy reports non-convergence, but the convergence flag is recorded.
    """
    p = X.shape[1]
    if x0 is None:
        x0 = np.zeros(p, dtype=float)
    else:
        x0 = np.asarray(x0, dtype=float)

    opt = minimize(
        rank_loss_and_grad,
        x0,
        args=(X, y),
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
        "rank_success": bool(opt.success),
        "rank_status": int(opt.status),
        "rank_message": str(opt.message),
        "rank_nit": int(getattr(opt, "nit", -1)),
        "rank_fun": float(opt.fun) if np.isfinite(opt.fun) else np.nan,
    }

    return np.asarray(opt.x, dtype=float), info


def simulate_one_rep(
    *,
    df: float,
    delta: float,
    i_df: int,
    i_delta: int,
    rep: int,
    seed: int,
    n: int,
    do_lad: bool,
    do_rank: bool,
    lad_solver: str,
    lad_fit_intercept: bool,
    lad_max_iter: Optional[int],
    rank_start: str,
    rank_maxiter: int,
    rank_gtol: float,
    rank_ftol: float,
) -> Dict[str, Any]:
    """
    Run one Monte Carlo repetition and return a raw row.
    """
    row: Dict[str, Any] = {
        "df": float(df),
        "delta": float(delta),
        "i_df": int(i_df),
        "i_delta": int(i_delta),
        "rep": int(rep),
        "seed": int(seed),
        "n": int(n),
    }

    try:
        X, y, beta_true, p, actual_delta = sample_problem(
            n=n,
            delta=delta,
            df=df,
            seed=seed,
        )
        row["p"] = int(p)
        row["actual_delta"] = float(actual_delta)

        beta_ls = fit_ls(X, y)
        row["mse_LS"] = mse(beta_ls, beta_true)
        row["success_LS"] = True
        row["error_LS"] = ""

        beta_lad = None
        if do_lad:
            try:
                beta_lad = fit_lad(
                    X,
                    y,
                    solver=lad_solver,
                    fit_intercept=lad_fit_intercept,
                    max_iter=lad_max_iter,
                )
                row["mse_LAD"] = mse(beta_lad, beta_true)
                row["success_LAD"] = True
                row["error_LAD"] = ""
            except Exception as e:
                row["mse_LAD"] = np.nan
                row["success_LAD"] = False
                row["error_LAD"] = repr(e)
        else:
            row["mse_LAD"] = np.nan
            row["success_LAD"] = False
            row["error_LAD"] = "skipped"

        if do_rank:
            try:
                if rank_start.lower() == "lad" and beta_lad is not None and np.all(np.isfinite(beta_lad)):
                    x0 = beta_lad
                elif rank_start.lower() == "ls" and np.all(np.isfinite(beta_ls)):
                    x0 = beta_ls
                elif rank_start.lower() == "zero":
                    x0 = np.zeros_like(beta_true)
                else:
                    x0 = beta_ls if np.all(np.isfinite(beta_ls)) else np.zeros_like(beta_true)

                beta_rank, info = fit_rank(
                    X,
                    y,
                    x0=x0,
                    maxiter=rank_maxiter,
                    gtol=rank_gtol,
                    ftol=rank_ftol,
                )
                row["mse_rank"] = mse(beta_rank, beta_true)
                row.update(info)
                row["error_rank"] = ""
            except Exception as e:
                row["mse_rank"] = np.nan
                row["rank_success"] = False
                row["rank_status"] = -999
                row["rank_message"] = "exception"
                row["rank_nit"] = -1
                row["rank_fun"] = np.nan
                row["error_rank"] = repr(e)
        else:
            row["mse_rank"] = np.nan
            row["rank_success"] = False
            row["rank_status"] = -999
            row["rank_message"] = "skipped"
            row["rank_nit"] = -1
            row["rank_fun"] = np.nan
            row["error_rank"] = "skipped"

    except Exception as e:
        row.update(
            {
                "p": np.nan,
                "actual_delta": np.nan,
                "mse_LS": np.nan,
                "mse_LAD": np.nan,
                "mse_rank": np.nan,
                "success_LS": False,
                "success_LAD": False,
                "rank_success": False,
                "rank_status": -999,
                "rank_message": "outer exception",
                "rank_nit": -1,
                "rank_fun": np.nan,
                "error_LS": repr(e),
                "error_LAD": "outer exception",
                "error_rank": traceback.format_exc(limit=3),
            }
        )

    return row


def summarize_raw_rows(raw_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Summarize Monte Carlo repetitions at one (df, delta) grid point.
    Winner is defined by the smallest mean MSE.
    """
    if len(raw_rows) == 0:
        raise ValueError("raw_rows is empty.")

    base = raw_rows[0]
    out: Dict[str, Any] = {
        "df": base["df"],
        "delta": base["delta"],
        "i_df": base["i_df"],
        "i_delta": base["i_delta"],
        "n": base["n"],
        "p": base.get("p", np.nan),
        "actual_delta": base.get("actual_delta", np.nan),
        "n_rep": len(raw_rows),
    }

    for est in ESTIMATORS:
        col = f"mse_{est}"
        vals = np.array([r.get(col, np.nan) for r in raw_rows], dtype=float)
        finite = np.isfinite(vals)
        out[f"mse_{est}_mean"] = float(np.nanmean(vals)) if finite.any() else np.nan
        out[f"mse_{est}_median"] = float(np.nanmedian(vals)) if finite.any() else np.nan
        out[f"mse_{est}_std"] = float(np.nanstd(vals, ddof=1)) if finite.sum() >= 2 else np.nan
        out[f"finite_rate_{est}"] = float(finite.mean())
        flag = "rank_success" if est == "rank" else f"success_{est}"
        out[f"success_rate_{est}"] = float(np.mean([bool(r.get(flag, False)) and np.isfinite(r.get(col, np.nan)) for r in raw_rows]))

    means = np.array(
        [out["mse_LS_mean"], out["mse_LAD_mean"], out["mse_rank_mean"]],
        dtype=float,
    )
    means_for_argmin = np.where(np.isfinite(means), means, np.inf)

    if np.all(np.isinf(means_for_argmin)):
        winner = "none"
        winner_code = np.nan
        min_mse = np.nan
    else:
        idx = int(np.argmin(means_for_argmin))
        winner = ESTIMATORS[idx]
        winner_code = WINNER_CODE[winner]
        min_mse = float(means_for_argmin[idx])

    out["winner"] = winner
    out["winner_code"] = winner_code
    out["min_mse_mean"] = min_mse

    # Pairwise ratios are often useful for diagnostics.
    def ratio(a: float, b: float) -> float:
        if not (np.isfinite(a) and np.isfinite(b)) or b == 0:
            return np.nan
        return float(a / b)

    out["ratio_rank_to_LAD"] = ratio(out["mse_rank_mean"], out["mse_LAD_mean"])
    out["ratio_rank_to_LS"] = ratio(out["mse_rank_mean"], out["mse_LS_mean"])
    out["ratio_LAD_to_LS"] = ratio(out["mse_LAD_mean"], out["mse_LS_mean"])

    return out


def run_grid_point(job: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    raw_rows: List[Dict[str, Any]] = []

    for rep in range(job["n_rep"]):
        seed = make_seed(job["base_seed"], job["i_df"], job["i_delta"], rep)
        raw = simulate_one_rep(
            df=job["df"],
            delta=job["delta"],
            i_df=job["i_df"],
            i_delta=job["i_delta"],
            rep=rep,
            seed=seed,
            n=job["n"],
            do_lad=job["do_lad"],
            do_rank=job["do_rank"],
            lad_solver=job["lad_solver"],
            lad_fit_intercept=job["lad_fit_intercept"],
            lad_max_iter=job["lad_max_iter"],
            rank_start=job["rank_start"],
            rank_maxiter=job["rank_maxiter"],
            rank_gtol=job["rank_gtol"],
            rank_ftol=job["rank_ftol"],
        )
        raw_rows.append(raw)

    summary = summarize_raw_rows(raw_rows)
    return summary, raw_rows
