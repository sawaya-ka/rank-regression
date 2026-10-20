"""Repair empirical figures while retaining the recovered random-number streams.

Jobs use the original grid indices, never indices of a selected subset. Unpenalized
jobs reuse archived LS/LAD summaries. Penalized jobs retune Rank with an exact
convex objective, and retune LS/LAD only where the forced-nonzero signal changed
an actual simulated repetition. Optional overrides are reserved for smoke tests.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import _empirical as empirical
from . import _unpenalized as unpenalized
from .experiments import _empirical_defaults


KINDS = ("unpenalized_rank", "penalized_repair")


def _read_source(root: Path, relative: str) -> tuple[pd.DataFrame, dict]:
    path = root / relative
    frame = pd.read_csv(path)
    metadata = json.loads(path.with_suffix(".metadata.json").read_text())
    original = metadata.get("original_settings", {})
    args = dict(original.get("args", original))
    return frame, args


def _index_key(row: dict) -> tuple[int, int, int]:
    return int(row["i_penalty"]), int(row["i_df"]), int(row["i_delta"])


def _filtered_rows(frame: pd.DataFrame, settings: dict) -> list[dict]:
    if "n_override" in settings:
        # Avoid the almost-square high-cost corner in a reduced-size smoke run.
        frame = frame.loc[frame["delta"] >= float(settings.get("smoke_min_delta", 2.0))]
    limit = settings.get("max_jobs_per_kind")
    if limit is not None:
        frame = frame.head(int(limit))
    return frame.to_dict("records")


def get_jobs(data_root: str | Path, settings: dict | None = None) -> list[dict]:
    """Return serializable point jobs; parallelism/checkpoints belong to the caller.

    ``data_root`` is ``data/recovered``. Defaults repair all unpenalized Rank
    points and all penalized Rank points. Set ``include_all_penalized_rank=False``
    to restrict penalized jobs to the exactly identified forced-signal points.
    ``n_override`` / ``n_rep_override`` are explicitly tagged smoke computations
    and must never be merged into paper results.
    """
    settings = dict(settings or {})
    root = Path(data_root)
    kinds = settings.get("kinds", KINDS)
    if any(kind not in KINDS for kind in kinds):
        raise ValueError(f"Unknown empirical recovery kind: {kinds}")
    smoke = "n_override" in settings or "n_rep_override" in settings
    jobs: list[dict] = []

    if "unpenalized_rank" in kinds:
        frame, args = _read_source(root, "empirical/unpenalized_phase.csv")
        for row in _filtered_rows(frame, settings):
            n = int(settings.get("n_override", row["n"]))
            p = max(1, min(int(np.floor(n / float(row["delta"]))), n - 1))
            cfg = dict(args)
            cfg.update({key: value for key, value in settings.items() if key.startswith("rank_admm_")})
            cfg.update(n=n, p=p, actual_delta=n / p,
                       n_rep=int(settings.get("n_rep_override", row["n_rep"])),
                       base_seed=int(args.get("base_seed", 20260610)))
            jobs.append({"kind": "unpenalized_rank", "key": f"unpenalized_rank_{int(row['i_df']):03d}_{int(row['i_delta']):03d}",
                         "row": row, "cfg": cfg, "smoke": smoke})

    if "penalized_repair" in kinds:
        frame, args = _read_source(root, "empirical/penalized_phase.csv")
        forced = json.loads((root / "forced_nonzero_points.json").read_text())
        if int(forced.get("base_seed", args["base_seed"])) != int(args["base_seed"]):
            raise ValueError("Forced-signal point list uses a different base seed")
        if int(forced.get("n_repetitions_each", args["n_rep"])) != int(args["n_rep"]):
            raise ValueError("Forced-signal point list uses a different repetition count")
        forced_keys = {_index_key(row) for row in forced["points"]}
        source_keys = {_index_key(row) for row in frame.to_dict("records")}
        if not forced_keys.issubset(source_keys):
            raise ValueError("Forced-signal point list contains points absent from the source table")
        if not settings.get("include_all_penalized_rank", True):
            frame = frame.loc[[_index_key(row) in forced_keys for row in frame.to_dict("records")]]
        for row in _filtered_rows(frame, settings):
            n = int(settings.get("n_override", row["n"]))
            p = max(1, int(round(n / float(row["delta"]))))
            actual_delta = n / p
            cfg = _empirical_defaults(args)
            cfg.update({key: value for key, value in settings.items()
                        if key.startswith("rank_admm_") or key in {
                            "lambda_initial_span", "lambda_refine_span", "lambda_n_initial",
                            "lambda_n_refine", "lambda_expand_rounds", "lambda_refine_rounds"}})
            cfg.update(n=n, p=p, actual_delta=actual_delta,
                       n_rep=int(settings.get("n_rep_override", row["n_rep"])),
                       penalty=str(row["penalty"]), df=float(row["df"]), delta=float(row["delta"]),
                       i_penalty=int(row["i_penalty"]), i_df=int(row["i_df"]), i_delta=int(row["i_delta"]),
                       base_seed=int(args.get("base_seed", 20260610)),
                       rank_backend="admm", ensure_nonzero_beta=False, lad_quantile_threads=1,
                       lambda_criterion=args.get("lambda_criterion", "mse_median"),
                       min_lambda_success_rate=1.0,
                       lambda_min=actual_delta * float(args.get("lambda_min", 0.0)),
                       lambda_min_pos=actual_delta * float(args.get("lambda_min_pos", 1e-4)),
                       lambda_max=actual_delta * float(args.get("lambda_max", 1000.0)))
            for est in empirical.ESTIMATORS:
                key = f"lambda_{est}_opt_archived"
                if key not in row:
                    raise ValueError(f"Recovered penalty table must preserve original lambda as {key}")
                cfg[f"lambda_init_{est}"] = actual_delta * float(row[key])
            changed = _index_key(row) in forced_keys
            estimators = list(empirical.ESTIMATORS) if changed or smoke or settings.get("force_all_methods", False) else ["rank"]
            jobs.append({"kind": "penalized_repair", "key": f"penalized_repair_{int(row['i_penalty'])}_{int(row['i_df']):03d}_{int(row['i_delta']):03d}",
                         "row": row, "cfg": cfg, "estimators": estimators,
                         "forced_signal_changed": changed, "smoke": smoke})
    return jobs


def sample_original_unpenalized(n: int, delta: float, df: float, seed: int):
    """Replay the archived norm-one beta generation exactly (no change of model)."""
    if n < 3 or delta <= 1 or df <= 0:
        raise ValueError("Require n>=3, delta>1 and df>0")
    p = max(1, min(int(np.floor(n / delta)), n - 1))
    rng = np.random.default_rng(seed)
    X = rng.standard_normal(size=(n, p))
    beta = rng.standard_normal(size=p)
    norm = np.linalg.norm(beta)
    if norm == 0:
        beta[0] = 1.0
    else:
        beta /= norm
    noise = rng.standard_t(df=df, size=n)
    return X, X @ beta + noise, beta


def _winner_fields(values: dict[str, float], *, penalized: bool, lambdas: dict | None = None) -> dict:
    finite = all(np.isfinite(value) for value in values.values())
    if not finite:
        return {"winner": "none", "winner_code": np.nan,
                **({"min_mse_mean": np.nan, "winner_lambda": np.nan} if penalized else {})}
    winner = min(empirical.ESTIMATORS, key=values.__getitem__)
    out = {"winner": winner, "winner_code": empirical.WINNER_CODE[winner]}
    if penalized:
        out.update(min_mse_mean=values[winner], winner_lambda=lambdas[winner])
    for a, b in (("rank", "LAD"), ("rank", "LS"), ("LAD", "LS")):
        out[f"ratio_{a}_to_{b}"] = values[a] / values[b] if values[b] != 0 else np.nan
    return out


def _run_unpenalized(job: dict) -> dict:
    from .rank_solver import fit_rank_convex

    row, cfg = job["row"], job["cfg"]
    raw: list[dict] = []
    trace: list[dict] = []
    for rep in range(int(cfg["n_rep"])):
        seed = unpenalized.make_seed(cfg["base_seed"], int(row["i_df"]), int(row["i_delta"]), rep)
        record = {"kind": job["kind"], "df": float(row["df"]), "delta": float(row["delta"]),
                  "i_df": int(row["i_df"]), "i_delta": int(row["i_delta"]),
                  "n": int(cfg["n"]), "p": int(cfg["p"]), "actual_delta": cfg["actual_delta"],
                  "rep": rep, "seed": seed}
        try:
            X, y, beta_true = sample_original_unpenalized(cfg["n"], row["delta"], row["df"], seed)
            # The solver's n=p+1 branch solves the zero-residual-difference system exactly.
            controls = dict(maxiter=int(cfg.get("rank_admm_maxiter", 20000)),
                            atol=float(cfg.get("rank_admm_atol", 1e-6)),
                            rtol=float(cfg.get("rank_admm_rtol", 1e-5)),
                            rho=cfg.get("rank_admm_rho"))
            beta, info = fit_rank_convex(X, y, penalty="none", lam=0.0, **controls)
            # A residual-balancing cycle can exhaust the iteration budget.
            # Restart only this failed repetition, using the same data, initial
            # rho (default 1/n), tolerances, budget, and zero initialization.
            # Successful repetitions never enter this fallback.
            if (not info["success"] and info.get("status") == "max_iterations"
                    and cfg.get("rank_admm_fixed_rho_fallback", True)):
                attempt = {"rep": rep, "seed": seed,
                           "repair_version": "fixed-rho-unpenalized-v1",
                           "prior_mse_rank": unpenalized.mse(beta, beta_true),
                           "prior_solver_info": dict(info), "controls": controls,
                           "rho_schedule": "constant", "scientific_inputs_unchanged": True,
                           "success": False}
                trace.append(attempt)
                beta, info = fit_rank_convex(X, y, penalty="none", lam=0.0,
                                             adaptive_rho=False, **controls)
                attempt.update(success=bool(info["success"]), solver_info=dict(info))
            record.update(mse_rank=unpenalized.mse(beta, beta_true), rank_success=bool(info["success"]),
                          rank_status=info.get("status"), rank_nit=info.get("iterations"),
                          rank_fun=info.get("objective"), error_rank="",
                          **{f"solver_{key}": value for key, value in info.items()})
            if job.get("smoke"):
                record["mse_LS"] = unpenalized.mse(unpenalized.fit_ls(X, y), beta_true)
                record["mse_LAD"] = unpenalized.mse(unpenalized.fit_lad(X, y, solver_threads=1), beta_true)
        except Exception as exc:
            record.update(mse_rank=np.nan, rank_success=False, error_rank=repr(exc))
        raw.append(record)
    vals = np.asarray([r["mse_rank"] for r in raw])
    complete = all(r["rank_success"] and np.isfinite(r["mse_rank"]) for r in raw)
    updates = {"mse_rank_mean": float(np.mean(vals)), "mse_rank_median": float(np.median(vals)),
               "mse_rank_std": float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan,
               "success_rate_rank": float(np.mean([r["rank_success"] for r in raw])),
               "rank_solver": "convex_admm_with_certified_fixed_rho_fallback" if trace else "convex_admm_or_exact_centered_system",
               "rank_recomputed": True}
    updates.update(df=float(row["df"]), delta=float(row["delta"]),
                   i_df=int(row["i_df"]), i_delta=int(row["i_delta"]),
                   n=int(cfg["n"]), p=int(cfg["p"]), n_rep=int(cfg["n_rep"]),
                   actual_delta=float(cfg["actual_delta"]))
    means = {"LS": float(row["mse_LS_mean"]), "LAD": float(row["mse_LAD_mean"]), "rank": updates["mse_rank_mean"]}
    if job.get("smoke"):
        for est in ("LS", "LAD"):
            means[est] = float(np.mean([r.get(f"mse_{est}", np.nan) for r in raw]))
            updates[f"mse_{est}_mean"] = means[est]
    updates.update(_winner_fields(means, penalized=False))
    if not complete:
        updates.update(winner="none", winner_code=np.nan)
    return {"updates": updates, "raw": raw, "trace": trace, "complete": complete,
            "smoke": bool(job.get("smoke"))}


def _archived_best(row: dict, est: str) -> dict:
    best = {"lambda": row[f"lambda_{est}_opt"], "lambda_eff": row[f"lambda_eff_{est}_opt"],
            "lambda_init": row.get(f"lambda_{est}_init", np.nan),
            "lambda_n_evaluated": row.get(f"lambda_{est}_n_evaluated", 0),
            "success_rate": row[f"success_rate_{est}"], "first_error": row.get(f"first_error_{est}", "")}
    for metric in ("mse", "sqerr"):
        for stat in ("mean", "median", "std"):
            best[f"{metric}_{stat}"] = row.get(f"{metric}_{est}_{stat}", np.nan)
    for key, value in row.items():
        if key.startswith(("beta_", "signal_", "noise_", "y_var", "x_entry_")):
            best[key] = value
    return best


def _run_penalized(job: dict) -> dict:
    row, cfg = job["row"], dict(job["cfg"])
    best_by_est = {est: _archived_best(row, est) for est in empirical.ESTIMATORS}
    raw: list[dict] = []
    trace: list[dict] = []
    numerical_flags: dict[str, Any] = {}
    for est in job["estimators"]:
        sink: list[dict] = []
        task = dict(cfg, raw_sink=sink)
        best, est_trace = empirical.tune_lambda_local_serial(task, est, cfg[f"lambda_init_{est}"])
        best_by_est[est] = best
        selected = best.get("lambda", np.nan)
        for record in sink:
            if np.isfinite(selected) and empirical._lambda_key(record["lambda"]) == empirical._lambda_key(selected):
                raw.append(dict(record, kind=job["kind"], selected_lambda=True))
        trace.extend(dict(r, kind=job["kind"], selection_criterion=cfg["lambda_criterion"]) for r in est_trace)
        numerical_flags[f"{est}_recomputed"] = True
        numerical_flags[f"lambda_{est}_on_global_lower"] = bool(np.isfinite(selected) and np.isclose(selected, cfg["lambda_min_pos"], rtol=1e-8, atol=0))
        numerical_flags[f"lambda_{est}_on_global_upper"] = bool(np.isfinite(selected) and np.isclose(selected, cfg["lambda_max"], rtol=1e-8, atol=0))
        numerical_flags[f"lambda_{est}_failed_candidates"] = sum(float(r.get("success_rate", 0)) < 1 for r in est_trace)
    complete = all(float(best_by_est[est].get("success_rate", 0)) >= 1 and np.isfinite(best_by_est[est].get("mse_mean", np.nan)) for est in empirical.ESTIMATORS)
    updates = empirical.merge_best_rows_to_summary(cfg, best_by_est)
    updates.update(numerical_flags, forced_signal_repaired=bool(job.get("forced_signal_changed")),
                   signal_force_nonzero=False, lambda_normalization="lambda/n",
                   lambda_selection_criterion=cfg["lambda_criterion"],
                   rank_solver="convex_admm", rank_lasso_penalty="exact_absolute_value")
    for est in empirical.ESTIMATORS:
        updates[f"lambda_{est}"] = updates[f"lambda_{est}_opt"]
    if not complete:
        updates.update(winner="none", winner_code=np.nan, min_mse_mean=np.nan, winner_lambda=np.nan)
    return {"updates": updates, "raw": raw, "trace": trace, "complete": complete,
            "smoke": bool(job.get("smoke"))}


def run_job(job: dict) -> dict:
    """Run one independent point without creating threads, processes or files."""
    if job["kind"] == "unpenalized_rank":
        return _run_unpenalized(job)
    if job["kind"] == "penalized_repair":
        return _run_penalized(job)
    raise ValueError(f"Unknown recovery job kind: {job.get('kind')}")
