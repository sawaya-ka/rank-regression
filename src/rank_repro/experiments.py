"""Reproducible CPU experiment runners with immutable run settings and checkpoints.

``run_experiment`` is intentionally separate from figure rendering.  A smoke
configuration checks the pipeline; it does not reproduce a manuscript figure.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import math
import multiprocessing
import os
import platform
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

_OUTPUTS = {
    "theory_penalized": "theory/penalized_phase.csv",
    "theory_penalized_are": "theory/penalized_are.csv",
    "empirical_unpenalized": "empirical/unpenalized_phase.csv",
    "empirical_penalized": "empirical/penalized_phase.csv",
}
_THREAD_VARIABLES = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS",
)


def _plain(value: Any) -> Any:
    """Produce strict JSON; unavailable numerical values are represented by null."""
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if hasattr(value, "item"):
        return _plain(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_plain(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _atomic_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(_plain(rows))
    temporary.replace(path)


def _grid(values: Any, name: str) -> list[float]:
    if not isinstance(values, list) or not values:
        raise ValueError(f"settings.{name} must be a nonempty explicit list")
    result = [float(value) for value in values]
    if any(not math.isfinite(value) or value <= 0 for value in result):
        raise ValueError(f"settings.{name} must contain positive finite numbers")
    if result != sorted(set(result)):
        raise ValueError(f"settings.{name} must be sorted and contain no duplicates")
    return result


def _configuration(name: str, path: str | Path) -> dict:
    config = json.loads(Path(path).read_text())
    if name not in _OUTPUTS:
        raise ValueError(f"Unknown experiment {name!r}; choose from {sorted(_OUTPUTS)}")
    if config.get("experiment") != name:
        raise ValueError("The requested experiment and configuration disagree")
    settings = dict(config["settings"])
    settings["df_values"] = _grid(settings.get("df_values"), "df_values")
    settings["delta_values"] = _grid(settings.get("delta_values"), "delta_values")
    if name == "empirical_unpenalized":
        if min(settings["delta_values"]) <= 1:
            raise ValueError("Unpenalized estimation requires delta > 1")
    else:
        settings.setdefault("penalties", ["ridge", "lasso"])
        if not settings["penalties"] or any(p not in {"ridge", "lasso"} for p in settings["penalties"]):
            raise ValueError("penalties must be a nonempty list drawn from ridge, lasso")
        settings.setdefault("beta_zero_prob", 0.8)
        settings.setdefault("beta_slab_var", 5.0)
        if not (0 <= settings["beta_zero_prob"] < 1 and settings["beta_slab_var"] > 0):
            raise ValueError("Invalid spike-and-slab parameters")
    if name.startswith("empirical"):
        for field in ("n", "n_rep", "base_seed"):
            if field not in settings:
                raise ValueError(f"settings.{field} is required for empirical runs")
        if int(settings["n"]) <= 2 or int(settings["n_rep"]) < 1:
            raise ValueError("Require n > 2 and n_rep >= 1")
        if any(int(settings[field]) != settings[field] for field in ("n", "n_rep", "base_seed")) or settings["base_seed"] < 0:
            raise ValueError("n, n_rep, and base_seed must be integers, with base_seed >= 0")
    if name == "empirical_penalized":
        if settings.get("lambda_criterion") not in {"mse_mean", "mse_median"}:
            raise ValueError("Choose lambda_criterion explicitly: mse_mean or mse_median")
        if settings.get("ensure_nonzero_beta", False):
            raise ValueError("Forcing a nonzero signal changes the stated iid prior; leave this false")
        settings = _empirical_defaults(settings)
        if not 0 < settings["lambda_min_pos"] < settings["lambda_max"]:
            raise ValueError("Require 0 < lambda_min_pos < lambda_max")
    if name == "empirical_unpenalized":
        settings.setdefault("rank_maxiter", 500)
        settings.setdefault("rank_gtol", 1e-6)
        settings.setdefault("rank_ftol", 1e-9)
    if name.startswith("theory"):
        settings.setdefault("estimators", ["ls", "rank"] if name.endswith("_are") else ["ls", "lad", "rank"])
        if any(est not in {"ls", "lad", "rank"} for est in settings["estimators"]):
            raise ValueError("Unknown theoretical estimator")
        for field in ("lambda_min", "lambda_max", "n_lambda", "gh_n", "grid_size"):
            if field not in settings:
                raise ValueError(f"settings.{field} is required for theoretical runs")
        if not 0 < settings["lambda_min"] < settings["lambda_max"]:
            raise ValueError("Require 0 < lambda_min < lambda_max")
    config["settings"] = settings
    return config


def _fingerprint(config: dict) -> str:
    source = Path(__file__).parent
    hashes = {}
    for filename in ("experiments.py", "_unpenalized.py", "_empirical.py", "_theory_penalized.py"):
        hashes[filename] = hashlib.sha256((source / filename).read_bytes()).hexdigest()
    return hashlib.sha256(json.dumps({"config": config, "engines": hashes}, sort_keys=True).encode()).hexdigest()


def _empirical_defaults(settings: dict) -> dict:
    defaults = {
        "beta_zero_prob": 0.8, "beta_slab_var": 5.0, "ensure_nonzero_beta": False,
        "estimators": ["LS", "LAD", "rank"],
        "ls_lasso_maxiter": 5000, "ls_lasso_tol": 1e-5,
        "lad_lasso_backend": "quantile", "lad_lasso_alpha_factor": 0.5,
        "lad_quantile_solver": "highs", "lad_quantile_maxiter": None,
        "lad_maxiter": 1000, "lad_gtol": 1e-6, "lad_ftol": 1e-10,
        "rank_maxiter": 1000, "rank_gtol": 1e-6, "rank_ftol": 1e-10, "rank_maxls": 50,
        "eps_abs": 1e-4, "eps_l1": 1e-5,
        "lambda_min": 0.0, "lambda_min_pos": 1e-4, "lambda_max": 1000.0,
        "include_zero_lambda": False, "lambda_init": 1.0,
        "lambda_initial_span": 8.0, "lambda_refine_span": 3.0,
        "lambda_n_initial": 7, "lambda_n_refine": 5,
        "lambda_expand_rounds": 1, "lambda_refine_rounds": 1,
        "min_lambda_success_rate": 1.0,
    }
    defaults.update(settings)
    for estimator in ("LS", "LAD", "rank"):
        defaults.setdefault(f"lambda_init_{estimator}", defaults["lambda_init"])
        defaults.setdefault(f"lambda_init_value_{estimator}", defaults[f"lambda_init_{estimator}"])
        defaults[f"lambda_init_source_{estimator}"] = "configuration"
    return defaults


def _jobs(name: str, settings: dict) -> list[dict]:
    penalties = [None] if name == "empirical_unpenalized" else settings["penalties"]
    jobs = []
    for ip, penalty in enumerate(penalties):
        for ix, df in enumerate(settings["df_values"]):
            for iy, delta in enumerate(settings["delta_values"]):
                job = dict(settings)
                job.update(experiment=name, df=df, delta=delta, i_penalty=ip, i_df=ix, i_delta=iy)
                if penalty:
                    job["penalty"] = penalty
                if name == "empirical_penalized":
                    job = _empirical_defaults(job)
                    job["p"] = max(1, round(job["n"] / delta))
                    job["actual_delta"] = job["n"] / job["p"]
                job.pop("df_values", None)
                job.pop("delta_values", None)
                jobs.append(job)
    return jobs


def _run_theory(job: dict) -> tuple[dict, list[dict], list[dict], bool]:
    from . import _theory_penalized as kernel

    kernel.init_gh(int(job["gh_n"]))
    results, traces = {}, []
    complete = True
    for estimator in job["estimators"]:
        if estimator == "ls" and job["df"] <= 2:
            results[estimator] = {"alpha2_opt": math.inf, "lambda_opt": math.nan, "unavailable_reason": "infinite_error_variance"}
            continue
        sigma2 = kernel.t_variance_for_ls(job["df"], False) if estimator == "ls" else None
        seed = None
        if estimator == "ls" and job["penalty"] == "ridge":
            seed = kernel.analytic_ls_ridge_lambda_seed(job["delta"], sigma2, 1 - job["beta_zero_prob"], job["beta_slab_var"], job["lambda_min"], job["lambda_max"])
        result = kernel.optimize_lambda_direct(
            estimator, job["penalty"], job["delta"], job["df"], False,
            1 - job["beta_zero_prob"], job["beta_slab_var"],
            job["lambda_min"], job["lambda_max"], int(job["n_lambda"]),
            int(job["grid_size"]), 1.0, int(job["gh_n"]), lambda_hint=seed, sigma2=sigma2,
        )
        results[estimator] = result
        complete = complete and kernel.good_solution(result.get("state"))
        for row in result["trace"]:
            traces.append({"df": job["df"], "delta": job["delta"], "penalty": job["penalty"], "estimator": estimator, **row})
    summary = kernel.make_phase_row(job["df"], job["delta"], job["penalty"], results.get("ls", {}), results.get("lad", {}), results.get("rank", {}))
    for estimator, result in results.items():
        state = result.get("state") or {}
        summary[f"{estimator}_residual_squared"] = state.get("cost", math.nan)
        summary[f"{estimator}_lambda_evaluations"] = len(result.get("trace", []))
        summary[f"{estimator}_status"] = result.get("unavailable_reason", "solved" if kernel.good_solution(state) else "failed")
    if not complete:
        summary.update(winner="none", winner_code=-1)
    return summary, traces, [], bool(complete)


def _run_empirical_unpenalized(job: dict) -> tuple[dict, list[dict], list[dict], bool]:
    from . import _unpenalized as kernel

    task = dict(job)
    task.update(do_lad=True, do_rank=True, lad_solver="highs", lad_fit_intercept=False, lad_max_iter=None, rank_start="lad")
    task.setdefault("rank_maxiter", 500)
    task.setdefault("rank_gtol", 1e-6)
    task.setdefault("rank_ftol", 1e-9)
    summary, raw = kernel.run_grid_point(task)
    complete = all(summary[f"success_rate_{est}"] == 1 for est in kernel.ESTIMATORS)
    if not complete:
        summary.update(winner="none", winner_code=None)
    return summary, [], raw, bool(complete)


def _run_empirical_penalized(job: dict) -> tuple[dict, list[dict], list[dict], bool]:
    from . import _empirical as kernel

    summary, trace = kernel.run_grid_point_parallel(job)
    complete = all(summary.get(f"success_rate_{est}", 0) == 1 and math.isfinite(summary.get(f"mse_{est}_mean", math.nan)) for est in kernel.ESTIMATORS)
    if not complete:
        summary.update(winner="none", winner_code=None)
    return summary, trace, [], bool(complete)


def _worker(job: dict) -> dict:
    for name in _THREAD_VARIABLES:
        os.environ[name] = "1"
    import numpy  # Load thread pools before imposing runtime limits.
    import scipy
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1):
        if job["experiment"].startswith("theory"):
            summary, trace, raw, complete = _run_theory(job)
        elif job["experiment"] == "empirical_unpenalized":
            summary, trace, raw, complete = _run_empirical_unpenalized(job)
        else:
            summary, trace, raw, complete = _run_empirical_penalized(job)
    return {"key": [job["i_penalty"], job["i_df"], job["i_delta"]], "complete": complete, "summary": summary, "trace": trace, "raw": raw}


def _metadata(name: str, config: dict, fingerprint: str, workers: int) -> dict:
    settings = config["settings"]
    result = {
        "experiment": name, "preset": config.get("preset", "custom"),
        "config": config, "config_fingerprint": fingerprint,
        "noise": "standard_student_t",
        "design": {"distribution": "gaussian", "entry_variance": "1" if name == "empirical_unpenalized" else "1/p"},
        "error_metric": "squared_l2_error" if name == "empirical_unpenalized" else "squared_l2_error/p",
        "signal": {"kind": "gaussian", "variance": 1} if name == "empirical_unpenalized" else {
            "kind": "spike_slab", "nonzero_probability": 1 - settings["beta_zero_prob"], "nonzero_variance": settings["beta_slab_var"], "force_nonzero": False,
        },
        "grid": {"df": settings["df_values"], "delta": settings["delta_values"], "penalty": settings.get("penalties", [])},
        "lambda_convention": "loss_average + (lambda/n) * sum(g(beta_j))",
        "workers": workers, "blas_threads_per_worker": 1,
        "python_version": platform.python_version(),
        "package_versions": {package: importlib.metadata.version(package) for package in ("numpy", "scipy", "pandas", "scikit-learn", "threadpoolctl")},
        "complete": False,
        "provenance": "new computation with corrected and explicit settings; not a recovered historical run",
    }
    if name.startswith("empirical"):
        result.update(n=settings["n"], n_rep=settings["n_rep"], base_seed=settings["base_seed"])
        result["rank_solver"] = "legacy L-BFGS-B with a rank subgradient; convergence flag is not an optimality certificate"
    if name == "empirical_unpenalized":
        result["selection"] = {"criterion": "not_applicable", "evaluation": "monte_carlo_repetitions", "display_statistic": "mean"}
        result["normalization_conversion"] = "X_paper = X_code/sqrt(p), beta_paper = sqrt(p)*beta_code; squared_l2_error_paper/p = squared_l2_error_code"
        result["lambda_convention"] = "not_applicable"
    if name == "empirical_penalized":
        result["selection"] = {"criterion": settings["lambda_criterion"], "evaluation": "same_repetitions", "display_statistic": "mean", "target": "finite adaptive candidate set", "independent_test_set": False}
        result["approximations"] = {"lad_ridge_smooth_abs_epsilon": settings.get("eps_abs", 1e-4), "rank_lasso_smooth_l1_epsilon": settings.get("eps_l1", 1e-5)}
    return result


def run_experiment(name: str, config_path: str | Path, output_dir: str | Path, workers: int, dry_run: bool = False) -> dict:
    """Run or resume an explicit configuration; ``output_dir`` is the data root.

    Cached jobs are reused only if their settings/source fingerprint matches and
    all requested fits succeeded.  Failed or corrupt jobs are retried on resume.
    No automatic cloud or VM launch is performed.
    """
    config = _configuration(name, config_path)
    if int(workers) < 1:
        raise ValueError("workers must be >= 1")
    root = Path(output_dir).resolve()
    output = root / _OUTPUTS[name]
    jobs = _jobs(name, config["settings"])
    fingerprint = _fingerprint(config)
    metadata_path = output.with_suffix(".metadata.json")
    cache = root / "checkpoints" / name
    plan = {"experiment": name, "jobs": len(jobs), "workers": int(workers), "output": str(output), "config_fingerprint": fingerprint, "dry_run": dry_run}
    if dry_run:
        return plan
    if metadata_path.exists():
        previous = json.loads(metadata_path.read_text())
        if previous.get("config_fingerprint") != fingerprint:
            raise ValueError(f"Settings or engine source changed: use a fresh output directory instead of reusing {output}")
    elif output.exists():
        raise ValueError(f"Existing CSV has no matching run metadata: {output}")
    metadata = _metadata(name, config, fingerprint, int(workers))
    _atomic_json(metadata_path, metadata)
    cache.mkdir(parents=True, exist_ok=True)
    completed, pending = [], []
    for job in jobs:
        key = f"{job['i_penalty']:02d}_{job['i_df']:04d}_{job['i_delta']:04d}"
        job["checkpoint"] = str(cache / f"{key}.json")
        path = Path(job["checkpoint"])
        try:
            record = json.loads(path.read_text())
            if record.get("fingerprint") == fingerprint and record.get("complete"):
                completed.append(record)
                continue
        except (OSError, ValueError):
            pass
        pending.append(job)
    print(f"{name}: {len(completed)}/{len(jobs)} valid cached jobs; {len(pending)} pending; {workers} worker(s)", flush=True)
    old_environment = {key: os.environ.get(key) for key in _THREAD_VARIABLES}
    for key in _THREAD_VARIABLES:
        os.environ[key] = "1"
    try:
        if int(workers) == 1:
            iterable = ((job, _worker(job)) for job in pending)
            for job, record in iterable:
                record["fingerprint"] = fingerprint
                _atomic_json(Path(job["checkpoint"]), record)
                completed.append(record)
                print(f"  {len(completed)}/{len(jobs)} ({'complete' if record['complete'] else 'failed fit; see checkpoint'})", flush=True)
        else:
            with ProcessPoolExecutor(max_workers=int(workers), mp_context=multiprocessing.get_context("spawn")) as pool:
                futures = {pool.submit(_worker, job): job for job in pending}
                for future in as_completed(futures):
                    job = futures[future]
                    record = future.result()
                    record["fingerprint"] = fingerprint
                    _atomic_json(Path(job["checkpoint"]), record)
                    completed.append(record)
                    print(f"  {len(completed)}/{len(jobs)} ({'complete' if record['complete'] else 'failed fit; see checkpoint'})", flush=True)
    finally:
        for key, value in old_environment.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    completed.sort(key=lambda record: record["key"])
    _atomic_csv(output, [record["summary"] for record in completed])
    trace_path = output.with_name(output.stem + "_lambda_trace.csv")
    raw_path = output.with_name(output.stem + "_raw.csv")
    _atomic_csv(trace_path, [row for record in completed for row in record["trace"]])
    _atomic_csv(raw_path, [row for record in completed for row in record["raw"]])
    metadata.update(complete=all(record["complete"] for record in completed), completed_jobs=sum(record["complete"] for record in completed), expected_jobs=len(jobs))
    _atomic_json(metadata_path, metadata)
    return {**plan, "complete": metadata["complete"], "completed_jobs": metadata["completed_jobs"], "metadata": str(metadata_path)}
