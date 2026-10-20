"""Checkpointed unpenalized Student-t risk calculations on an explicit grid.

A complete file records successful finite robust-risk solves, not certified
quadrature accuracy. Compare quadrature orders near delta=1 and small df.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS")
_RISK_FIELDS = ("alpha2_LS", "alpha2_LAD", "alpha2_rank", "alpha_LS", "alpha_LAD", "alpha_rank")


def _atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def _atomic_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    fields = ["df", "delta", *_RISK_FIELDS, "winner", "winner_code", "error_LAD", "error_rank",
              "quadrature_order", "rank_branch", "rank_cutoff"]
    with tmp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def _grid(values, name, minimum):
    if not isinstance(values, list) or not values:
        raise ValueError(f"settings.{name} must be a nonempty explicit list")
    result = [float(value) for value in values]
    if any(not math.isfinite(value) or value <= minimum for value in result):
        raise ValueError(f"settings.{name} values must be finite and greater than {minimum}")
    if result != sorted(set(result)):
        raise ValueError(f"settings.{name} must be sorted with no duplicates")
    return result


def _configuration(config_path):
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if config.get("experiment") != "unpenalized_theory":
        raise ValueError("Expected experiment='unpenalized_theory'")
    if config.get("preset") not in {"smoke", "paper"}:
        raise ValueError("preset must be smoke or paper")
    settings = config.get("settings", {})
    settings["df_values"] = _grid(settings.get("df_values"), "df_values", 0)
    settings["delta_values"] = _grid(settings.get("delta_values"), "delta_values", 1)
    order = settings.get("n_quad", 300)
    if isinstance(order, bool) or not isinstance(order, int) or order < 20:
        raise ValueError("settings.n_quad must be an integer >=20")
    settings["n_quad"] = order
    config["settings"] = settings
    return config


def _valid_row(row, df, delta, order):
    try:
        if row["df"] != df or row["delta"] != delta or row["quadrature_order"] != order:
            return False
        for method in ("LAD", "rank"):
            risk, alpha = row[f"alpha2_{method}"], row[f"alpha_{method}"]
            if not math.isfinite(risk) or risk <= 0 or not math.isfinite(alpha) or alpha <= 0:
                return False
            if not math.isclose(alpha*alpha, risk, rel_tol=1e-10):
                return False
        ls, alpha_ls = row["alpha2_LS"], row["alpha_LS"]
        if df <= 2:
            if ls != math.inf or alpha_ls != math.inf:
                return False
        else:
            expected = df / ((df-2)*(delta-1))
            if not math.isfinite(ls) or not math.isclose(ls, expected, rel_tol=1e-12):
                return False
            if not math.isfinite(alpha_ls) or not math.isclose(alpha_ls*alpha_ls, ls, rel_tol=1e-10):
                return False
        if not math.isfinite(row["rank_cutoff"]) or row["rank_cutoff"] < 0:
            return False
        risks = (ls,row["alpha2_LAD"],row["alpha2_rank"])
        winner = min(range(3), key=lambda i:risks[i])
        return row["winner_code"] == winner and row["winner"] == ("LS","LAD","rank")[winner]
    except (KeyError,TypeError,ValueError):
        return False


def _worker(df, delta, order):
    for variable in _THREAD_VARS:
        os.environ[variable] = "1"
    from threadpoolctl import threadpool_limits
    from .student import solve_student_t
    try:
        with threadpool_limits(limits=1):
            row = solve_student_t(df, delta, n_quad=order)
        row.update(error_LAD="", error_rank="")
        if not _valid_row(row, df, delta, order):
            raise ArithmeticError("Solver returned invalid risk, threshold, or winner")
        return {"complete":True, "row":row, "error":None}
    except Exception as exc:
        return {"complete":False, "row":None, "error":f"{type(exc).__name__}: {exc}"}


def _stored_record(record):
    value = dict(record)
    if value.get("row") is not None:
        value["row"] = dict(value["row"])
        if value["row"]["df"] <= 2:
            # Null in strict JSON denotes mathematically infinite LS risk here;
            # the explicit flag separates it from a failed robust solve.
            value["ls_infinite_variance"] = True
            value["row"]["alpha2_LS"] = None
            value["row"]["alpha_LS"] = None
    return value


def _load_record(path, fingerprint, df, delta, order):
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError,ValueError):
        return None
    if record.get("config_fingerprint") != fingerprint:
        raise ValueError(f"Checkpoint settings/source mismatch: {path}; use a fresh output directory")
    if not record.get("complete") or not isinstance(record.get("row"),dict):
        return None
    row = record["row"]
    if record.get("ls_infinite_variance") is True and df <= 2:
        row["alpha2_LS"] = math.inf
        row["alpha_LS"] = math.inf
    return record if _valid_row(row, df, delta, order) else None


def run_light_experiment(config_path: Path, output_dir: Path, workers: int, dry_run: bool=False):
    """Run/resume a grid into ``output_dir/unpenalized_student_t.csv``.

    Completed points are checkpointed atomically. A configuration/source SHA
    guard prevents resume into a different calculation. Failed points are
    retried on the next invocation, and metadata remain incomplete until every
    point returns valid results. ``dry_run`` does not create any output files.
    """
    config = _configuration(config_path)
    if isinstance(workers,bool) or not isinstance(workers,int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    root = Path(output_dir).resolve()
    output = root / "unpenalized_student_t.csv"
    meta_path = output.with_suffix(".metadata.json")
    cache = root / "checkpoints" / "unpenalized_theory"
    canonical = json.dumps(config,sort_keys=True,separators=(",",":"))
    config_sha = hashlib.sha256(canonical.encode()).hexdigest()
    source = Path(__file__).parent
    engine_sha = {name:hashlib.sha256((source/name).read_bytes()).hexdigest()
                  for name in ("student.py","light_experiments.py")}
    fingerprint = hashlib.sha256(json.dumps({"config_sha256":config_sha,"source_sha256":engine_sha},sort_keys=True).encode()).hexdigest()
    settings = config["settings"]
    order = settings["n_quad"]
    jobs = [(i,j,df,delta) for i,df in enumerate(settings["df_values"])
            for j,delta in enumerate(settings["delta_values"])]
    plan = {"experiment":"unpenalized_theory","jobs":len(jobs),"workers":workers,
            "output":str(output),"config_sha256":config_sha,"config_fingerprint":fingerprint,
            "dry_run":bool(dry_run)}
    if dry_run:
        return plan
    if meta_path.exists():
        prior = json.loads(meta_path.read_text(encoding="utf-8"))
        if prior.get("config_fingerprint") != fingerprint:
            raise ValueError("Configuration or solver source changed; use a fresh output directory")
    elif output.exists():
        raise ValueError("Existing CSV has no matching metadata; use a fresh output directory")
    metadata = {**plan,"config":config,"preset":config["preset"],"source_sha256":engine_sha,
                "noise":"standard_student_t","noise_scale":1.0,
                "signal":{"kind":"arbitrary_unpenalized","note":"risk is invariant to the coefficient vector"},
                "aspect_ratio":"n/p","risk":"limiting normalized squared coefficient error",
                "grid":{"df":settings["df_values"],"delta":settings["delta_values"]},
                "quadrature_order":order,"blas_threads_per_worker":1,
                "complete":False,"completed_jobs":0,"expected_jobs":len(jobs),
                "provenance":"new deterministic computation; not recovered historical output",
                "accuracy_note":"Completeness is a solver-success check, not an integration-error bound; compare quadrature orders near delta=1 and small df."}
    completed, pending = {}, []
    for i,j,df,delta in jobs:
        path = cache/f"df{i:04d}_delta{j:04d}.json"
        record = _load_record(path,fingerprint,df,delta,order) if path.exists() else None
        if record:
            completed[(i,j)] = record
        else:
            pending.append((i,j,df,delta,path))
    _atomic_json(meta_path,metadata)
    print(f"unpenalized_theory: {len(completed)}/{len(jobs)} cached, {len(pending)} pending; {workers} worker(s)",flush=True)
    def accept(job,record):
        i,j,df,delta,path = job
        record.update(config_sha256=config_sha,config_fingerprint=fingerprint,df=df,delta=delta)
        _atomic_json(path,_stored_record(record))
        if record["complete"]:
            completed[(i,j)] = record
        else:
            print(f"  failed df={df:g}, delta={delta:g}: {record['error']}",flush=True)
    old_environment = {name:os.environ.get(name) for name in _THREAD_VARS}
    for name in _THREAD_VARS:
        os.environ[name] = "1"
    try:
        if workers == 1:
            for job in pending:
                accept(job,_worker(job[2],job[3],order))
        elif pending:
            with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context("spawn")) as pool:
                futures = {pool.submit(_worker,job[2],job[3],order):job for job in pending}
                for future in as_completed(futures):
                    job = futures[future]
                    try:
                        record = future.result()
                    except Exception as exc:
                        record = {"complete":False,"row":None,"error":f"{type(exc).__name__}: {exc}"}
                    accept(job,record)
    finally:
        for name,value in old_environment.items():
            if value is None:
                os.environ.pop(name,None)
            else:
                os.environ[name] = value
    _atomic_csv(output,[completed[key]["row"] for key in sorted(completed)])
    metadata.update(complete=len(completed)==len(jobs),completed_jobs=len(completed))
    _atomic_json(meta_path,metadata)
    return {**plan,"complete":metadata["complete"],"completed_jobs":len(completed),"metadata":str(meta_path)}
