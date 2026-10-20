#!/usr/bin/env python3
"""Execute the experiment plan locally or inside the detached VM worker."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_variable] = "1"


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if hasattr(value, "item"):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(plan):
    paths = [Path(__file__), *sorted((ROOT / "src/rank_repro").glob("*.py")),
             *sorted((ROOT / plan["data_root"]).rglob("*.csv")),
             *sorted((ROOT / plan["data_root"]).rglob("*.json")),
             ROOT / plan["unpenalized_theory_source"]]
    hashes = {str(p.relative_to(ROOT)): file_sha(p) for p in paths}
    return hashlib.sha256(json.dumps({"plan": plan, "files": hashes}, sort_keys=True).encode()).hexdigest()


def make_jobs(plan):
    import numpy as np
    import pandas as pd
    from rank_repro.recovery_empirical import get_jobs

    jobs = []
    phase = pd.read_csv(ROOT / plan["unpenalized_theory_source"], float_precision="round_trip")
    risks = phase[["alpha2_LS", "alpha2_LAD", "alpha2_rank"]].to_numpy()
    ordered = np.sort(risks, axis=1)
    close = (ordered[:, 1] - ordered[:, 0]) / ordered[:, 0] <= plan["quadrature"]["near_tie_relative_gap"]
    corner = (phase.df <= plan["quadrature"]["low_df_max"]) & (phase.delta <= plan["quadrature"]["small_delta_max"])
    selected = phase[close | corner]
    if plan["mode"] == "smoke":
        selected = pd.DataFrame([{"df": 8., "delta": 2.}])
    for _, row in selected.iterrows():
        jobs.append({"kind": "quadrature", "df": float(row.df), "delta": float(row.delta), "settings": plan["quadrature"]})

    data_root = ROOT / plan["data_root"]
    theory = pd.read_csv(data_root / "theory/penalized_phase.csv", float_precision="round_trip")
    selected = theory[theory.penalty == "lasso"]
    if plan["mode"] == "smoke":
        selected = selected[(selected.df >= 8) & (selected.delta >= 4)].iloc[:1]
    for _, row in selected.iterrows():
        jobs.append({"kind": "theory_lasso", "df": float(row.df), "delta": float(row.delta),
                     "penalty": "lasso", **plan["theory_lasso"]})
    empirical = get_jobs(data_root, plan["empirical"])
    for job in empirical:
        jobs.append(job)
    for index, job in enumerate(jobs):
        job["job_id"] = f"{index:06d}_{job['kind']}"
    return jobs


def worker(job):
    from threadpoolctl import threadpool_limits
    started = time.monotonic()
    try:
        with threadpool_limits(limits=1):
            if job["kind"] == "quadrature":
                from rank_repro.student import solve_student_t
                previous, history, complete = None, [], False
                for order in job["settings"]["orders"]:
                    row = solve_student_t(job["df"], job["delta"], n_quad=order)
                    if any(not math.isfinite(row[k]) or row[k] <= 0 for k in ("alpha2_rank", "alpha2_LAD")):
                        raise ValueError(f"Quadrature order {order} returned nonpositive or nonfinite risk")
                    entry = {"order": order, "rank": row["alpha2_rank"], "lad": row["alpha2_LAD"]}
                    if previous is not None:
                        change = max(abs(row[k] - previous[k]) / row[k] for k in ("alpha2_rank", "alpha2_LAD"))
                        entry["max_relative_change"] = change
                        complete = change <= job["settings"]["relative_tolerance"]
                    history.append(entry)
                    if complete:
                        break
                    previous = row
                result = {"updates": row, "complete": complete, "raw": [], "trace": history,
                          "note": "Order comparison is an accuracy diagnostic, not a rigorous error bound."}
            elif job["kind"] == "theory_lasso":
                from rank_repro.experiments import _run_theory
                row, trace, raw, complete = _run_theory(job)
                result = {"updates": row, "trace": trace, "raw": raw, "complete": complete}
            else:
                from rank_repro.recovery_empirical import run_job
                result = run_job(job)
                coordinates = {key: job["row"][key] for key in ("df", "delta", "penalty") if key in job["row"]}
                result["updates"] = {**coordinates, **result.get("updates", {})}
        result.update(job_id=job["job_id"], kind=job["kind"], elapsed_seconds=time.monotonic() - started)
        return clean(result)
    except Exception as exc:
        return {"job_id": job["job_id"], "kind": job["kind"], "complete": False,
                "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(),
                "elapsed_seconds": time.monotonic() - started, "updates": {}, "raw": [], "trace": []}


def coordinate_key(row, penalized=False):
    key = (round(float(row["df"]), 11), round(float(row["delta"]), 11))
    return (str(row["penalty"]), *key) if penalized else key


def refresh_empirical_winners(frame):
    import numpy as np
    methods = ("LS", "LAD", "rank")
    risks = frame[[f"mse_{method}_mean" for method in methods]].to_numpy(dtype=float)
    winner = np.argmin(risks, axis=1)
    frame["winner"] = [methods[i] for i in winner]
    frame["winner_code"] = winner
    frame["min_mse_mean"] = np.min(risks, axis=1)
    if "winner_lambda" in frame:
        frame["winner_lambda"] = [frame.iloc[j][f"lambda_{method}_opt"] for j, method in enumerate(frame["winner"])]
    for name, first, second in (("rank_to_LAD", "rank", "LAD"), ("rank_to_LS", "rank", "LS"), ("LAD_to_LS", "LAD", "LS")):
        frame[f"ratio_{name}"] = frame[f"mse_{first}_mean"] / frame[f"mse_{second}_mean"]


def write_outputs(plan, output, jobs, fingerprint_value):
    """Merge only validated new rows; keep failed tables explicitly incomplete."""
    import pandas as pd
    data_root = output / "data"
    shutil.copytree(ROOT / plan["data_root"], data_root, dirs_exist_ok=True)
    parent_manifest = data_root / "recovery_manifest.json"
    if parent_manifest.exists():
        parent_manifest.replace(data_root / "parent_recovery_manifest.json")
    for name in ("unpenalized_student_t.csv", "unpenalized_student_t.metadata.json", "gaussian_scores.csv", "gaussian_parameters.csv"):
        shutil.copyfile(ROOT / "data" / name, data_root / name)
    groups = {
        "quadrature": "unpenalized_student_t.csv",
        "theory_lasso": "theory/penalized_phase.csv",
        "unpenalized_rank": "empirical/unpenalized_phase.csv",
        "penalized_repair": "empirical/penalized_phase.csv",
    }
    frames = {kind: pd.read_csv(data_root / relative, float_precision="round_trip") for kind, relative in groups.items()}
    indices = {kind: {coordinate_key(row, kind in {"theory_lasso", "penalized_repair"}): index
                      for index, row in frame.iterrows()} for kind, frame in frames.items()}
    failed = {kind: 0 for kind in groups}
    updated = {kind: 0 for kind in groups}
    diagnostic_dir = output / "diagnostics"
    diagnostic_dir.mkdir(exist_ok=True)
    # Streaming JSONL avoids holding all selected-repetition diagnostics in RAM.
    import gzip
    with gzip.open(diagnostic_dir / "selected_repetitions.jsonl.gz", "wt") as raw_stream, gzip.open(diagnostic_dir / "lambda_and_quadrature_trace.jsonl.gz", "wt") as trace_stream:
        for job in jobs:
            record = json.loads((output / "checkpoints" / (job["job_id"] + ".json")).read_text())
            kind = job["kind"]
            for row in record.get("raw", []):
                raw_stream.write(json.dumps({"job_id": job["job_id"], **row}) + "\n")
            for row in record.get("trace", []):
                trace_stream.write(json.dumps({"job_id": job["job_id"], **row}) + "\n")
            if not record.get("complete"):
                failed[kind] += 1
                continue
            update = record["updates"]
            key = coordinate_key(update, kind in {"theory_lasso", "penalized_repair"})
            index = indices[kind][key]
            frame = frames[kind]
            for field, value in update.items():
                if isinstance(value, (dict, list)):
                    continue
                if field not in frame:
                    frame[field] = None
                frame.at[index, field] = value
            updated[kind] += 1
    for kind, frame in frames.items():
        if kind in {"unpenalized_rank", "penalized_repair"}:
            refresh_empirical_winners(frame)
        relative = groups[kind]
        path = data_root / relative
        if kind == "quadrature":
            # CSV Infinity explicitly means unavailable finite-variance LS theory.
            import numpy as np
            frame.loc[frame.df <= 2, ["alpha2_LS", "alpha_LS"]] = np.inf
        frame.to_csv(path, index=False)
        meta_path = path.with_suffix(".metadata.json")
        meta = json.loads(meta_path.read_text())
        # Keep the input metadata as parent provenance for the updated table.
        parent_fields = ("source_files", "imported_csv_sha256", "risk_round_trip_verified",
                         "risk_values_changed", "solver_diagnostics", "rerun_reasons",
                         "boundary_counts", "original_settings", "lambda_conversion",
                         "manuscript_figure_verification")
        parent = {key: meta.pop(key) for key in parent_fields if key in meta}
        meta.update(complete=failed[kind] == 0, config_fingerprint=fingerprint_value,
                    recomputation={"kind": kind, "updated_rows": updated[kind], "failed_rows": failed[kind], "plan": plan},
                    parent_provenance=parent, current_csv_sha256=file_sha(path),
                    risk_values_changed=updated[kind] > 0,
                    provenance="Table combines newly computed rows with saved baseline values; parent provenance and updated row counts are recorded separately.")
        meta["remaining_limitations"] = [
            "Finite lambda searches do not certify an unrestricted infimum.",
            "Numerical residual and quadrature comparisons are subject to floating-point error.",
        ]
        if failed[kind]:
            meta["remaining_limitations"].append("Failed jobs retain their baseline values; the table is incomplete and cannot be used for figure generation.")
        if kind in {"theory_lasso", "penalized_repair"}:
            meta["signal"] = {"kind": "spike_slab", "nonzero_probability": .2, "nonzero_variance": 5., "force_nonzero": False}
        if kind == "theory_lasso":
            meta["ridge_reuse"] = "Gaussian-design ridge risk depends on the signal second moment, which is one under both priors; saved ridge rows are retained."
        if kind == "penalized_repair":
            meta["selection"] = {"criterion": "mse_median", "evaluation": "same_repetitions", "display_statistic": "mean"}
            meta["rank_solver"] = "Exact rank and penalty proximal operators with primal/dual residual and gap checks."
            meta["forced_zero_repair"] = "All methods retuned at the 760 affected points; unaffected LS/LAD samples and estimates reused."
        if kind in {"unpenalized_rank", "penalized_repair"}:
            meta["solver_diagnostics"] = "Recomputed Rank rows pass primal/dual residual and objective-gap checks; retained LS/LAD use historical diagnostics."
        atomic_json(meta_path, meta)
    atomic_json(data_root / "recomputation_manifest.json", {
        "complete": not any(failed.values()), "config_fingerprint": fingerprint_value,
        "parent_manifest": "parent_recovery_manifest.json", "updated_rows": updated,
        "datasets": {str(path.relative_to(data_root)): {"sha256": file_sha(path)}
                     for path in sorted(data_root.rglob("*.csv"))},
    })
    return failed, updated


def run(plan_path, output, workers=1, dry_run=False):
    plan = json.loads(Path(plan_path).read_text())
    if plan.get("mode") not in {"paper", "smoke"} or workers < 1:
        raise ValueError("Require mode=paper|smoke and workers>=1")
    jobs = make_jobs(plan)
    counts = {kind: sum(job["kind"] == kind for job in jobs) for kind in sorted({job["kind"] for job in jobs})}
    print(json.dumps({"mode": plan["mode"], "jobs": counts, "workers": workers, "output": str(output)}, indent=2), flush=True)
    if dry_run:
        return {"complete": None, "dry_run": True, "jobs": counts}
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    signature = fingerprint(plan)
    meta_path = output / "run_status.json"
    if meta_path.exists() and json.loads(meta_path.read_text()).get("fingerprint") != signature:
        raise ValueError("Source, data, or plan changed: use a fresh run directory")
    state = {"mode": plan["mode"], "fingerprint": signature, "jobs": counts, "complete": False, "status": "running", "workers": workers}
    atomic_json(output / "resolved_plan.json", plan)
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    pending, cached = [], 0
    for job in jobs:
        path = checkpoint_dir / (job["job_id"] + ".json")
        try:
            record = json.loads(path.read_text())
            if record.get("fingerprint") == signature and record.get("complete"):
                cached += 1
                continue
        except (OSError, ValueError):
            pass
        pending.append(job)
    state.update(cached_jobs=cached, completed_jobs=cached, pending_jobs=len(pending))
    atomic_json(meta_path, state)
    print(f"{cached} matching completed checkpoints; {len(pending)} pending", flush=True)
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {pool.submit(worker, job): job for job in pending}
        for number, future in enumerate(as_completed(futures), 1):
            job = futures[future]
            try:
                record = future.result()
            except Exception as exc:
                record = {"complete": False, "error": repr(exc), "kind": job["kind"], "job_id": job["job_id"]}
            record["fingerprint"] = signature
            atomic_json(checkpoint_dir / (job["job_id"] + ".json"), record)
            state.update(finished_jobs=cached + number, last_job=job["job_id"], last_success=record.get("complete"))
            atomic_json(meta_path, state)
            if number % 10 == 0 or not record.get("complete") or number == len(pending):
                print(f"{cached+number}/{len(jobs)}: {job['job_id']} {'complete' if record.get('complete') else 'FAILED'}", flush=True)
    failed_records = []
    for job in jobs:
        record = json.loads((checkpoint_dir / (job["job_id"] + ".json")).read_text())
        if not record.get("complete"):
            failed_records.append({"job_id": job["job_id"], "kind": job["kind"], "error": record.get("error", "numerical checks failed")})
    atomic_json(output / "failed_jobs.json", failed_records)
    if plan["mode"] == "paper":
        failed, updated = write_outputs(plan, output, jobs, signature)
        state.update(failed_by_kind=failed, updated_by_kind=updated)
    else:
        state["scope"] = "Smoke jobs only; no manuscript result table was overwritten or constructed."
    state.update(complete=not failed_records, completed_jobs=len(jobs) - len(failed_records), pending_jobs=0,
                 failed_jobs=len(failed_records), status="complete" if not failed_records else "incomplete")
    atomic_json(meta_path, state)
    if state["complete"] and plan["mode"] == "paper" and plan.get("figures"):
        import matplotlib
        matplotlib.use("Agg")
        from rank_repro.figures_light import generate_light_figures
        from rank_repro.figures_heavy import generate_heavy_figures
        try:
            generate_light_figures(output / "data", output / "figures")
            generate_heavy_figures(output / "data", output / "figures")
            state["figures_complete"] = True
        except Exception as exc:
            state.update(complete=False, status="figure_generation_failed", figures_complete=False,
                         figure_error=f"{type(exc).__name__}: {exc}")
            (output / "figure_error.log").write_text(traceback.format_exc())
        atomic_json(meta_path, state)
    print(json.dumps(state, indent=2), flush=True)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=ROOT / "configs/recompute-plan.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        state = run(args.plan, args.output_dir, args.workers, args.dry_run)
        return 2 if state.get("complete") is False else 0
    except Exception:
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
