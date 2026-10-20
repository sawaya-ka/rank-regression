#!/usr/bin/env python3
"""Import complete recovered historical results without recomputing their risks.

Only the labels of regularization parameters are changed to the paper's
``loss_average + lambda/n * sum(g(beta_j))`` convention. Every converted
column is retained with an ``_archived`` suffix. Metadata distinguish the
historical finite-sample prior and tuning criterion from corrected reruns.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd


SOURCES = {
    "theory/penalized_phase.csv": "t_phase_diagram_results_fast/phase_summary.csv",
    "theory/penalized_are.csv": "rank_penalty_results_t/opt_t_std_0.csv",
    "empirical/unpenalized_phase.csv": "results/t_mse_phase_test/t_mse_phase_summary.csv",
    "empirical/penalized_phase.csv": "results/main_lambda_tuned_parallel_100rep5/penalized_t_mse_phase_summary.csv",
}
PAPER_LAMBDA = "loss_average + (lambda/n) * sum(g(beta_j))"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_record(root: Path, relative: str) -> dict:
    path = root / relative
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Missing or empty recovered source: {relative}")
    return {"path": relative, "bytes": path.stat().st_size, "sha256": sha256(path)}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Missing or empty recovered CSV: {path}")
    return pd.read_csv(path, float_precision="round_trip")


def _grid(frame: pd.DataFrame, grid: dict) -> None:
    keys = list(grid)
    if any(key not in frame for key in keys):
        raise ValueError(f"Missing grid coordinates: {keys}")
    if frame[keys].isna().any().any() or frame.duplicated(keys).any():
        raise ValueError("Grid has missing or duplicate coordinates")
    count = math.prod(len(grid[key]) for key in keys)
    if len(frame) != count:
        raise ValueError(f"Incomplete recovered grid: {len(frame)} rows, expected {count}")
    for key, values in grid.items():
        actual = frame[key].unique()
        if key == "penalty":
            valid = set(actual) == set(values)
        else:
            valid = (len(actual) == len(values) and
                     np.allclose(np.sort(actual), np.sort(values), rtol=1e-12, atol=1e-14))
        if not valid:
            raise ValueError(f"Recovered {key} coordinates differ from expected settings")


def _finite_risks(frame: pd.DataFrame, empirical: bool, methods: list[str]) -> None:
    for method in methods:
        column = f"mse_{method}_mean" if empirical else f"alpha2_{method.lower()}_opt"
        if column not in frame:
            raise ValueError(f"Missing risk column: {column}")
        values = frame[column].to_numpy(dtype=float)
        valid = np.isfinite(values) & (values > 0)
        if not empirical and method == "LS":
            valid |= frame["df"].to_numpy() <= 2
        if not valid.all():
            raise ValueError(f"Invalid recovered risks in {column}")


def _empirical_checks(frame: pd.DataFrame, settings: dict) -> None:
    for key in ("n", "n_rep"):
        if key not in frame or not (frame[key] == settings[key]).all():
            raise ValueError(f"Recovered {key} values do not match metadata")
    if not (frame["p"].to_numpy() > 0).all():
        raise ValueError("Invalid recovered dimension p")
    if not np.allclose(frame["actual_delta"], frame["n"] / frame["p"], rtol=1e-12):
        raise ValueError("Recovered actual_delta is inconsistent with n/p")
    for method in ("LS", "LAD", "rank"):
        key = f"success_rate_{method}"
        if key not in frame or not np.isfinite(frame[key]).all() or not (frame[key] == 1).all():
            raise ValueError(f"Recovered repetitions include failures or missing diagnostics: {key}")
    _finite_risks(frame, True, ["LS", "LAD", "rank"])


def _convert_column(frame: pd.DataFrame, column: str, multiplier) -> None:
    if column not in frame:
        raise ValueError(f"Missing regularization column: {column}")
    frame[column + "_archived"] = frame[column]
    frame[column] = frame[column] * multiplier


def _parse_scalar(value):
    if value in ("True", "False"):
        return value == "True"
    try:
        number = float(value)
        return int(number) if number.is_integer() else number
    except (TypeError, ValueError):
        return value


def _base_metadata(root: Path, relative: str, support: list[str], grid: dict) -> dict:
    return {
        "schema_version": 1,
        "provenance": "recovered historical results; risks unchanged; parameter labels normalized",
        "complete": True,
        "noise": "standard_student_t",
        "grid": grid,
        "lambda_convention": PAPER_LAMBDA,
        "source_files": [source_record(root, relative)] + [source_record(root, value) for value in support],
        "risk_values_changed": False,
        "grid_values_changed": False,
        "solver_diagnostics": "Historical success flags are retained; they are not exact optimality certificates.",
    }


def _phase(root: Path) -> tuple[pd.DataFrame, dict]:
    relative = SOURCES["theory/penalized_phase.csv"]
    frame = _read_csv(root / relative)
    meta_relative = "t_phase_diagram_results_fast/phase_meta.csv"
    meta_frame = _read_csv(root / meta_relative)
    original = {str(k): _parse_scalar(v) for k, v in meta_frame.itertuples(index=False, name=None)}
    grid = {"df": list(range(1, 51)), "delta": np.geomspace(.05, 50, 200).tolist(), "penalty": ["ridge", "lasso"]}
    _grid(frame, grid)
    _finite_risks(frame, False, ["LS", "LAD", "rank"])
    meta = _base_metadata(root, relative, [meta_relative, "compute_t_phase_diagram_fast.py"], grid)
    meta.update({
        "figure": "7",
        "experiment": "theory_penalized",
        "original_settings": original,
        "original_settings_source": "recovered runtime phase_meta.csv",
        "signal": {"kind": "spike_slab", "nonzero_probability": float(original["eps_b"]), "nonzero_variance": float(original["gamma_b_nz2"]), "force_nonzero": False},
        "lambda_conversion": {"LS": "lambda_paper = delta * lambda_archived", "LAD": "identity", "rank": "identity", "risk_effect": "none; same selected solutions and same finite search candidates"},
        "selection": {"criterion": "limiting_risk", "target": "finite coarse-to-refined search", "display_statistic": "limiting_risk"},
        "rerun_reasons": ["Historical prior has nonzero_probability=0.1, whereas Figure 10 uses 0.2; use matching theory for overlays.", "Validate finite search boundaries and numerical quadrature before final release."],
    })
    _convert_column(frame, "lambda_ls_opt", frame["delta"])
    _convert_column(frame, "lambda_lad_opt", 1)
    _convert_column(frame, "lambda_rank_opt", 1)
    _convert_column(frame, "winner_lambda", np.where(frame["winner"].str.lower() == "ls", frame["delta"], 1))
    for method in ("ls", "lad", "rank"):
        multiplier = frame["delta"] if method == "ls" else 1
        frame[f"lambda_{method}_search_min"] = float(original["lambda_min"]) * multiplier
        frame[f"lambda_{method}_search_max"] = float(original["lambda_max"]) * multiplier
    meta["boundary_counts"] = {method: int(frame[f"{method}_opt_on_boundary"].sum()) for method in ("ls", "lad", "rank")}
    return frame, meta


def _are(root: Path) -> tuple[pd.DataFrame, dict]:
    relative = SOURCES["theory/penalized_are.csv"]
    frame = _read_csv(root / relative)
    grid = {"df": list(range(3, 21)), "delta": np.geomspace(.1, 16, 100).tolist(), "penalty": ["ridge", "lasso"]}
    _grid(frame, grid)
    _finite_risks(frame, False, ["LS", "rank"])
    meta = _base_metadata(root, relative, ["ARE_pnl_t.py"], grid)
    cache_root = root / "rank_penalty_results_t/cache_t_std_0_spikeslab_eps0.2_gammabnz25"
    cache_paths = sorted(cache_root.glob("*_opt.csv"))
    if len(cache_paths) != len(frame):
        raise ValueError("Figure 8 requires the complete spike-and-slab cache to corroborate its prior")
    cache = pd.concat([_read_csv(path) for path in cache_paths], ignore_index=True)
    keys = ["df", "penalty", "delta"]
    ordered = frame.sort_values(keys).reset_index(drop=True)
    cache = cache.sort_values(keys).reset_index(drop=True)
    for column in frame:
        if pd.api.types.is_numeric_dtype(frame[column]):
            matches = np.allclose(ordered[column], cache[column], rtol=1e-12, atol=1e-15)
        else:
            matches = ordered[column].equals(cache[column])
        if not matches:
            raise ValueError(f"Figure 8 summary differs from spike-and-slab cache: {column}")
    cache_digest = hashlib.sha256()
    for path in cache_paths:
        cache_digest.update((path.name + " " + sha256(path) + "\n").encode())
    meta["prior_cache_verification"] = {"directory": str(cache_root.relative_to(root)), "files": len(cache_paths), "all_summary_columns_match": True, "relative_tolerance": 1e-12, "absolute_tolerance": 1e-15, "sorted_filename_sha256_manifest_digest": cache_digest.hexdigest(), "note": "Historical aggregation introduces floating-point CSV round-trip differences; risks and parameter values match to the stated tolerance."}
    original = {"DFS": list(range(3, 21)), "DELTA_GRID": grid["delta"], "LAMBDAS_OPT": np.geomspace(.01, 30, 500).tolist(), "STANDARDIZE_T": False, "EPS_B": .2, "GAMMA_B_NZ2": 5., "GH_N": 40, "GRID_SIZE": 1401}
    meta.update({
        "figure": "8",
        "experiment": "theory_penalized_are",
        "original_settings": original,
        "original_settings_source": "recovered ARE_pnl_t.py defaults; runtime settings file absent; df/delta grids verified against CSV",
        "signal": {"kind": "spike_slab", "nonzero_probability": .2, "nonzero_variance": 5., "force_nonzero": False},
        "lambda_conversion": {"LS": "lambda_paper = delta * lambda_archived", "rank": "identity", "risk_effect": "none; same selected solutions and same finite search candidates"},
        "selection": {"criterion": "limiting_risk", "target": "finite 500-point grid", "display_statistic": "limiting_risk"},
        "rerun_reasons": ["The manuscript lambda panels require the corrected LS labels provided here; this is an exact relabeling, not a new experiment.", "The LS ridge optimum at df=3, delta=0.1 reaches the historical upper search bound."],
    })
    _convert_column(frame, "lambda_ls_opt", frame["delta"])
    _convert_column(frame, "lambda_rank_opt", 1)
    for method in ("ls", "rank"):
        multiplier = frame["delta"] if method == "ls" else 1
        frame[f"lambda_{method}_search_min"] = .01 * multiplier
        frame[f"lambda_{method}_search_max"] = 30. * multiplier
    meta["boundary_counts"] = {method: int(frame[f"{method}_opt_on_boundary"].sum()) for method in ("ls", "rank")}
    return frame, meta


def _empirical(root: Path, penalized: bool) -> tuple[pd.DataFrame, dict]:
    name = "penalized" if penalized else "unpenalized"
    relative = SOURCES[f"empirical/{name}_phase.csv"]
    prefix = "penalized_t_mse_phase" if penalized else "t_mse_phase"
    meta_relative = str(Path(relative).with_name(prefix + "_metadata.json"))
    original = json.loads((root / meta_relative).read_text())
    settings = original["args"]
    frame = _read_csv(root / relative)
    grid = {"df": original["df_grid"], "delta": original["delta_grid"]}
    if penalized:
        grid["penalty"] = original["penalties"]
    _grid(frame, grid)
    _empirical_checks(frame, settings)
    source_script = "make_penalized_t_mse_phase_csv_tuned.py" if penalized else "make_t_mse_phase_csv.py"
    support = [meta_relative, source_script]
    if not penalized:
        support.append(str(Path(relative).with_name("t_mse_phase_raw.csv")))
    meta = _base_metadata(root, relative, support, grid)
    meta.update({
        "figure": "10" if penalized else "9",
        "experiment": f"empirical_{name}",
        "original_settings": original,
        "original_settings_source": "complete recovered runtime metadata JSON",
        "n": int(settings["n"]), "n_rep": int(settings["n_rep"]), "base_seed": int(settings["base_seed"]),
        "design": {"distribution": "gaussian", "entry_variance": "1/p" if penalized else "1"},
        "error_metric": "squared_l2_error/p" if penalized else "squared_l2_error",
        "selection": {"criterion": settings["lambda_criterion"] if penalized else "not_applicable", "evaluation": "same_repetitions" if penalized else "monte_carlo_repetitions", "display_statistic": "mean", "independent_test_set": False},
    })
    if penalized:
        meta["signal"] = {"kind": "spike_slab", "nonzero_probability": round(1 - settings["beta_zero_prob"], 12), "nonzero_variance": settings["beta_slab_var"], "force_nonzero": settings["ensure_nonzero_beta"]}
        meta["lambda_conversion"] = {"all_estimators": "lambda_paper = actual_delta * lambda_archived, with actual_delta=n/p", "unchanged_columns": "lambda_eff_* (coefficient lambda_archived/p = lambda_paper/n), counts and source labels", "risk_effect": "none; same selected fitted estimators"}
        meta["replicate_level_results_available"] = False
        meta["rerun_reasons"] = ["ensure_nonzero_beta=True modifies the iid prior when all sampled coordinates are zero; correct simulations must permit the all-zero draw.", "Selecting lambda by median MSE and reporting mean MSE defines a specific oracle comparison; disclose both statistics. The correction plan retains median tuning, so this distinction alone does not require a rerun.", "Inspect historical upper/lower lambda search boundaries rather than treating selected values as unrestricted infima."]
        for method in ("LS", "LAD", "rank"):
            for column in (f"lambda_{method}", f"lambda_{method}_opt", f"lambda_{method}_init"):
                _convert_column(frame, column, frame["actual_delta"])
            if not np.allclose(frame[f"lambda_eff_{method}_opt"], frame[f"lambda_{method}_opt"] / frame["n"], rtol=1e-12, atol=1e-14):
                raise ValueError(f"Converted lambda coefficient mismatch for {method}")
        _convert_column(frame, "winner_lambda", frame["actual_delta"])
        frame["lambda_search_min_paper"] = settings["lambda_min_pos"] * frame["actual_delta"]
        frame["lambda_search_max_paper"] = settings["lambda_max"] * frame["actual_delta"]
        meta["boundary_counts"] = {method: {"lower": int(np.isclose(frame[f"lambda_{method}_opt_archived"], settings["lambda_min_pos"], rtol=1e-10, atol=0).sum()), "upper": int(np.isclose(frame[f"lambda_{method}_opt_archived"], settings["lambda_max"], rtol=1e-10, atol=0).sum())} for method in ("LS", "LAD", "rank")}
    else:
        meta["signal"] = {"kind": "uniform_on_unit_sphere", "construction": "iid N(0,1) vector divided by its Euclidean norm"}
        meta["normalization_conversion"] = "X_paper = X_code/sqrt(p), beta_paper = sqrt(p)*beta_code; squared_l2_error_paper/p = squared_l2_error_code"
        meta["lambda_convention"] = "not_applicable"
        meta["replicate_level_results_available"] = True
        meta["replicate_level_results_included"] = False
        meta["rerun_reasons"] = ["Archived unpenalized rank success flags mean finite outputs; audit actual optimizer convergence before relying on nearly tied winners.", "The archived signal is normalized to norm one, rather than iid Gaussian; unpenalized exact optimizers are shift equivariant, but numerical approximation still needs checking."]
    return frame, meta


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def import_recovered(source_root: Path, output_dir: Path, audit_report: Path | None = None) -> dict:
    source_root, output_dir = Path(source_root).resolve(), Path(output_dir).resolve()
    if source_root == output_dir or source_root in output_dir.parents:
        raise ValueError("Output directory must be outside the recovered source tree")
    # Validate all four inputs before writing any imported table.
    results = {
        "theory/penalized_phase.csv": _phase(source_root),
        "theory/penalized_are.csv": _are(source_root),
        "empirical/unpenalized_phase.csv": _empirical(source_root, False),
        "empirical/penalized_phase.csv": _empirical(source_root, True),
    }
    if audit_report is not None:
        audit = json.loads(Path(audit_report).read_text())
        evidence = audit["figure10_provenance"]
        if evidence["source"] != SOURCES["empirical/penalized_phase.csv"]:
            raise ValueError("Figure 10 audit evidence refers to a different source")
        for panel in evidence["proof"]:
            if (not panel["identical_pixels"] or
                    panel["reference_pixel_sha256"] != panel["manuscript_pixel_sha256"] or
                    panel["classification_matches_mean"] != panel["grid_cells_checked"]):
                raise ValueError("Figure 10 audit does not establish matching heatmap classifications")
        results["empirical/penalized_phase.csv"][1]["manuscript_figure_verification"] = evidence
        results["empirical/penalized_phase.csv"][1]["manuscript_figure_verification"]["audit_report_sha256"] = sha256(Path(audit_report))
        results["empirical/penalized_phase.csv"][1]["manuscript_figure_verification"]["scope"] = "All 5000 winner classifications match; this verifies plotted classes, not unique identity of all underlying numerical risks."
    manifest = {"schema_version": 1, "status": "complete_historical_tables_imported; scientific corrections still required", "source_tree_modified": False, "risks_recomputed": False, "lambda_convention": PAPER_LAMBDA, "datasets": {}}
    for relative, (frame, metadata) in results.items():
        target = output_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".tmp")
        frame.to_csv(temporary, index=False, float_format="%.17g")
        # A round-trip check ensures that no archived risk was altered by export.
        round_trip = _read_csv(temporary)
        risk_columns = [column for column in frame if column.startswith(("alpha2_", "mse_", "sqerr_"))]
        for column in risk_columns:
            if not np.array_equal(frame[column].to_numpy(), round_trip[column].to_numpy(), equal_nan=True):
                raise ValueError(f"CSV round-trip changed historical values in {column}")
        os.replace(temporary, target)
        metadata["imported_csv_sha256"] = sha256(target)
        metadata["risk_round_trip_verified"] = True
        _write_json(target.with_suffix(".metadata.json"), metadata)
        manifest["datasets"][relative] = {"rows": len(frame), "sha256": sha256(target), "source_files": metadata["source_files"], "figure": metadata["figure"], "boundary_counts": metadata.get("boundary_counts", {}), "complete_grid": True, "risk_values_unchanged": True, "metadata": str(Path(relative).with_suffix(".metadata.json"))}
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "recovery_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path, help="Extracted historical Rank directory")
    parser.add_argument("--output-dir", type=Path, default=Path("data/recovered"))
    parser.add_argument("--audit-report", type=Path, help="Optional audit_recovered.json with manuscript image verification")
    args = parser.parse_args()
    try:
        manifest = import_recovered(args.source_root, args.output_dir, args.audit_report)
    except (ValueError, KeyError, OSError, pd.errors.ParserError) as exc:
        parser.exit(1, f"Recovery import failed: {exc}\n")
    for relative, item in manifest["datasets"].items():
        print(f"Imported {item['rows']:5d} rows: {relative}")
    print("Historical risk values preserved; see recovery_manifest.json and dataset metadata for required corrections.")


if __name__ == "__main__":
    main()
