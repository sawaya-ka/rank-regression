#!/usr/bin/env python3
"""Export the small result files needed to recover manuscript Figures 7–10.

Run on the original CPU VM, using only the Python standard library::

    python export_results.py --source-root /path/to/Rank2 --output recovered-results.tar.gz

Missing/incomplete inputs produce an archive with an honest manifest and a
nonzero exit status. ``--allow-partial`` changes the exit status only; it never
changes an incomplete file's classification. Caches, environments, raw samples,
and per-lambda full curves are deliberately excluded.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path


def geomspace(low: float, high: float, count: int) -> list[float]:
    return [math.exp(math.log(low) + i * math.log(high / low) / (count - 1)) for i in range(count)]


def specifications(figure10_directory: str) -> list[dict]:
    return [
        {"figure": "7", "path": "t_phase_diagram_results_fast/phase_summary.csv",
         "metadata": "t_phase_diagram_results_fast/phase_meta.csv", "metadata_required": True,
         "expected_rows": 20000, "df": list(range(1, 51)), "delta": geomspace(.05, 50, 200),
         "penalty": ["ridge", "lasso"], "risks": ["alpha2_ls_opt", "alpha2_lad_opt", "alpha2_rank_opt"]},
        {"figure": "8", "path": "rank_penalty_results_t/opt_t_std_0.csv",
         "metadata": "rank_penalty_results_t/metadata.json", "metadata_required": False,
         "expected_rows": 3600, "df": list(range(3, 21)), "delta": geomspace(.1, 16, 100),
         "penalty": ["ridge", "lasso"], "risks": ["alpha2_ls_opt", "alpha2_rank_opt"]},
        {"figure": "9", "path": "results/t_mse_phase_test/t_mse_phase_summary.csv",
         "metadata": "results/t_mse_phase_test/t_mse_phase_metadata.json", "metadata_required": True,
         "expected_rows": 2500, "df": geomspace(1, 50, 50), "delta": geomspace(1.001, 50, 50),
         "risks": ["mse_LS_mean", "mse_LAD_mean", "mse_rank_mean"]},
        {"figure": "10", "path": f"{figure10_directory}/penalized_t_mse_phase_summary.csv",
         "metadata": f"{figure10_directory}/penalized_t_mse_phase_metadata.json", "metadata_required": True,
         "expected_rows": 5000, "df": geomspace(1, 50, 50), "delta": geomspace(.5, 50, 50),
         "penalty": ["ridge", "lasso"], "risks": ["mse_LS_mean", "mse_LAD_mean", "mse_rank_mean"]},
    ]


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_file(root: Path, relative: str) -> tuple[dict, Path | None]:
    path = root / relative
    entry = {"path": relative, "status": "missing", "size_bytes": None, "sha256": None}
    if not path.is_file():
        return entry, None
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        entry.update(status="outside_source_root", issue="Symlink points outside source-root; not exported.")
        return entry, None
    entry.update(size_bytes=path.stat().st_size, sha256=checksum(path))
    entry["status"] = "present" if entry["size_bytes"] else "zero_byte"
    if path.suffix in (".json", ".ipynb") and entry["size_bytes"]:
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
            if path.suffix == ".ipynb" and not isinstance(document.get("cells"), list):
                raise ValueError("Notebook has no cells array")
        except (ValueError, UnicodeError) as exc:
            entry.update(status="invalid_json", issue=str(exc))
    return entry, path


def read_metadata(path: Path | None) -> dict:
    if path is None or path.stat().st_size == 0:
        return {}
    try:
        if path.suffix == ".json":
            value = json.loads(path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
        with path.open(newline="", encoding="utf-8-sig") as stream:
            rows = list(csv.reader(stream, strict=True))
        return {row[0]: row[1] for row in rows[1:] if len(row) == 2}
    except (ValueError, csv.Error, UnicodeError):
        return {}


def configured_grid(metadata: dict) -> dict:
    grid = metadata.get("grid", {})
    if not grid:
        grid = {"df": metadata.get("df_grid"), "delta": metadata.get("delta_grid"),
                "penalty": metadata.get("penalties")}
    if grid.get("df") and grid.get("delta"):
        return {key: value for key, value in grid.items() if key in ("df", "delta", "penalty") and value}
    # The original theory runner wrote a two-column CSV, not JSON.
    try:
        if all(key in metadata for key in ("df_min", "df_max", "df_step", "delta_min", "delta_max", "n_delta")):
            low, high, step = (float(metadata[key]) for key in ("df_min", "df_max", "df_step"))
            count = int(round((high - low) / step)) + 1
            return {"df": [low + step * i for i in range(count)],
                    "delta": geomspace(float(metadata["delta_min"]), float(metadata["delta_max"]), int(metadata["n_delta"])),
                    "penalty": str(metadata.get("penalties", "ridge,lasso")).split(",")}
    except (ValueError, ZeroDivisionError):
        return {}
    return {}


def equal_coordinates(actual: set, expected: list, numeric: bool) -> bool:
    if len(actual) != len(expected):
        return False
    if not numeric:
        return actual == set(expected)
    try:
        return all(math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-12)
                   for a, b in zip(sorted(actual), sorted(float(value) for value in expected)))
    except (TypeError, ValueError):
        return False


def inspect_result(path: Path, specification: dict, metadata: dict) -> dict:
    keys = ["df", "delta"] + (["penalty"] if "penalty" in specification else [])
    expected = {key: specification[key] for key in keys}
    configured = configured_grid(metadata)
    seen: set = set()
    coordinates = {key: set() for key in keys}
    issues: list[str] = []
    rows = duplicates = malformed = invalid_risks = 0
    header: list[str] = []
    final_width = None
    try:
        with path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.reader(stream, strict=True)
            header = next(reader, [])
            required = keys + specification["risks"]
            if len(set(header)) != len(header):
                issues.append("Duplicate column names")
            missing = set(required) - set(header)
            if missing:
                issues.append(f"Missing columns: {sorted(missing)}")
            for row in reader:
                if not row:
                    continue
                rows += 1
                final_width = len(row)
                if len(row) != len(header):
                    malformed += 1
                    continue
                record = dict(zip(header, row))
                try:
                    coordinate = tuple(record[key] if key == "penalty" else float(record[key]) for key in keys)
                    if any(not math.isfinite(value) or value <= 0 for value in coordinate if isinstance(value, float)):
                        raise ValueError("Nonfinite/nonpositive coordinate")
                except (KeyError, ValueError):
                    malformed += 1
                    continue
                if coordinate in seen:
                    duplicates += 1
                seen.add(coordinate)
                for key, value in zip(keys, coordinate):
                    coordinates[key].add(value)
                for name in specification["risks"]:
                    if name == "alpha2_ls_opt" and float(record["df"]) <= 2:
                        continue  # Undefined finite-variance formula, not a failed solve.
                    try:
                        value = float(record[name])
                        if not math.isfinite(value) or value < 0:
                            raise ValueError("invalid risk")
                    except (ValueError, KeyError):
                        invalid_risks += 1
                for name, raw in record.items():
                    if name.startswith("success_rate_"):
                        try:
                            if not math.isclose(float(raw), 1):
                                invalid_risks += 1
                        except ValueError:
                            invalid_risks += 1
    except (csv.Error, UnicodeError) as exc:
        issues.append(f"CSV parsing failed: {exc}")
    if malformed:
        issues.append(f"{malformed} malformed records; final row has {final_width} fields for {len(header)} columns")
    if duplicates:
        issues.append(f"{duplicates} duplicate grid coordinates")
    if invalid_risks:
        issues.append(f"{invalid_risks} invalid/failed estimator values or availability diagnostics")
    paper_match = rows == specification["expected_rows"] and all(
        equal_coordinates(coordinates[key], expected[key], key != "penalty") for key in keys
    )
    configured_expected = math.prod(len(configured[key]) for key in keys) if all(key in configured for key in keys) else None
    configured_match = (rows == configured_expected and all(
        equal_coordinates(coordinates[key], configured[key], key != "penalty") for key in keys
    )) if configured_expected is not None else None
    if not paper_match:
        issues.append(f"Grid differs from the audited manuscript profile ({specification['expected_rows']} expected rows)")
    if configured_match is False:
        issues.append(f"Grid incomplete/inconsistent with supplied metadata ({configured_expected} expected rows)")
    if metadata.get("complete") is False:
        issues.append("Generating metadata reports incomplete results")
    with path.open("rb") as stream:
        stream.seek(-1, 2)
        terminal_newline = stream.read(1) in (b"\n", b"\r")
    return {"row_count": rows, "column_count": len(header), "columns": header,
            "expected_manuscript_rows": specification["expected_rows"],
            "expected_rows_from_metadata": configured_expected,
            "coordinate_counts": {key: len(value) for key, value in coordinates.items()},
            "manuscript_grid_match": paper_match, "metadata_grid_match": configured_match,
            "malformed_records": malformed, "duplicate_coordinates": duplicates,
            "terminal_newline": terminal_newline,
            "last_record_width": final_width,
            "verified_complete_for_manuscript_profile": not issues and paper_match,
            "issues": issues,
            "limitation": "Grid and value integrity checks do not certify optimizer convergence or identify which overwritten PDF used these data."}


def export_results(root: Path, output: Path, *, include_notebooks: bool = False,
                   figure10_directory: str = "results/main_lambda_tuned_parallel2") -> dict:
    root, output = Path(root), Path(output)
    if not root.is_dir():
        raise ValueError(f"Source directory does not exist: {root}")
    if output.exists():
        raise ValueError(f"Output already exists; choose a new filename: {output}")
    if Path(figure10_directory).is_absolute() or ".." in Path(figure10_directory).parts:
        raise ValueError("--figure10-directory must be relative to source-root, without '..'")
    entries: dict[str, dict] = {}
    selected: dict[str, Path] = {}
    figures = []
    for spec in specifications(figure10_directory):
        meta_entry, meta_path = inspect_file(root, spec["metadata"])
        meta_entry["required"] = spec["metadata_required"]
        metadata = read_metadata(meta_path)
        if meta_entry["status"] == "present" and not metadata:
            meta_entry.update(status="invalid_metadata", issue="Metadata must contain a nonempty mapping of settings.")
        if spec["metadata_required"] and meta_entry["status"] == "present" and not configured_grid(metadata):
            meta_entry.update(status="invalid_metadata", issue="Expected df/delta grid settings are missing or invalid.")
        entries[spec["metadata"]] = meta_entry
        if meta_path is not None:
            selected[spec["metadata"]] = meta_path
        result_entry, result_path = inspect_file(root, spec["path"])
        result_entry["required"] = True
        if result_path is not None:
            selected[spec["path"]] = result_path
            if result_entry["status"] == "present":
                result_entry["csv_checks"] = inspect_result(result_path, spec, metadata)
                if not result_entry["csv_checks"]["verified_complete_for_manuscript_profile"]:
                    result_entry["status"] = "incomplete_or_inconsistent_csv"
        entries[spec["path"]] = result_entry
        metadata_ok = not spec["metadata_required"] or meta_entry["status"] == "present"
        figures.append({"figure": spec["figure"], "data": spec["path"], "metadata": spec["metadata"],
                        "ready_for_integrity_review": result_entry["status"] == "present" and metadata_ok,
                        "note": "Ready means structural checks passed; provenance and signal/tuning settings still need review."})
    provenance = ["ARE_pnl_t.py", "compute_t_phase_diagram_fast.py", "compute_t_phase_diagram_fast_lambda_cont.py",
                  "make_t_mse_phase_csv.py", "make_penalized_t_mse_phase_csv_tuned.py"]
    if include_notebooks:
        provenance.extend(["vis.ipynb", "vis_emp.ipynb", "vis_light.ipynb"])
    for relative in provenance:
        entry, path = inspect_file(root, relative)
        entry["required"] = include_notebooks and relative.endswith(".ipynb")
        entries[relative] = entry
        if path is not None:
            selected[relative] = path
    manifest = {"schema_version": 1, "exported_at_utc": datetime.now(timezone.utc).isoformat(),
                "source_directory_name": root.name, "figures": figures,
                "all_required_structural_checks_passed": all(item["ready_for_integrity_review"] for item in figures) and all(
                    entry["status"] == "present" for entry in entries.values() if entry.get("required")),
                "files": list(entries.values()),
                "excluded": ["virtual environments", "cache directories", "raw repetitions", "full lambda curves", "existing plot PDFs"],
                "caveat": "Parsing a CSV is insufficient: counts, expected coordinates, duplicates, required values and metadata are checked. No rows are repaired or dropped."}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w:gz") as archive:
        for relative, path in sorted(selected.items()):
            archive.add(path, arcname=f"Rank2/{relative}", recursive=False)
        contents = (json.dumps(manifest, indent=2, allow_nan=False) + "\n").encode()
        info = tarfile.TarInfo("manifest.json")
        info.size = len(contents)
        info.mode = 0o644
        archive.addfile(info, io.BytesIO(contents))
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-notebooks", action="store_true", help="Include and validate the three visualization notebooks as provenance.")
    parser.add_argument("--figure10-directory", default="results/main_lambda_tuned_parallel2",
                        help="Relative directory of the confirmed Figure 10 run; never silently selects another run.")
    parser.add_argument("--allow-partial", action="store_true", help="Exit zero after exporting incomplete evidence; manifest remains explicitly incomplete.")
    args = parser.parse_args(argv)
    try:
        manifest = export_results(args.source_root, args.output, include_notebooks=args.include_notebooks,
                                  figure10_directory=args.figure10_directory)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(f"Wrote {args.output}")
    for item in manifest["figures"]:
        state = "structural checks passed" if item["ready_for_integrity_review"] else "MISSING / INCOMPLETE / INCONSISTENT"
        print(f"Figure {item['figure']}: {state}")
    for entry in manifest["files"]:
        if entry["status"] != "present":
            print(f"  {entry['status']}: {entry['path']}")
    return 0 if manifest["all_required_structural_checks_passed"] or args.allow_partial else 1


if __name__ == "__main__":
    sys.exit(main())
