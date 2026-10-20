"""Small command-line interface; expensive experiments require an explicit config."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

EXPERIMENTS = (
    "unpenalized_theory", "theory_penalized", "theory_penalized_are",
    "empirical_unpenalized", "empirical_penalized",
)


def audit(data_dir: Path, strict: bool = False) -> int:
    """Verify bundled file integrity and report availability, without filling gaps."""
    from .figures_heavy import FILES
    provenance = data_dir / "provenance.json"
    bad = False
    if provenance.is_file():
        manifest = json.loads(provenance.read_text())
        for item in manifest["datasets"]:
            path = data_dir / item["file"]
            valid = path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
            print(f"{'OK' if valid else 'CHANGED/MISSING'}  {path}")
            bad |= not valid
    else:
        print("No archival provenance manifest in this data directory.")
    light = False
    try:
        from .student import load_interpolated_risks
        load_interpolated_risks(data_dir / "unpenalized_student_t.csv", 8, 8)
        light = True
        print("Figures 1–6, S1–S2: phase grid validation passed")
    except (OSError, ValueError) as exc:
        print(f"Light phase grid unavailable: {exc}")
    print("Figure S3: new illustrative reconstruction; original figure source was not supplied.")
    from .figures_heavy import _load, _risk_columns, _check_empirical
    full = light
    for number, (relative, _) in FILES.items():
        try:
            frame, meta = _load(data_dir / relative, penalized=number != "9")
            _risk_columns(frame, number in {"9", "10"}, include_lad=number != "8")
            if number in {"9", "10"}:
                _check_empirical(frame, meta, penalized=number == "10")
            print(f"Figure {number}: table and metadata checks passed")
        except (ValueError, OSError) as exc:
            full = False
            print(f"Figure {number}: UNAVAILABLE — {exc}")
    print("Archive integrity and numerical accuracy are separate checks; see docs/reproduction.md.")
    return int(bad or (strict and not full))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rank-repro", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plots = sub.add_parser("figures", help="Draw figures from available complete inputs")
    plots.add_argument("--group", choices=("light", "heavy", "all"), default="light")
    plots.add_argument("--data-dir", type=Path, default=Path("data/paper"))
    plots.add_argument("--output-dir", type=Path, default=Path("figures"))
    plots.add_argument("--figures", nargs="+", choices=("7", "8", "9", "10"), help="Select heavy figures; use --group heavy")
    compute = sub.add_parser("compute", help="Run or resume an explicitly configured experiment")
    compute.add_argument("experiment", choices=EXPERIMENTS)
    compute.add_argument("--config", type=Path, required=True)
    compute.add_argument("--output-dir", type=Path, default=Path("data/runs"))
    compute.add_argument("--workers", type=int, default=1)
    compute.add_argument("--dry-run", action="store_true", help="Print job count and output paths without computing")
    check = sub.add_parser("audit", help="Report data integrity and figure availability")
    check.add_argument("--data-dir", type=Path, default=Path("data/paper"))
    check.add_argument("--strict", action="store_true", help="Fail if any original numerical figure lacks complete inputs")
    args = parser.parse_args(argv)
    try:
        if args.command == "audit":
            return audit(args.data_dir, args.strict)
        if args.command == "compute":
            if args.experiment == "unpenalized_theory":
                from .light_experiments import run_light_experiment
                result = run_light_experiment(args.config, args.output_dir, args.workers, args.dry_run)
            else:
                from .experiments import run_experiment
                result = run_experiment(args.experiment, args.config, args.output_dir, args.workers, args.dry_run)
            print(json.dumps(result, indent=2))
            return 2 if result.get("complete") is False else 0
        if args.figures and args.group != "heavy":
            parser.error("--figures is supported with --group heavy")
        # Select a headless backend before importing any plotting modules.
        import matplotlib
        matplotlib.use("Agg")
        paths = []
        if args.group in {"light", "all"}:
            from .figures_light import generate_light_figures
            from .figures_schematic import generate_schematic
            print("Figure 1 reads the selected data directory; numerical-check scope is recorded in its metadata.")
            paths.extend(generate_light_figures(args.data_dir, args.output_dir))
            paths.extend(generate_schematic(args.output_dir))
            print("S3 is an explicitly new Gaussian illustration, not the missing original figure source.")
        if args.group in {"heavy", "all"}:
            from .figures_heavy import generate_heavy_figures
            paths.extend(generate_heavy_figures(args.data_dir, args.output_dir, args.figures))
        for path in paths:
            print(path)
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"rank-repro: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
