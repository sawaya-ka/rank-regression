# Reproduction guide

Run commands from the repository root after following the [installation instructions](../README.md#installation).

## Draw figures from the supplied tables

```bash
rank-repro audit --strict
rank-repro figures --group all --output-dir build/figures
```

The default data directory is `data/paper/`. Figures 1 and 7–10 use these tables; the remaining curves use deterministic scalar calculations. The animation renderer uses its separate stored risk grid. This workflow does not run finite-sample simulations.

Pass `--data-dir PATH` to draw a complete compatible set of newly computed tables. The renderer checks grid completeness and the noise/signal assumptions before overlaying theoretical boundaries on empirical results. See the [figure map](figure-map.md) for inputs and definitions.

## Compute numerical results

```bash
python3 tools/run_recompute_plan.py --output-dir results --dry-run
python3 tools/run_recompute_plan.py --workers 8 --output-dir results
```

The plan in `configs/recompute-plan.json` computes 865 selected Student-t quadrature points, 10,000 lasso theory points, 2,500 unpenalized Rank points, and 5,000 penalized empirical points. Ridge theory, the Figure 8 optimum table, and selected LS/LAD empirical summaries are supplied as inputs in `data/` and `data/recovered/`.

The output directory contains checkpoints, assembled tables, diagnostics, and figures. Running the same command again reuses successful checkpoints and retries failed ones. A source/configuration fingerprint prevents reuse after incompatible changes; use a new output directory when changing the implementation, settings, or input data.

For other designs or grids, use the standalone commands with explicit configurations:

```bash
rank-repro compute unpenalized_theory --config configs/unpenalized_theory.smoke.json --dry-run
rank-repro compute unpenalized_theory --config configs/unpenalized_theory.smoke.json --output-dir build/theory-smoke
```

Corresponding configurations are available for `theory_penalized`, `theory_penalized_are`, `empirical_unpenalized`, and `empirical_penalized`. The `*.paper.json` files specify standalone experiments with explicit designs and grids.

A small end-to-end check of the plan is:

```bash
python3 tools/run_recompute_plan.py --plan configs/recompute-smoke.json --workers 2 --output-dir build/smoke
```

## Numerical methods and limitations

**Quadrature.** Student-t risk calculations use deterministic quadrature. At 865 selected sensitive or nearly tied Figure 1 grid points, successive orders up to 4,800 are compared at relative tolerance 5 × 10⁻⁴. The remaining 9,135 points use 100 nodes. These comparisons diagnose numerical stability but do not provide a uniform error bound over the full grid. Display interpolation does not add computed grid points.

**Rank optimization.** Finite-sample Rank fits use convex ADMM with exact proximal operations, primal/dual residual checks, and an objective-gap check. The computation plan applies a fixed-rho fallback to unpenalized maximum-iteration failures, using the same tolerance and iteration cap. Small independent linear/quadratic-program comparisons are included in the tests. Accepted fits are numerical solutions to the recorded tolerances.

**Penalty selection.** The theoretical calculations use finite penalty searches. Figure 7 starts from 200 candidates and refines eligible minima. Figure 8 uses the supplied optimum table; one LS ridge optimum at Student-t df = 3 and δ = 0.1 reaches the search upper boundary. These minima do not certify unrestricted infima. Very close risks in a phase diagram should not be interpreted as resolved strict superiority.

**Monte Carlo summaries.** Figure 9 uses n = 1,000 and ten repetitions per point. Figure 10 uses n = 500 and 50 repetitions per point, selecting penalties by median MSE and displaying mean MSE at the selected penalty on the same repetitions. The latter measures oracle tuning rather than cross-validation performance. The raw data contain selected repetitions and candidate summaries; they do not contain every fitted coefficient vector. Per-table metadata specify the available repetition-level records.

**Signal and normalization.** Penalized simulations use iid spike-and-slab coefficients with zero probability 0.8 and nonzero variance 5, including possible all-zero vectors. The unpenalized simulations use design variance one and a coefficient vector uniform on the unit sphere. In paper coordinates, `X_paper = X_code / sqrt(p)` and `beta_paper = sqrt(p) * beta_code`, so the code's squared coefficient error equals the paper's normalized squared coefficient error. Unpenalized coefficient-error distributions are invariant under shifts of the true coefficient vector.

## Data provenance and verification

The figure tables are in `data/paper/`; their metadata describe the grid, noise and signal laws, penalty normalization, solver settings, and raw-record availability. Source snapshots and checkpoint hashes are in `provenance/`. The generating source and configuration are identified by their fingerprints.

Download the [v0.3.0 raw result archive](https://github.com/sawaya-ka/rank-regression/releases/download/v0.3.0/rank-results-v0.3.0.tar.gz), verify its SHA-256 against `67ffa4c7bdf1468885cdd8ebfb18f077bf67f3b6860fb8a757c1fc95946f1f82`, and extract it into `build/raw-results/`. Then run:

```bash
python3 tools/audit_archive.py --extracted build/raw-results --output-dir build/archive-audit
```

This read-only check uses the Python standard library to verify checkpoint hashes, numerical acceptance gates, and checkpoint-to-CSV transfers. It requires the complete raw archive, including its `results/` and `source/` directories. A failed check returns a nonzero status and writes details under `build/archive-audit/`.

## Cloud execution

The optional launcher requires an authenticated Google Cloud CLI and Python 3.9 or later on the local machine. The selected VM needs Python 3.10–3.12, `flock`, `setsid`, persistent storage, and sufficient CPU/RAM. Package installation requires access to PyPI.

```bash
bash run_gcp.sh --project YOUR_PROJECT_ID --workers 100
```

The launcher asks which existing VM to use. It does not create VMs or change IAM permissions. Each worker uses one BLAS thread. Choose worker count to fit both memory and CPU; high-order quadrature can use substantial memory.

Computation runs independently of the SSH connection. In direct mode, stopping the VM requires successful computation and a locally downloaded archive with a verified SHA-256. Computation or pipeline errors leave the VM running. If the client sleeps or disconnects, reconnect from the same project directory with:

```bash
bash run_gcp.sh --resume RUN_ID
```

The local `.gcp-runs/` directory stores the information needed to reconnect. Add `--retry-failed` to retry failed jobs using the unchanged source snapshot. An optional existing GCS bucket can provide verified remote backup before stopping; see `bash run_gcp.sh --help` for `--gcs-prefix`.

## Environment

The production run used Python 3.10.12, NumPy 2.2.6, SciPy 1.15.3, pandas 2.2.3, scikit-learn 1.7.2, and Matplotlib 3.10.8. `requirements-tested.txt` pins these dependencies. Floating-point results can vary slightly across BLAS libraries, operating systems, and Python versions; fonts and PDF metadata can also change rendering bytes.

The production computation took approximately 27.5 hours on a Google Cloud `n2d-highcpu-128` VM. This is a hardware-specific observation, not a runtime bound.
