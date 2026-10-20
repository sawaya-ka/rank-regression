# High-dimensional rank regression

Reproducibility code for **High-Dimensional Rank Regression: Exact Risk and Effective-Score Phase Transition**, by K. Sawaya, Y. Uematsu, and M. Imaizumi.

The project compares least squares (LS), least absolute deviations (LAD), and rank regression under Gaussian designs. It includes scalar risk solvers, finite-sample optimization, the paper's figure inputs, and the computational record of the corrected VM run.

![LS, LAD, and Rank scaled risk as dimensionality increases](assets/animations/fig1a_scaled_risk.gif)

**From the classical limit to high dimensions.** The horizontal axis is the Student-t degrees of freedom; the vertical axis is **δ α²(δ)**, where δ = n/p. The initial δ = ∞ frame shows the classical limiting constants. Subsequent frames decrease δ from 100 to 1.01 on a logarithmic grid rounded to three significant digits. The vertical axis rescales between frames. LS is omitted when its risk is infinite. [Animation data, checks, and generation code](docs/animation.md).

## Reproduce the figures

Use Python 3.10–3.12 from this repository's root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-tested.txt
python -m pip install -e '.[animation]'
rank-repro audit --strict
rank-repro figures --group all --output-dir build/figures
python animation/render_gifs.py --output-dir build/animation
```

On Windows, activate with `.venv\Scripts\activate`. The plotting command reads the corrected tables in `data/paper/`, recomputes deterministic scalar curves, and writes PDFs, PNGs, and numerical diagnostics. It does not launch Monte Carlo simulations or contact a VM. Rendering all static figures takes several minutes on a CPU; the committed outputs are in [`figures/`](figures/).

For an interactive entry point, install `python -m pip install -e '.[notebook]'` and open [`notebooks/vis.ipynb`](notebooks/vis.ipynb).

## Numerical conventions

- δ = n/p. Student-t errors have scale one, without variance standardization. The unpenalized LS risk is infinite for degrees of freedom at most two.
- ARE(A:B) = risk(B) / risk(A); values above one favor A.
- The penalized objective is `loss_average + (lambda/n) * sum(g(beta_j))`, with `g(b) = b²/2` for ridge and `g(b) = |b|` for lasso. Archived penalty labels are converted without altering their fitted risks.
- Corrected penalized simulations use iid `0.8 delta_0 + 0.2 N(0,5)` coefficients. An all-zero vector is allowed and does not trigger a forced nonzero draw.
- Figure 10 uses **median MSE to choose a penalty**, then displays **mean MSE at that penalty**, using the same repetitions. This is an oracle Monte Carlo comparison, not cross-validation.
- Figure 1's checks cover 865 selected cells, not every cell. Finite penalty searches and close risk ties remain numerical limitations. Figure 8 retains its historical search interval, including one LS boundary optimum.

## Recompute the experiments

The correction plan uses historical inputs under `data/` and `data/recovered/`; corrected publication inputs under `data/paper/` are separate. Fresh runs include the successful quadrature extensions and fixed-rho fallback. Their source fingerprints differ from the archived run; the original numerical source is preserved for provenance.

```bash
# Inspect the workload without computing or contacting the cloud.
python tools/run_recompute_plan.py --output-dir recompute-results --dry-run

# Run locally (choose workers for your CPU and memory).
python tools/run_recompute_plan.py --workers 8 --output-dir recompute-results

# Or use an existing GCP VM, retrieve verified results, then stop it.
bash run_gcp.sh --project YOUR_PROJECT_ID --workers 100
```

The cloud launcher asks which VM to use. **On any computation or pipeline error it keeps the VM running.** In direct mode, successful computation and a checksum-verified local download are both required before stopping; reconnect with `bash run_gcp.sh --resume RUN_ID` if the client was offline. [Execution and resume guide](docs/one-command-ja.md).

The original production computation took approximately 27.5 hours on its VM; the subsequent two-worker repair took approximately 28 minutes. These are recorded runs, not a guarantee for other hardware. [Reproduction details](docs/reproduction.md).

## Files and provenance

| Location | Purpose |
|---|---|
| `src/rank_repro/` | Shared risk equations, solvers, simulations, plotting |
| `data/paper/` | Corrected publication tables and annotated metadata |
| `data/recovered/`, root `data/*.csv` | Historical inputs retained for the correction plan |
| `figures/` | Static outputs, Gaussian residuals, Student-t curve values |
| `animation/`, `assets/animations/` | GIF generator, grid, checks, README animation |
| `configs/` | Correction plan, smoke plan, standalone experiment configurations |
| `provenance/` | Archived fingerprints, metadata, checkpoint hashes, original numerical source |
| `tools/` | Local/cloud execution, export, GitHub release preparation |
| `tests/` | Numerical identities, independent optimization comparisons, workflow guards |

Large raw checkpoints and per-repetition records belong in the versioned **GitHub Release asset**, not Git history. The publication tool builds that asset from the verified received archive and omits operational cloud logs and launch details. [Publish the repository and raw-data release](docs/publishing.md).

```bash
python -m unittest discover -s tests -v
```

No software license has been selected on the author's behalf. `CITATION.cff` identifies the software and manuscript; add the publication DOI when available.
