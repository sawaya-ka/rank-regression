# Figure and manuscript alignment audit

This audit reads the restored VM result tables and figures, the v0.2.2 source, and the supplied manuscript.
It does not modify those inputs and does not rerun the experiments.
The manuscript title is **High-Dimensional Rank Regression: Exact Risk and Effective-Score Phase Transition**, by **K. Sawaya, Y. Uematsu, and M. Imaizumi**.
Its apparent Biometrika volume/date header is a manuscript template, not evidence of publication; do not carry that header into bibliographic metadata.

## Current figure map

Paths below are relative to `data/paper/`, `figures/`, or `src/rank_repro/`, as indicated. The original received archive uses `results/data/` and `results/figures/`.
Every received figure listed here has both PDF and PNG versions.

| Paper figure | Output basename | Data or construction | Generator | Status and interpretation |
|---|---|---|---|---|
| 1 | `fig01_student_efficiency` | `unpenalized_student_t.csv`; classical constants from exact beta-function formulas | `figures_light.make_figure` | Complete 100 × 100 grid; 865 numerical checks/replacements, 9,135 retained archival values |
| 2 | `fig02_gaussian_efficiency` | Gaussian scalar risk equations, 701 distinct delta values including sqrt(2) | `figures_light._gaussian_efficiency` | Recomputed Rank:LS **and LAD:LS** curves; manuscript caption describes only Rank |
| 3 | `fig03_student_rank_vs_ls` | Generated `figures/student_risk_curves.csv`, df 3–10, delta 1.01–10 | `figures_light._student_figures` | Ratio LS risk / Rank risk; values above one favor Rank |
| 4 | `fig04_isotonic_projection` | Standard Gaussian effective residual, c = 1/sqrt(2), Z variance 1/2, tau = 0.45 and 0.90 | `figures_light._isotonic_figure` | Illustrative proximal parameters; not self-consistent risk parameters |
| 5 | `fig05_effective_scores` | Each method's self-consistent Gaussian parameters for delta = 1.1, …, 1.8 | `figures_light._effective_scores` | Recomputed score curves; `gaussian_diagnostics.json` contains equation residual checks |
| 6 | `fig06_proximal_atom` | Gaussian zero-atom probabilities at self-consistent parameters | `figures_light._atom_figure` | Rank, LAD = 1/delta, LS = 0; caption already states evaluation at alpha*(delta), kappa*(delta) |
| 7 | `figure_07_penalized_phase` | `theory/penalized_phase.csv`, 20,000 rows | `figures_heavy.generate_heavy_figures` | Lasso rerun with 0.8 delta0 + 0.2 N(0,5); ridge reused because its signal second moment remains one |
| 8 | `figure_08_penalized_are` | `theory/penalized_are.csv`, 3,600 rows | `figures_heavy._figure8` | Historical optimum risks retained; LS lambda labels corrected; lower two panels show selected lambdas |
| 9 | `figure_09_empirical_unpenalized` | `empirical/unpenalized_phase.csv`, 2,500 points, n = 1,000, 10 repetitions; Figure 1 table supplies theoretical overlay | `figures_heavy.generate_heavy_figures` | Rank recomputed throughout; LS/LAD retained; complete matched theory overlay present |
| 10 | `figure_10_empirical_penalized` | `empirical/penalized_phase.csv`, 5,000 points, n = 500, 50 repetitions; Figure 7 supplies theory overlay | `figures_heavy.generate_heavy_figures` | All Rank retuned; LS/LAD retuned at 760 affected points; iid spike-and-slab with all-zero vectors allowed; matched theory overlay present |
| S1 | `figS01_student_lad_vs_ls` | Same cached Student-t curve set as Figure 3, df 3–10 | `figures_light._student_figures` | Ratio LS risk / LAD risk; values above one favor LAD |
| S2 | `figS02_student_rank_over_lad` | Same cached Student-t curve set, df 1–10 | `figures_light._student_figures` | Ratio Rank risk / LAD risk = ARE(LAD:Rank), not ARE(Rank:LAD); values above one favor LAD |
| S3 | `figS03_isotonic_schematic` | Explicit new Gaussian schematic, tau = 0.9 | `figures_schematic.generate_schematic` | **Absent from received VM figures**; project baseline includes this documented replacement; original manuscript drawing source is unavailable |

## Verified differences from the historical results

Tables were aligned by df, delta, and penalty where applicable; coordinates match to relative/absolute tolerance 1e-14.
Winners were recomputed by argmin from the actual risk columns, with unavailable LS theory treated as infinity.
Every stored winner label agrees with the recomputed label.
The machine-readable counts are in `figure-winner-comparison.json` beside this report.

| Figure | Changed numerical argmin cells | Breakdown |
|---|---:|---|
| 1 | 2 / 10,000 | Rechecked quadrature cells |
| 7 | 146 / 20,000 | Lasso 146; ridge 0 |
| 9 | 54 / 2,500 | Changes occur at delta from 1.001 through about 2.224 |
| 10 | 91 / 5,000 | Lasso 70; ridge 21 |

Therefore the README's statement that the recovered Figure 10 heatmap agrees in all 5,000 cells with the manuscript is true only of the **historical recovery**, not the corrected figure now proposed for publication.
The corrected figures and manuscript figure assets must be updated together.
The broad three-region structure remains visually apparent, but avoid implying that every cell is unchanged or that every numerical winner is statistically or numerically separated.
Figure 7 has 20 cells with relative gap below 1e-8 between the two best numerical risks and 8,037 below 1%; the colors show numerical argmin, not a certificate of strict superiority.

## Required manuscript explanations

1. **Section 6 unpenalized algorithm:** replace the L-BFGS-B-at-LAD description.
   The corrected fits use exact rank proximal operations through convex ADMM, with a direct centered linear-system solution in the interpolation case p = n − 1.
   The one repaired ADMM repetition uses fixed rho = 1/n after adaptive-rho failure, at the same tolerance.
   Refer to public diagnostics for residual and objective-gap checks rather than call a numerical solution exact.
2. **Section 6 unpenalized simulation convention:** the manuscript states iid N(0,1) coefficients, while recorded simulations use a normalized Gaussian vector (uniform on the unit sphere) and design variance one.
   Explain the normalization equivalence: X_paper = X_code/sqrt(p), beta_paper = sqrt(p) beta_code, so squared coefficient error in code equals normalized squared coefficient error in paper coordinates.
   Unpenalized coefficient-error distributions are translation equivariant, so the arbitrary coefficient law does not change the risk comparison; still state the actual simulation construction for reproducibility.
3. **Section 6 penalized experiment:** explicitly state n = 500, iid 0.8 delta0 + 0.2 N(0,5), and no forced nonzero coordinate.
   Lambda minimizes the median Monte Carlo MSE over the finite adaptive candidate set; the plotted winner minimizes the arithmetic mean MSE at each method's selected lambda, using the same 50 repetitions.
   This is oracle Monte Carlo tuning, not cross-validation or independent evaluation.
4. **Figure 2 caption:** add the LAD comparison and classical limit 2/pi, or remove that additional curve if the author intends the original Rank-only figure.
   The generated figure deliberately contains both; its present manuscript caption is incomplete.
5. **Figure 8 caption / Section 5.2:** describe the lower panels as selected lambda values and say the unrestricted infimum is approximated numerically by finite searches.
   LS ridge at df = 3, delta = 0.1 reaches the archived upper bound; the complete optimum table is available, but the full lambda-curve archive is truncated.
6. **Figure S3:** adopt the explicit Gaussian replacement knowingly or retain the original manuscript drawing.
   Do not label the new Gaussian construction as recovery of the original drawing source.

## Publication metadata

The public `data/paper/` metadata describe the mixed-origin grid, numerical-check scope, and available raw repetitions. Original metadata and fingerprints remain under `provenance/`. Numerical CSV bytes are unchanged. The default CLI reads corrected data; historical inputs remain separate for recomputation.

## Visual checks

All 12 received paper/S1/S2 figures were visually inspected.
Axes, legends, overlays, and titles are readable; no clipped labels or visibly broken layouts were found.
Figures 9–10 clearly distinguish observed discrete Monte Carlo cells from smooth theoretical boundaries.
Figure 8 is dense at small display sizes; use its full-size PDF/PNG link rather than a small README thumbnail for numerical reading.
The current Figure 4 uses the correct quantile fit h versus proximal score psi and separates illustrative parameters from Figures 5–6's self-consistent parameters.
