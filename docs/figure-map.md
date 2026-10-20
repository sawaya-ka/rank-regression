# Figure inputs and interpretation

Generate all static figures with:

```bash
rank-repro figures --group all --output-dir build/figures
```

Each figure is written as a PDF and PNG. Data paths below are relative to `data/paper/`; generator names are modules in `src/rank_repro/`.

| Figure | Output basename | Input or construction | Generator |
|---|---|---|---|
| 1 | `fig01_student_efficiency` | `unpenalized_student_t.csv`; classical constants from beta-function formulas | `figures_light` |
| 2 | `fig02_gaussian_efficiency` | Gaussian scalar risk equations, including δ = √2 | `figures_light` |
| 3 | `fig03_student_rank_vs_ls` | Student-t risks for df 3–10 and δ from 1.01 to 10 | `figures_light` |
| 4 | `fig04_isotonic_projection` | Standard Gaussian effective residual; τ = 0.45 and 0.90 | `figures_light` |
| 5 | `fig05_effective_scores` | Gaussian scores at each method's self-consistent parameters; δ = 1.1, …, 1.8 | `figures_light` |
| 6 | `fig06_proximal_atom` | Gaussian population proximal output's zero-atom mass | `figures_light` |
| 7 | `figure_07_penalized_phase` | `theory/penalized_phase.csv` | `figures_heavy` |
| 8 | `figure_08_penalized_are` | `theory/penalized_are.csv` | `figures_heavy` |
| 9 | `figure_09_empirical_unpenalized` | `empirical/unpenalized_phase.csv`; Figure 1 theory grid for boundaries | `figures_heavy` |
| 10 | `figure_10_empirical_penalized` | `empirical/penalized_phase.csv`; Figure 7 theory grid for boundaries | `figures_heavy` |
| S1 | `figS01_student_lad_vs_ls` | Same Student-t risk curves as Figure 3, df 3–10 | `figures_light` |
| S2 | `figS02_student_rank_over_lad` | Same Student-t risk curves, df 1–10 | `figures_light` |
| S3 | `figS03_isotonic_schematic` | Gaussian illustration with τ = 0.9 | `figures_schematic` |

To regenerate selected heavy figures, use `--group heavy --figures 8 9 10`. The default input directory is `data/paper/`; use `--data-dir PATH` for another complete compatible data set.

## Risk and score curves

Figure 1 uses a 100 × 100 grid of Student-t degrees of freedom and δ. The top panel shows classical limiting variances; the phase diagram identifies the smallest numerical risk. See [quadrature scope](reproduction.md#numerical-methods-and-limitations) before interpreting close boundaries.

Figures 2 and 3 plot LS risk divided by the comparison method's risk, so values above one favor that method. Figure 2 includes both Rank and LAD. Figure S1 plots LS risk / LAD risk. Figure S2 instead plots Rank risk / LAD risk, so values above one favor LAD.

Figure 4 illustrates the proximal construction with c = 1/√2 and Gaussian noise variance 1/2, giving a standard Gaussian effective residual. Its τ values are illustrative. Figures 5 and 6 use self-consistent parameters instead. In Figure 6, the zero-atom mass is 1/δ for LAD and zero for LS.

The renderer exports `gaussian_diagnostics.json` and `student_risk_curves.csv` alongside the figures. Figure S3 uses an explicit Gaussian example of the same isotonic construction.

## Penalized theory and finite-sample comparisons

Figures 7 and 10 use the signal law `0.8 delta_0 + 0.2 N(0,5)` for lasso comparisons. Ridge theory depends on its second moment, which is one. Figure 8's upper panels show optimized LS risk / Rank risk; its lower panels show the selected penalty values in the paper's `lambda/n` convention. The finite-search boundary limitation is described in the [reproduction guide](reproduction.md#numerical-methods-and-limitations).

Figure 9 uses n = 1,000 and ten repetitions at each of 2,500 points. Figure 10 uses n = 500 and 50 repetitions at each of 5,000 points across ridge and lasso. In Figure 10, each method's penalty is selected by median Monte Carlo MSE, then the displayed winner is determined by mean MSE at the selected penalty, using the same repetitions.

The heatmap colors identify numerical minima. They are not significance tests and do not imply that every pair of neighboring regions is separated beyond numerical or Monte Carlo uncertainty. Black curves in Figures 9 and 10 show matching theoretical boundaries.
