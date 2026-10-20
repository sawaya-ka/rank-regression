# Reproduction and numerical scope

## Three separate tasks

1. **Redraw**: `rank-repro figures --group all --output-dir build/figures` reads `data/paper` and recomputes deterministic scalar curves. The animation renderer reads its committed grid. Neither starts finite-sample simulations.
2. **Recompute the reviewed corrections**: `python tools/run_recompute_plan.py --workers 8 --output-dir recompute-results` starts the 18,365-job plan from the retained historical inputs. Successful checkpoints are reused only with a matching source/configuration fingerprint; failed checkpoints are retried on the next invocation.
3. **Explore other experiments**: `rank-repro compute EXPERIMENT --config configs/EXPERIMENT.smoke.json --dry-run` describes a standalone computation. The standalone `*.paper.json` files are examples of full experiment specifications, not the historical correction plan.

Run commands from the repository root after installing the package. The cloud equivalent and disconnect behavior are described in [the execution guide](one-command-ja.md).

## Archived run and new source

The original source fingerprint is `17d6759bacbcbeb3c821745b21e102216e67dc3ac023fc618d22190d75ffe39a`.
The repaired composite fingerprint is `80e2d1b5014ebb2019f5dd5851c53f9543b1639a00d7a4a410e26ce8edad9375`.
The received archive has SHA-256 `9186dbf80fdfe6b25b67a72f05d026015a3ab022d3ed023c0c925d2db2bc48f4` and 546,645,073 bytes.

The composite archive contains 18,346 unchanged parent checkpoints, 19 repaired checkpoints, and the 19 original failed records. All hashes are recorded in `provenance/checkpoint_provenance.json` and `provenance/repair_manifest.json`. The raw release asset retains numerical file bytes and records its derivation from the received archive.

`provenance/original-run-source.tar.gz` preserves the original numerical source, settings, and historical inputs. Its `tools/run_recompute_plan.py` reproduces the parent fingerprint. The repair-specific source is included in the raw release asset's `source/` directory. The archived source is evidence of the producing implementation, rather than the recommended entry point for a fresh run. Cloud setup files and operational logs are not needed to reproduce this numerical fingerprint.

The public working source incorporates the successful numerical recovery into fresh runs, changes plotting defaults to corrected tables, and removes the author's cloud-project default. A new fingerprint is therefore expected. Do not rewrite old checkpoint fingerprints or resume an old run with a changed source snapshot.

## Numerical recovery

Eighteen quadrature cells at δ = 1.001 failed the original 300/600/1200-node comparison. Continuing to 2400 and, where needed, 4800 nodes resolved them at the same relative threshold of 5 × 10⁻⁴. Fifteen accepted at 2400 nodes and three at 4800 nodes. This is a successive-order diagnostic, not an analytic error bound.

At one unpenalized empirical coordinate, an adaptive ADMM penalty cycled and one of ten repetitions reached its iteration limit. Holding rho = 1/n fixed converged in 3,506 iterations with the unchanged 50,000-iteration cap, absolute tolerance 10⁻⁶, and relative tolerance 10⁻⁵. The other nine successful repetitions were retained. Fresh reruns apply this fallback only after a maximum-iteration failure; both attempts are recorded.

All selected Rank fits in the received archive satisfy their primal-residual, dual-residual, and objective-gap gates. Independent small linear/quadratic-program comparisons are included in the tests. These checks establish accuracy to the stated tolerances, not exact optimization.

## Reuse and remaining limitations

- Figure 1 replaces 865 checked cells; 9,135 cells retain original 100-node values. An unexamined near-tie cell can still be sensitive to quadrature. Plot interpolation is for display only.
- Figure 7 reuses ridge theory, because the two signal distributions have the same second moment, and recomputes lasso for the Figure 10 signal law. The search starts from 200 penalty candidates and refines eligible minima. Numerical argmin regions near ties should not be read as large efficiency differences.
- Figure 8 retains the complete recovered optimum table. One LS optimum reaches the upper search boundary; no unrestricted infimum is claimed. The complete historical candidate-curve file was not recovered.
- Figure 9 retains historical LS/LAD summaries. Their original raw repetitions are not part of the new correction archive.
- Figure 10 recomputes all Rank points and LS/LAD at the 760 points affected by forced-nonzero signal generation. Other LS/LAD points retain historical diagnostics. The new raw release contains selected repetitions and candidate summaries, not every fitted coefficient vector or every historical repetition.
- Figure S3 is a documented new Gaussian illustration, not recovery of the original drawing source.

## Environment and runtime

The VM used Python 3.10.12, NumPy 2.2.6, SciPy 1.15.3, pandas 2.2.3, scikit-learn 1.7.2, and Matplotlib 3.10.8. `requirements-tested.txt` pins these dependencies. Results can differ slightly across BLAS libraries, operating systems, and Python versions. PDF metadata and font rendering can prevent byte-identical figures even when numerical inputs agree.

The parent run took approximately 27.5 hours. The repair lasted 1,676.49 seconds with two workers, including output assembly and figures. These times do not predict a fresh full run on a different machine. High-order quadrature can require substantial memory; each worker uses one BLAS thread, and worker count must fit both CPU and RAM.

The software never treats a smoke test as a completed paper run. The publication validation report states which checks were executed and on which environment.

## Repeat the raw-record audit

After downloading and checksum-verifying the raw release asset, extract it into `build/raw-results/` and run:

```bash
python tools/audit_archive.py --extracted build/raw-results --output-dir build/archive-audit
```

This read-only check uses the Python standard library. It verifies all checkpoint hashes and provenance, retained parent failures, numerical acceptance gates, and checkpoint-to-CSV transfers. It requires the complete raw asset, not just the small tables committed to Git. A failed check returns a nonzero exit status; details are written to `build/archive-audit/received-run-audit.json`.
