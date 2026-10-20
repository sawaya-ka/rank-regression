#!/usr/bin/env python3
"""Recompute Student-t LS/LAD/Rank risks for the animation.

Run from the repository root after installing the package:
    python animation/compute_risks.py --workers 6
The shared rank_repro.student solver is used. No empirical data or archived
risk interpolation is used. Display interpolation is left to the renderer.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from rank_repro import student

ROOT = Path(__file__).resolve().parent

def compute_column(task):
    index, df, deltas, order = task
    s = student
    rows = s.risk_curve(float(df), np.array(deltas), order=order)
    return index, rows[:, 1:4], rows[:, 4]


def check_point(task):
    i, j, df, delta, order = task
    s = student
    row = s.risk_curve(float(df), [float(delta)], order=order)[0]
    return i, j, row[1:4], row[4]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--order', type=int, default=600)
    parser.add_argument('--output-dir', type=Path, default=ROOT.parent/'build'/'animation-data')
    args = parser.parse_args()
    if args.workers < 1 or args.order < 2:
        parser.error("workers must be positive and order must be at least 2")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    s = student
    boundaries = [x[1] for x in s._classical_regions(1., 50.)][:-1]
    df = np.unique(np.r_[np.geomspace(1., 50., 100), 2., 3., boundaries])
    delta = np.unique([float(f'{d:.3g}') for d in np.geomspace(100., 1.01, 100)])[::-1]
    risks = np.full((len(delta), 3, len(df)), np.nan)
    cutoffs = np.full((len(delta), len(df)), np.nan)
    print(json.dumps({'df_count':len(df), 'delta_count':len(delta), 'quadrature_order':args.order, 'workers':args.workers}), flush=True)
    tasks = [(i, nu, delta.tolist(), args.order) for i, nu in enumerate(df)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for n, future in enumerate(as_completed([pool.submit(compute_column, t) for t in tasks]), 1):
            i, values, cutoff = future.result()
            risks[:, :, i] = values
            cutoffs[:, i] = cutoff
            if n % 10 == 0 or n == len(df):
                print(f'Computed {n}/{len(df)} df curves; elapsed {time.monotonic()-started:.1f}s', flush=True)
    # Preserve completed computations even if a later diagnostic needs refinement.
    classical = s.classical_risks(df)
    np.savez_compressed(args.output_dir/'risk_grid.npz', df=df, delta=delta, risks=risks,
                        classical=classical, rank_cutoffs=cutoffs, quadrature_order=args.order)
    # Cover heavy tails, the high-dimensional endpoint, and both sides of
    # classical crossings. This checks quadrature stability, not an error proof.
    coords = []
    for target_df in (1., 1.5, 2., 3., 5., 20., 50.):
        j = int(np.argmin(abs(df - target_df)))
        for target_delta in (100., 2., 1.2, 1.01):
            i = int(np.argmin(abs(delta - target_delta)))
            coords.append((i, j))
    checks = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(check_point, (i,j,df[j],delta[i],2*args.order)) for i,j in coords]
        for future in as_completed(futures):
            i, j, higher, cutoff = future.result()
            rel = np.abs(higher[1:] - risks[i,1:,j]) / higher[1:]
            check = {'df':float(df[j]), 'delta':float(delta[i]), 'orders':[args.order,2*args.order], 'LAD_relative_difference':float(rel[0]), 'Rank_relative_difference':float(rel[1])}
            checks.append(check)
            if np.max(rel) > 5e-4:
                raise RuntimeError(f'Quadrature check exceeds relative tolerance: {check}')
    classical = s.classical_risks(df)
    assert np.isfinite(risks[:,1:,:]).all() and (risks[:,1:,:] > 0).all()
    assert np.isinf(risks[:,0,df<=2]).all()
    assert np.isfinite(risks[:,0,df>2]).all()
    # Monotonicity in delta, allowing only numerical roundoff.
    assert np.all(np.diff(risks[:,1:,:],axis=0) >= -1e-8)
    np.savez_compressed(args.output_dir/'risk_grid.npz', df=df, delta=delta, risks=risks,
                        classical=classical, rank_cutoffs=cutoffs, quadrature_order=args.order)
    rows = []
    for i,d in enumerate(delta):
        for j,nu in enumerate(df):
            rows.append({'delta':d, 'df':nu, 'alpha2_LS':risks[i,0,j], 'alpha2_LAD':risks[i,1,j], 'alpha2_Rank':risks[i,2,j], 'rank_cutoff':cutoffs[i,j], 'quadrature_order':args.order})
    pd.DataFrame(rows).to_csv(args.output_dir/'risk_grid.csv',index=False)
    report = {'noise':'standard Student-t with location zero and scale one; not variance standardized',
              'delta_definition':'n/p', 'finite_grid':'100 descending log-spaced delta values from 100 to 1.01, rounded to 3 significant digits before solving',
              'df_grid':'100 logarithmic values from 1 to 50, augmented with 2, 3 and classical winner crossings',
              'delta_count':len(delta), 'df_count':len(df), 'order':args.order,
              'checks':sorted(checks,key=lambda x:(x['df'],-x['delta'])),
              'max_relative_difference':max(max(c['LAD_relative_difference'],c['Rank_relative_difference']) for c in checks),
              'check_tolerance':5e-4,
              'limitation':'Quadrature order comparisons at selected points are numerical diagnostics, not rigorous or grid-uniform error bounds.',
              'classical_frame':'Exact closed-form constants for lim delta*alpha^2. Raw risks tend to zero for LAD/Rank and for LS when df>2. LS remains infinite at df<=2.',
              'elapsed_seconds':time.monotonic()-started}
    (args.output_dir/'numerical_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='checks'},indent=2),flush=True)

if __name__ == '__main__':
    main()
