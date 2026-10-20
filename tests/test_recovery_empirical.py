import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from rank_repro import _empirical
from rank_repro.recovery_empirical import get_jobs, run_job, sample_original_unpenalized


class RecoveryEmpiricalTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / 'empirical').mkdir()
        self.rows = []
        for i in (7, 13):
            row = dict(penalty='ridge', df=4.0, delta=2.0, actual_delta=2.0,
                       n=40, p=20, n_rep=2, i_penalty=0, i_df=i, i_delta=9)
            for est, value in [('LS', 2.), ('LAD', 3.), ('rank', 4.)]:
                row.update({f'lambda_{est}_opt': 2 * value,
                            f'lambda_{est}_opt_archived': value,
                            f'lambda_eff_{est}_opt': value / 20,
                            f'mse_{est}_mean': value,
                            f'mse_{est}_median': value,
                            f'mse_{est}_std': .1,
                            f'success_rate_{est}': 1.})
            self.rows.append(row)
        args = dict(n=40, n_rep=2, base_seed=12345, beta_zero_prob=.8, beta_slab_var=5.,
                    lambda_criterion='mse_median', lambda_min_pos=1e-4, lambda_max=1000.)
        for filename in ['penalized_phase', 'unpenalized_phase']:
            path = self.root / 'empirical' / (filename + '.csv')
            pd.DataFrame(self.rows).to_csv(path, index=False)
            path.with_suffix('.metadata.json').write_text(json.dumps({'original_settings': {'args': args}}))
        (self.root / 'forced_nonzero_points.json').write_text(json.dumps({
            'base_seed': 12345, 'n_repetitions_each': 2,
            'points': [{'i_penalty': 0, 'i_df': 13, 'i_delta': 9}]}))

    def tearDown(self):
        self.tmp.cleanup()

    def test_subset_keeps_original_seed_indices_and_scales_lambda_once(self):
        jobs = get_jobs(self.root, {'kinds': ['penalized_repair'], 'include_all_penalized_rank': False})
        self.assertEqual(len(jobs), 1)
        cfg = jobs[0]['cfg']
        self.assertEqual((cfg['i_penalty'], cfg['i_df'], cfg['i_delta']), (0, 13, 9))
        self.assertEqual(cfg['lambda_init_rank'], 8.)
        self.assertEqual(cfg['lambda_min_pos'], .0002)
        self.assertEqual(cfg['lambda_max'], 2000.)
        self.assertFalse(cfg['ensure_nonzero_beta'])
        self.assertEqual(cfg['min_lambda_success_rate'], 1.)
        self.assertEqual(jobs[0]['estimators'], ['LS', 'LAD', 'rank'])

    def test_unaffected_points_retain_ls_and_lad(self):
        jobs = get_jobs(self.root, {'kinds': ['penalized_repair']})
        self.assertEqual(jobs[0]['estimators'], ['rank'])
        self.assertEqual(jobs[1]['estimators'], ['LS', 'LAD', 'rank'])

    def test_smoke_changes_size_but_keeps_original_indices(self):
        jobs = get_jobs(self.root, {'max_jobs_per_kind': 1, 'n_override': 30, 'n_rep_override': 1})
        self.assertEqual(len(jobs), 2)
        for job in jobs:
            self.assertTrue(job['smoke'])
            self.assertEqual(job['cfg']['n'], 30)
            self.assertEqual(job['cfg']['p'], 15)
            self.assertEqual(job['cfg']['n_rep'], 1)
            self.assertEqual(job['row']['i_df'], 7)
        self.assertEqual(jobs[1]['estimators'], ['LS', 'LAD', 'rank'])

    def test_original_unpenalized_rng_and_normalization(self):
        seed = 42
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((30, 15))
        beta = rng.standard_normal(15)
        beta /= np.linalg.norm(beta)
        noise = rng.standard_t(4., 30)
        actual_X, y, actual_beta = sample_original_unpenalized(30, 2., 4., seed)
        np.testing.assert_array_equal(actual_X, X)
        np.testing.assert_array_equal(actual_beta, beta)
        np.testing.assert_array_equal(y, X @ beta + noise)

    def test_forced_list_seed_mismatch_is_rejected(self):
        path = self.root / 'forced_nonzero_points.json'
        doc = json.loads(path.read_text())
        doc['base_seed'] = 999
        path.write_text(json.dumps(doc))
        with self.assertRaisesRegex(ValueError, 'different base seed'):
            get_jobs(self.root, {'kinds': ['penalized_repair']})

    def test_repair_keeps_only_selected_lambda_raw_and_reuses_unaffected_risks(self):
        job = get_jobs(self.root, {'kinds': ['penalized_repair']})[0]
        def fake_tune(cfg, estimator, init):
            cfg['raw_sink'].extend([
                {'lambda': init, 'rep': 0, 'success': True},
                {'lambda': init * 2, 'rep': 0, 'success': True},
            ])
            return ({'lambda': init, 'lambda_eff': init / cfg['n'],
                     'mse_mean': .5, 'mse_median': .4, 'mse_std': .1,
                     'success_rate': 1., 'lambda_n_evaluated': 2},
                    [{'lambda': init, 'success_rate': 1.}, {'lambda': init * 2, 'success_rate': 1.}])
        with patch.object(_empirical, 'tune_lambda_local_serial', side_effect=fake_tune):
            result = run_job(job)
        self.assertTrue(result['complete'])
        self.assertEqual(len(result['raw']), 1)
        self.assertEqual(len(result['trace']), 2)
        self.assertEqual(result['updates']['mse_LS_mean'], 2.)
        self.assertEqual(result['updates']['mse_LAD_mean'], 3.)
        self.assertEqual(result['updates']['mse_rank_mean'], .5)
        self.assertEqual(result['updates']['winner'], 'rank')

    def test_failed_rank_is_not_a_winner(self):
        job = get_jobs(self.root, {'kinds': ['penalized_repair']})[0]
        with patch.object(_empirical, 'tune_lambda_local_serial', return_value=(
                {'lambda': 1., 'mse_mean': .001, 'success_rate': 0.}, [])):
            result = run_job(job)
        self.assertFalse(result['complete'])
        self.assertEqual(result['updates']['winner'], 'none')

    def test_admm_dispatch_retains_gap_and_iteration_metadata(self):
        rng = np.random.default_rng(4)
        X = rng.normal(size=(24, 5)) / np.sqrt(5)
        y = rng.normal(size=24)
        beta, info = _empirical.fit_one_estimator('rank', 'ridge', X, y, .5, x0=None,
                                                  cfg={'rank_backend': 'admm'})
        self.assertTrue(info['success'])
        self.assertTrue(np.isfinite(beta).all())
        self.assertIn('duality_gap', info)
        self.assertEqual(info['nit'], info['iterations'])
        self.assertEqual(info['fun'], info['objective'])


    def test_admm_per_repetition_adapter_keeps_success_and_diagnostics(self):
        job = get_jobs(self.root, {'kinds': ['penalized_repair']})[0]
        task = dict(job['cfg'], estimator='rank', rep=0, lambdas=[job['cfg']['lambda_init_rank']])
        rows = _empirical.evaluate_lambdas_for_rep(task)
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]['success'], rows[0].get('error'))
        self.assertEqual(rows[0]['status'], 0)
        self.assertIn('solver_duality_gap', rows[0])
        self.assertEqual(rows[0]['solver_termination'], 'converged')

    def test_fixed_rho_fallback_reuses_only_failed_repetition_and_keeps_attempt(self):
        job = get_jobs(self.root, {'kinds': ['unpenalized_rank'], 'max_jobs_per_kind': 1,
                                  'rank_admm_maxiter': 50000})[0]
        successes = [True, False, True]
        calls = []

        def fake_fit(X, y, **kwargs):
            calls.append((X, y, kwargs))
            success = successes[len(calls) - 1]
            return np.zeros(X.shape[1]), {'success': success,
                'status': 'converged' if success else 'max_iterations',
                'iterations': 10 if success else 50000, 'objective': 1.}

        with patch('rank_repro.rank_solver.fit_rank_convex', side_effect=fake_fit):
            result = run_job(job)
        self.assertTrue(result['complete'])
        self.assertEqual(len(calls), 3)  # Successful first repetition is never retried.
        self.assertIs(calls[1][0], calls[2][0])
        self.assertIs(calls[1][1], calls[2][1])
        retry_controls = dict(calls[2][2])
        self.assertFalse(retry_controls.pop('adaptive_rho'))
        self.assertEqual(calls[1][2], retry_controls)
        self.assertEqual(result['raw'][0]['rank_nit'], 10)
        self.assertEqual(len(result['trace']), 1)
        attempt = result['trace'][0]
        self.assertEqual(attempt['rep'], 1)
        self.assertEqual(attempt['seed'], result['raw'][1]['seed'])
        self.assertEqual(attempt['prior_solver_info']['status'], 'max_iterations')
        self.assertTrue(attempt['scientific_inputs_unchanged'])

    def test_fallback_does_not_accept_failure_or_retry_other_status(self):
        job = get_jobs(self.root, {'kinds': ['unpenalized_rank'], 'max_jobs_per_kind': 1})[0]
        for status, expected_calls in [('max_iterations', 4), ('linear_solve_residual', 2)]:
            with self.subTest(status=status):
                with patch('rank_repro.rank_solver.fit_rank_convex', return_value=(
                        np.zeros(20), {'success': False, 'status': status,
                                       'iterations': 1, 'objective': 1.})) as fit:
                    result = run_job(job)
                self.assertEqual(fit.call_count, expected_calls)
                self.assertFalse(result['complete'])
                self.assertEqual(result['updates']['winner'], 'none')

    def test_constant_rho_matches_independent_pairwise_linear_program(self):
        from scipy.optimize import linprog
        from rank_repro.rank_solver import fit_rank_convex
        rng = np.random.default_rng(923)
        X, y = rng.normal(size=(12, 3)), rng.standard_t(3, size=12)
        left, right = np.triu_indices(len(y), 1)
        design, response = X[left] - X[right], y[left] - y[right]
        m, p = design.shape
        objective = np.r_[np.zeros(p), np.full(m, 1. / m)]
        inequality = np.vstack([np.c_[-design, -np.eye(m)], np.c_[design, -np.eye(m)]])
        reference = linprog(objective, A_ub=inequality, b_ub=np.r_[-response, response],
                            bounds=[(None, None)] * p + [(0, None)] * m, method='highs')
        self.assertTrue(reference.success)
        _, info = fit_rank_convex(X, y, adaptive_rho=False, atol=1e-9, rtol=1e-8)
        self.assertTrue(info['success'])
        self.assertEqual(info['rho_schedule'], 'constant')
        self.assertEqual(info['rho'], 1. / len(y))
        self.assertAlmostEqual(info['objective'], reference.fun, delta=2e-8)
        for residual, tolerance in [('primal_residual', 'primal_tolerance'),
                                    ('dual_residual', 'dual_tolerance'),
                                    ('duality_gap', 'gap_tolerance')]:
            self.assertLessEqual(info[residual], info[tolerance])


if __name__ == '__main__':
    unittest.main()
