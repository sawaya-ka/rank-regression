"""Mathematical and restart checks for the published light reproduction path."""
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from rank_repro.gaussian import (solve_ls_gaussian, solve_lad_gaussian, solve_rank_gaussian,
                                 verify_self_consistency, score_function, rank_atom_mass)
from rank_repro.student import solve_student_t, classical_risks, load_interpolated_risks
from rank_repro.light_experiments import run_light_experiment


class GaussianIdentities(unittest.TestCase):
    def test_explicit_inactive_risk_and_transition(self):
        for delta in (np.sqrt(2), 2., 10.):
            solution = solve_rank_gaussian(delta)
            self.assertAlmostEqual(solution.alpha2, np.pi/(3*delta-np.pi), places=12)
            self.assertEqual(solution.c, 0)
            self.assertEqual(rank_atom_mass(delta), 0)
        self.assertGreater(solve_rank_gaussian(1.2).c, 0)
        self.assertGreater(rank_atom_mass(1.2), 0)
        self.assertLess(rank_atom_mass(1.2), 1/1.2)

    def test_independent_quadrature_equations(self):
        for delta in (1.1, 1.4, 1.8, 4.):
            for solver in (solve_ls_gaussian, solve_lad_gaussian, solve_rank_gaussian):
                solution = verify_self_consistency(solver(delta))
                self.assertLess(abs(solution.residual_alpha2), 1e-10)
                self.assertLess(abs(solution.residual_alpha), 1e-10)

    def test_proximal_output_monotonicity_and_score_bounds(self):
        x = np.linspace(-20,20,10001)
        for delta in (1.1, 1.8):
            solution = solve_rank_gaussian(delta)
            psi = score_function(solution)(x)
            output = x-psi
            self.assertGreaterEqual(np.diff(output).min(), -1e-11)
            self.assertGreaterEqual(np.diff(psi).min(), -1e-11)
            self.assertLessEqual(np.diff(psi).max(), x[1]-x[0]+1e-11)
            self.assertLessEqual(np.abs(psi).max(), 2*solution.kappa+1e-11)
            if solution.c:
                central = np.abs(x)<solution.c
                np.testing.assert_allclose(output[central], 0, atol=1e-12)


class StudentChecks(unittest.TestCase):
    def test_representative_risk_and_quadrature_convergence(self):
        coarse = solve_student_t(8,2,n_quad=150)
        fine = solve_student_t(8,2,n_quad=300)
        self.assertAlmostEqual(fine["alpha2_LS"]/fine["alpha2_rank"], .97513, delta=2e-5)
        self.assertLess(abs(fine["alpha2_rank"]/coarse["alpha2_rank"]-1), 3e-5)
        self.assertLess(abs(fine["alpha2_LAD"]/coarse["alpha2_LAD"]-1), 3e-5)

    def test_infinite_ls_is_not_failed_robust_solve(self):
        row = solve_student_t(1,2,n_quad=150)
        self.assertTrue(math.isinf(row["alpha2_LS"]))
        self.assertAlmostEqual(row["alpha2_rank"], 10.00215, delta=1e-4)
        self.assertTrue(math.isfinite(row["alpha2_LAD"]))
        classical = classical_risks(np.array([1.,2.,8.]))
        self.assertTrue(np.isinf(classical[0,:2]).all())
        self.assertAlmostEqual(classical[0,2], 8/6)
        self.assertAlmostEqual(classical[1,0], np.pi**2/4, places=12)
        self.assertAlmostEqual(classical[2,0], np.pi**2/3, places=12)

    def test_phase_loader_rejects_missing_and_duplicate_points(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"grid.csv"
            rows = [{"df":df,"delta":delta,"alpha_LAD":1.,"alpha_rank":2.}
                    for df in (3.,4.) for delta in (2.,3.)]
            pd.DataFrame(rows[:-1]).to_csv(path,index=False)
            with self.assertRaises(ValueError):
                load_interpolated_risks(path,fine_n_df=4,fine_n_delta=4)
            pd.DataFrame(rows+[rows[0]]).to_csv(path,index=False)
            with self.assertRaises(ValueError):
                load_interpolated_risks(path,fine_n_df=4,fine_n_delta=4)


class RankLossIdentity(unittest.TestCase):
    def test_sorted_rank_loss_matches_pairwise_absolute_distance(self):
        from rank_repro._unpenalized import rank_loss_and_grad
        X = np.array([[1.,0.],[-1.,2.],[0.,1.],[2.,-1.]])
        y = np.array([2.,-1.,3.,.5])
        beta = np.array([.2,-.1])
        residual = y-X@beta
        loss, gradient = rank_loss_and_grad(beta,X,y)
        n = len(y)
        pairwise = sum(abs(residual[i]-residual[j]) for i in range(n) for j in range(i+1,n))
        self.assertAlmostEqual(loss, math.sqrt(12)*pairwise/(2*n*(n+1)), places=13)
        for j in range(2):
            step = np.zeros(2)
            step[j] = 1e-6
            numerical = (rank_loss_and_grad(beta+step,X,y)[0]-rank_loss_and_grad(beta-step,X,y)[0])/(2e-6)
            self.assertAlmostEqual(gradient[j], numerical, places=8)
        # Adding a common residual shift leaves the rank objective unchanged.
        self.assertAlmostEqual(loss,rank_loss_and_grad(beta,X,y+10)[0], places=13)


class LightCheckpointChecks(unittest.TestCase):
    def test_resume_and_config_guard(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = root/"config.json"
            payload = {"experiment":"unpenalized_theory","preset":"smoke",
                       "settings":{"df_values":[1.,8.],"delta_values":[2.],"n_quad":60}}
            config.write_text(json.dumps(payload))
            output = root/"data"
            plan = run_light_experiment(config,output,workers=1,dry_run=True)
            self.assertEqual(plan["jobs"],2)
            self.assertFalse(output.exists())
            first = run_light_experiment(config,output,workers=1)
            self.assertTrue(first["complete"])
            with patch("rank_repro.light_experiments._worker",side_effect=AssertionError("should use checkpoints")):
                second = run_light_experiment(config,output,workers=1)
            self.assertEqual(second["completed_jobs"],2)
            rows = pd.read_csv(output/"unpenalized_student_t.csv")
            self.assertTrue(math.isinf(rows.loc[0,"alpha2_LS"]))
            metadata = json.loads((output/"unpenalized_student_t.metadata.json").read_text())
            self.assertEqual(metadata["noise"],"standard_student_t")
            self.assertTrue(metadata["complete"])
            payload["settings"]["n_quad"] = 80
            config.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                run_light_experiment(config,output,workers=1)

    def test_failed_job_does_not_complete_the_grid(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = root/"config.json"
            config.write_text(json.dumps({"experiment":"unpenalized_theory","preset":"smoke",
                                        "settings":{"df_values":[3.],"delta_values":[2.],"n_quad":40}}))
            failure = {"complete":False,"row":None,"error":"numerical failure"}
            with patch("rank_repro.light_experiments._worker",return_value=failure):
                result = run_light_experiment(config,root/"data",workers=1)
            self.assertFalse(result["complete"])
            self.assertEqual(result["completed_jobs"],0)
            self.assertEqual(len(pd.read_csv(root/"data/unpenalized_student_t.csv")),0)
            retry = run_light_experiment(config,root/"data",workers=1)
            self.assertTrue(retry["complete"])


if __name__ == "__main__":
    unittest.main()
