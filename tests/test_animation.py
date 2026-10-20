"""Scientific invariants of the committed animation grid (no expensive rerun)."""
from pathlib import Path
import unittest

import numpy as np

from rank_repro.student import classical_risks

ROOT = Path(__file__).resolve().parents[1]


class AnimationData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = np.load(ROOT / 'animation' / 'data' / 'risk_grid.npz')

    @classmethod
    def tearDownClass(cls):
        cls.data.close()

    def test_ls_formula_and_infinite_variance_region(self):
        df, delta, risks = (self.data[k] for k in ('df', 'delta', 'risks'))
        finite = df > 2
        expected = df[None, finite] / ((df[None, finite] - 2) * (delta[:, None] - 1))
        np.testing.assert_allclose(risks[:, 0, finite], expected, rtol=1e-13)
        self.assertTrue(np.isinf(risks[:, 0, ~finite]).all())

    def test_classical_frame_and_dimensional_risk_order(self):
        np.testing.assert_allclose(self.data['classical'], classical_risks(self.data['df']), rtol=1e-12)
        robust = self.data['risks'][:, 1:]
        self.assertTrue(np.isfinite(robust).all())
        self.assertTrue((robust > 0).all())
        self.assertTrue((np.diff(robust, axis=0) >= -1e-8).all())


if __name__ == '__main__':
    unittest.main()
