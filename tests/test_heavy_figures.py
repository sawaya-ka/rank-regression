"""Integrity guards for heavy figures; all tables here are test-only synthetic data."""

import json
import tempfile
import unittest
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from rank_repro.figures_heavy import FigureDataError, generate_heavy_figures


class HeavyFigureTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.out = self.root / "figures"
        self.signal = {"kind": "spike_slab", "nonzero_probability": .2, "nonzero_variance": 5}
        self.meta = {"noise": "standard_student_t", "signal": self.signal,
                     "grid": {"df": [3, 6, 15], "delta": [1.2, 3, 10], "penalty": ["ridge", "lasso"]},
                     "n": 24, "n_rep": 2, "complete": True,
                     "selection": {"criterion": "mse_mean", "evaluation": "same_repetitions", "display_statistic": "mean"},
                     "optimization": {"purpose": "synthetic test only"}}
        rows = []
        for penalty in self.meta["grid"]["penalty"]:
            for df in self.meta["grid"]["df"]:
                for delta in self.meta["grid"]["delta"]:
                    ls, lad, rank = df / (10 * delta), 2 / (df * delta), .4 / delta
                    rows.append({"penalty": penalty, "df": df, "delta": delta,
                                 "alpha2_ls_opt": ls, "alpha2_lad_opt": lad, "alpha2_rank_opt": rank,
                                 "lambda_ls_opt": 1 / delta, "lambda_rank_opt": 2 / delta,
                                 "mse_LS_mean": ls, "mse_LAD_mean": lad, "mse_rank_mean": rank,
                                 "n": 24, "n_rep": 2, "success_rate_LS": 1., "success_rate_LAD": 1., "success_rate_rank": 1.})
        self.frame = pd.DataFrame(rows)
        for relative in ("theory/penalized_phase.csv", "theory/penalized_are.csv", "empirical/penalized_phase.csv"):
            self.write(relative, self.frame, self.meta)
        unpen = self.frame[self.frame["penalty"] == "ridge"].drop(columns="penalty").copy()
        unpen_meta = {**self.meta, "grid": {k: v for k, v in self.meta["grid"].items() if k != "penalty"}}
        self.write("empirical/unpenalized_phase.csv", unpen, unpen_meta)
        theory = unpen.rename(columns={"alpha2_ls_opt": "alpha2_LS", "alpha2_lad_opt": "alpha2_LAD", "alpha2_rank_opt": "alpha2_rank"})
        self.write("unpenalized_student_t.csv", theory, unpen_meta)

    def write(self, relative, frame, meta):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
        path.with_suffix(".metadata.json").write_text(json.dumps(meta))

    def tearDown(self):
        plt.close("all")
        self.temporary.cleanup()

    def test_all_four_figures_render_from_complete_tables(self):
        paths = generate_heavy_figures(self.root, self.out)
        self.assertEqual(len(paths), 8)
        for path in paths:
            signature = b"%PDF" if path.suffix == ".pdf" else b"\x89PNG\r\n\x1a\n"
            self.assertTrue(path.read_bytes().startswith(signature))
            record = json.loads(path.with_suffix(".metadata.json").read_text())
            self.assertIn("data_metadata", record)
            if record["figure"] in ("9", "10"):
                self.assertTrue(record["theory_overlay"])

    def test_one_missing_cell_is_rejected(self):
        self.write("theory/penalized_phase.csv", self.frame.iloc[:-1], self.meta)
        with self.assertRaisesRegex(FigureDataError, "incomplete grid"):
            generate_heavy_figures(self.root, self.out, ["7"])

    def test_rectangular_but_truncated_axis_is_rejected(self):
        self.write("theory/penalized_phase.csv", self.frame[self.frame["df"] != 15], self.meta)
        with self.assertRaisesRegex(FigureDataError, "coordinates differ"):
            generate_heavy_figures(self.root, self.out, ["7"])

    def test_duplicate_rows_are_not_dropped(self):
        self.write("theory/penalized_phase.csv", pd.concat([self.frame, self.frame.iloc[:1]]), self.meta)
        with self.assertRaisesRegex(FigureDataError, "duplicate"):
            generate_heavy_figures(self.root, self.out, ["7"])

    def test_signal_mismatch_blocks_theory_overlay(self):
        other = {**self.meta, "signal": {"kind": "spike_slab", "nonzero_probability": .1, "nonzero_variance": 10}}
        self.write("theory/penalized_phase.csv", self.frame, other)
        with self.assertRaisesRegex(FigureDataError, "signal distributions differ"):
            generate_heavy_figures(self.root, self.out, ["10"])

    def test_failed_repetitions_block_published_comparison(self):
        failed = self.frame.copy()
        failed.loc[0, "success_rate_rank"] = .5
        self.write("empirical/penalized_phase.csv", failed, self.meta)
        with self.assertRaisesRegex(FigureDataError, "failed repetitions"):
            generate_heavy_figures(self.root, self.out, ["10"])

    def test_incomplete_flag_blocks_even_full_table(self):
        self.write("theory/penalized_phase.csv", self.frame, {**self.meta, "complete": False})
        with self.assertRaisesRegex(FigureDataError, "reports incomplete"):
            generate_heavy_figures(self.root, self.out, ["7"])

    def test_missing_optional_overlay_is_disclosed(self):
        (self.root / "theory/penalized_phase.csv").unlink()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            path, _png_path = generate_heavy_figures(self.root, self.out, ["10"])
        self.assertTrue(any("overlay omitted" in str(item.message) for item in caught))
        record = json.loads(path.with_suffix(".metadata.json").read_text())
        self.assertFalse(record["theory_overlay"])

    def test_missing_required_table_does_not_copy_a_pdf(self):
        (self.root / "theory/penalized_phase.csv").unlink()
        with self.assertRaisesRegex(FigureDataError, "Required result table is missing"):
            generate_heavy_figures(self.root, self.out, ["7"])
        self.assertEqual(list(self.out.glob("*.pdf")), [])


if __name__ == "__main__":
    unittest.main()
