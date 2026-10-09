import json
import sys

from floatcsep.commands import main

import unittest
from unittest.mock import patch
import os


def _is_ci():
    return os.getenv("CI") == "true" or os.getenv("GITHUB_ACTIONS") == "true"


def skip_on_ci(reason="local-only test"):
    return unittest.skipIf(_is_ci(), reason)


class DataTest(unittest.TestCase):

    @staticmethod
    def get_runpath(case):
        return os.path.abspath(
            os.path.join(__file__, "../../..", "tutorials", f"case_{case}", f"config.yml")
        )

    @staticmethod
    def get_rerunpath(case):
        return os.path.abspath(
            os.path.join(
                __file__, "../../..", "tutorials", f"case_{case}", "results", f"repr_config.yml"
            )
        )

    @staticmethod
    def run_evaluation(cfg_file):
        main.run(cfg_file, show=False)

    @staticmethod
    def repr_evaluation(cfg_file):
        main.reproduce(cfg_file, show=False)

    def get_eval_dist(self):
        pass



@patch("floatcsep.commands.main.plot_forecasts")
@patch("floatcsep.commands.main.plot_catalogs")
@patch("floatcsep.commands.main.plot_custom")
@patch("floatcsep.commands.main.generate_report")
class RunExamples(DataTest):

    def test_case_a(self, *args):
        cfg = self.get_runpath("a")
        self.run_evaluation(cfg)
        self.assertEqual(1, 1)

    def test_case_b(self, *args):
        cfg = self.get_runpath("b")
        self.run_evaluation(cfg)
        self.assertEqual(1, 1)

    def test_case_c(self, *args):
        cfg = self.get_runpath("c")
        self.run_evaluation(cfg)
        self.assertEqual(1, 1)

    @skip_on_ci("Tested only locally")
    def test_case_d(self, *args):

        try:
            cfg = self.get_runpath("d")
            self.run_evaluation(cfg)
            self.assertEqual(1, 1)
        except Exception as e:
            self.skipTest(f"Skipping test involving Zenodo. Try locally: {e!r}")

    def test_case_e(self, *args):
        cfg = self.get_runpath("e")
        self.run_evaluation(cfg)
        self.assertEqual(1, 1)

    def test_case_f(self, *args):
        cfg = self.get_runpath("f")
        self.run_evaluation(cfg)
        self.assertEqual(1, 1)

    @skip_on_ci("Tested only locally")
    def test_case_g(self, *args):
        cfg = self.get_runpath("g")
        self.run_evaluation(cfg)
        self.assertEqual(1, 1)

    @skip_on_ci("Tested only locally")
    @unittest.skipUnless(sys.version_info >= (3, 10), "Requires Python 3.10+")
    def test_case_h(self, *args):
        cfg = self.get_runpath("h")
        self.run_evaluation(cfg)
        self.assertEqual(1, 1)

    @skip_on_ci("Tested only locally")
    def test_case_i(self, *args):
        cfg = self.get_runpath("i")
        self.run_evaluation(cfg)
        self.assertEqual(1, 1)

    @skip_on_ci("Tested only locally")
    def test_case_j(self, *args):
        cfg = self.get_runpath("j")
        self.run_evaluation(cfg)
        self.assertEqual(1, 1)


@patch("floatcsep.commands.main.plot_forecasts")
@patch("floatcsep.commands.main.plot_catalogs")
@patch("floatcsep.commands.main.plot_custom")
@patch("floatcsep.commands.main.generate_report")
class ReproduceExamples(DataTest):

    def test_case_c(self, *args):
        cfg = self.get_rerunpath("c")
        self.repr_evaluation(cfg)
        self.assertEqual(1, 1)

    def test_case_f(self, *args):
        cfg = self.get_rerunpath("f")
        self.repr_evaluation(cfg)
        self.assertEqual(1, 1)


class ExportExamples(DataTest):
    """Exports the tutorials run above as static dashboards and checks the written files."""

    def check_dashboard(self, case, td=False, forecasts=True):
        cfg = self.get_rerunpath(case)
        main.export(cfg)
        out = os.path.join(os.path.dirname(cfg), "dashboard")
        with open(os.path.join(out, "manifest.json")) as f:
            manifest = json.load(f)
        for name in ("index.html", "js/app.js", "vendor/echarts/echarts.min.js", "grid.json"):
            self.assertTrue(os.path.isfile(os.path.join(out, name)), name)

        n_win = len(manifest["time_windows"])
        self.assertGreater(n_win, 0)
        self.assertEqual(manifest["experiment"]["exp_class"], "Time-Dependent" if td else "Time-Independent")
        if forecasts:
            for model in manifest["models"]:
                self.assertEqual(len(model["forecasts"]), n_win, model["name"])
                for rel in model["forecasts"].values():
                    with open(os.path.join(out, rel)) as f:
                        fc = json.load(f)
                    self.assertGreater(fc["n_cells"], 0)
                    self.assertEqual(len(fc["totals"]), len(fc["mags"]))

        with open(os.path.join(out, manifest["catalog"]["file"])) as f:
            cats = json.load(f)
        self.assertEqual(len(cats["testing"]["w"]), cats["testing"]["n"])
        self.assertEqual(cats["input"] is not None, td)

        with open(os.path.join(out, manifest["results"]["index"])) as f:
            index = json.load(f)
        tests = {t["id"] for t in manifest["tests"]}
        self.assertTrue(index)
        self.assertEqual({r["test"] for r in index}, tests)
        return manifest

    def test_case_c(self, *args):
        self.check_dashboard("c")

    def test_case_f(self, *args):
        self.check_dashboard("f", td=True)

    @skip_on_ci("Tested only locally")
    def test_case_g(self, *args):
        self.check_dashboard("g", td=True)
