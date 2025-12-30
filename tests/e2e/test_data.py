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

    @staticmethod
    def view_dashboard(cfg_file):
        main.view(cfg_file, show=True, start=False)


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


@patch("floatcsep.commands.main.plot_forecasts")
@patch("floatcsep.commands.main.plot_catalogs")
@patch("floatcsep.commands.main.plot_custom")
@patch("floatcsep.commands.main.generate_report")
class ViewExamples(DataTest):
    def test_case_c(self, *args):
        cfg = self.get_rerunpath("c")
        self.view_dashboard(cfg)
        self.assertEqual(1, 1)

    def test_case_f(self, *args):
        cfg = self.get_rerunpath("f")
        self.view_dashboard(cfg)
        self.assertEqual(1, 1)

    def test_case_g(self, *args):
        cfg = self.get_rerunpath("g")
        self.view_dashboard(cfg)
        self.assertEqual(1, 1)