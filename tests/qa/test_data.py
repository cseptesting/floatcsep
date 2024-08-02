from floatcsep.cmd import main
from floatcsep.experiment import Experiment
import unittest
from unittest.mock import patch
import os


class DataTest(unittest.TestCase):

    @staticmethod
    def get_runpath(case):
        return os.path.abspath(
            os.path.join(__file__, '../../..',
                         'examples',
                         f'case_{case}',
                         f'config.yml')
        )

    @staticmethod
    def get_rerunpath(case):
        return os.path.abspath(
            os.path.join(__file__, '../../..', 'examples', f'case_{case}',
                         'results', f'repr_config.yml')
        )

    @staticmethod
    def run_evaluation(cfg_file):
        main.run(cfg_file, show=False)

    @staticmethod
    def repr_evaluation(cfg_file):
        main.reproduce(cfg_file, show=False)

    def get_eval_dist(self):
        pass


@patch.object(Experiment, "generate_report")
@patch.object(Experiment, "plot_forecasts")
@patch.object(Experiment, "plot_catalog")
class RunExamples(DataTest):

    def test_case_a(self, *args):
        cfg = self.get_runpath('a')
        self.run_evaluation(cfg)
        self.assertEqual(1, 1)

    def test_case_b(self, *args):
        cfg = self.get_runpath('b')
        self.run_evaluation(cfg)
        self.assertEqual(1, 1)

    def test_case_c(self, *args):
        cfg = self.get_runpath('c')
        self.run_evaluation(cfg)
        self.assertEqual(1, 1)

    def test_case_d(self, *args):
        cfg = self.get_runpath('d')
        self.run_evaluation(cfg)
        self.assertEqual(1, 1)

    def test_case_e(self, *args):
        cfg = self.get_runpath('e')
        self.run_evaluation(cfg)
        self.assertEqual(1, 1)

    def test_case_f(self, *args):
        cfg = self.get_runpath('f')
        self.run_evaluation(cfg)
        self.assertEqual(1, 1)

    def test_case_g(self, *args):
        cfg = self.get_runpath('g')
        self.run_evaluation(cfg)
        self.assertEqual(1, 1)

    def test_case_h(self, *args):
        cfg = self.get_runpath('h')
        self.run_evaluation(cfg)
        self.assertEqual(1, 1)


@patch.object(Experiment, "generate_report")
@patch.object(Experiment, "plot_forecasts")
@patch.object(Experiment, "plot_catalog")
class ReproduceExamples(DataTest):

    def test_case_c(self, *args):
        cfg = self.get_rerunpath('c')
        self.repr_evaluation(cfg)
        self.assertEqual(1, 1)

    def test_case_f(self, *args):
        cfg = self.get_rerunpath('f')
        self.repr_evaluation(cfg)
        self.assertEqual(1, 1)
