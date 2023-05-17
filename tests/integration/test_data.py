import floatcsep
from floatcsep.cmd import main
import unittest
import os


class DataTest(unittest.TestCase):

    @staticmethod
    def get_path(case):
        return os.path.abspath(
            os.path.join(__file__, '../../..',
                         'examples',
                         f'case_{case}',
                         f'config.yml')
        )

    @staticmethod
    def run_evaluation(cfg_file):
        main.run(cfg_file, show=False)

    def get_eval_dist(self):
        pass


class TimeIndependentTest(DataTest):

    def test_case_a(self):
        cfg = self.get_path('a')
        self.run_evaluation(cfg)
        self.assertEqual(1, 1)

    def test_case_b(self):
        cfg = self.get_path('b')
        self.run_evaluation(cfg)
        self.assertEqual(1, 1)

    def test_case_c(self):
        cfg = self.get_path('c')
        self.run_evaluation(cfg)
        self.assertEqual(1, 1)

    def test_case_d(self):
        cfg = self.get_path('d')
        self.run_evaluation(cfg)
        self.assertEqual(1, 1)

    def test_case_e(self):
        cfg = self.get_path('e')
        self.run_evaluation(cfg)
        self.assertEqual(1, 1)

    def test_case_e(self):
        cfg = self.get_path('f')
        self.run_evaluation(cfg)
        self.assertEqual(1, 1)
