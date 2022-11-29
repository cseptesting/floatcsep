from fecsep import main
import unittest
import os
from data_tests import nz


class DataTest(unittest.TestCase):

    def get_path(self):
        pass

    def run(self, result=None):
        res = super().run(result)
        return res

    def run_evaluation(self, folder):
        config_fn = os.path.join(folder, 'config.yml')
        main.run(config_fn)

    def get_eval_dist(self):
        pass


class TimeIndependentTest(DataTest):

    def test_nz(self):
        pass
        # a = self.run_evaluation(nz.__path__[0])
        # self.assertEqual(1, 1)
