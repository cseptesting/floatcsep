import unittest
from floatcsep.evaluation import Evaluation


class TestEvaluation(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        def mock_eval():
            return

        setattr(cls, "mock_eval", mock_eval)

    @staticmethod
    def init_noreg(name, func, **kwargs):
        evaluation = Evaluation(name=name, func=func, **kwargs)
        return evaluation

    def test_init(self):
        name = "N_test"
        eval_ = self.init_noreg(name=name, func=self.mock_eval)
        self.assertIs(None, eval_.type)
        dict_ = {
            "name": "N_test",
            "func": self.mock_eval,
            "func_kwargs": {},
            "ref_model": None,
            "plot_func": None,
            "plot_args": None,
            "plot_kwargs": None,
            "markdown": "",
            "_type": None,
            "results_repo": None,
            "catalog_repo": None,
        }
        self.assertEqual(dict_, eval_.__dict__)

    def test_discrete_args(self):
        pass

    def test_sequential_args(self):
        pass

    def test_write_result(self):
        pass

    def to_dict(self):
        pass

    @classmethod
    def tearDownClass(cls) -> None:
        pass
