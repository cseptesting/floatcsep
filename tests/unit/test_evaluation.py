import numpy
import unittest
from unittest import mock
from fecsep.evaluation import check_eval_args, Evaluation


def test_check_eval_args():
    eval = mock.MagicMock(type=['Sequential', ])

    pass


class TestEvaluation(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        def mock_eval():
            return

        setattr(cls, 'mock_eval', mock_eval)

    @staticmethod
    def init_noreg(name, func, **kwargs):
        """ Instantiates a model without using the @register deco,
        but mocks Model.Registry() attrs"""
        evaluation = Evaluation.__new__(Evaluation)
        Evaluation.__init__.__wrapped__(self=evaluation,
                                        name=name,
                                        func=func,
                                        **kwargs)
        return evaluation

    def test_init(self):
        name = 'N_test'
        eval_ = self.init_noreg(name=name,
                                func=self.mock_eval)
        self.assertIs(None, eval_.type)
        dict_ = {'name': 'N_test',
                 'func': self.mock_eval,
                 'func_kwargs': {},
                 'ref_model': None,
                 'plot_func': None,
                 'plot_args': [],
                 'plot_kwargs': {},
                 'markdown': '',
                 '_type': None}
        self.assertEqual(dict_, eval_.__dict__)

    def test_Evaluation_type(self):
        name = 'test'
        eval_ = self.init_noreg(name=name,
                                func='poisson_evaluations.number_test')
        self.assertEqual(['Absolute', 'Discrete'], eval_.type)
        eval_ = self.init_noreg(name=name,
                                func='w_test',
                                ref_model='SSSSSM')
        self.assertTrue(eval_.is_type('Discrete'))
        self.assertTrue(eval_.is_type('Comparative'))

        eval_ = self.init_noreg(name=name,
                                func='sequential_information_gain',
                                ref_model='SuperETAS')
        self.assertTrue(eval_.is_type('sequential'))
        self.assertTrue(eval_.is_type('comparative'))

        with self.assertRaises(TypeError) as e_:
            eval_ = self.init_noreg(name=name,
                                    func='w_test',
                                    ref_model=None)
            eval_.is_type('Comparative')

    def test_prepargs_abs_disc(self):
        pass
        # forecast = Model[timewindow]
        # reg = forecast.region
        # cat = load(catpath)
        # cat.region = reg
        #
        # return (forecast: csepfore, cat:csepcat)
        # pass

    def test_prepargs_comp_disc(self):
        pass
        # forecast = Model[timewindow]
        # reg = forecast.region
        # cat = load(catpath)
        # cat.region = reg
        # ref_forecast = ref_model[time_window]
        #
        # return (forecast: csepfore, ref_forecast:csepfore, cat:csepcat)

    def test_prepargs_batch_disc(self):
        pass
        # forecasts = [Model[timewindow] for MOdel in MOdels]
        # reg = forecast.region
        # cat = load(catpath)
        # cat.region = reg
        # ref_forecast = ref_model[time_window]
        #
        # return (forecast: List[csepfore], ref_forecast:csepfore, cat:csepcat)

    def test_prepargs_abs_seq(self):
        pass

    def test_prepargs_comp_seq(self):
        pass

    def test_discrete_args(self):
        pass

    def test_sequential_args(self):
        pass

    def test_write_result(self):
        pass

    def to_dict(self):
        pass

    def test_2str(self):
        pass

    @classmethod
    def tearDownClass(cls) -> None:
        pass
