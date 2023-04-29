import unittest
from typing import Sequence, List
from fecsep.evaluation import Evaluation
from csep.core.forecasts import GriddedForecast
from csep.core.catalogs import CSEPCatalog


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
                 'plot_args': {},
                 'plot_kwargs': {},
                 'markdown': '',
                 '_type': None}
        self.assertEqual(dict_, eval_.__dict__)

    def test_func_signature(self):
        """ Test Evaluation.func signature catch """
        name = 'test'
        eval_ = self.init_noreg(name=name,
                                func='poisson_evaluations.number_test')
        expected = [GriddedForecast, CSEPCatalog]
        self.assertEqual(expected, eval_.func_signature)

        eval_ = self.init_noreg(name=name,
                                ref_model='None',
                                func='poisson_evaluations.paired_t_test')
        expected = [GriddedForecast, GriddedForecast, CSEPCatalog]
        self.assertEqual(expected, eval_.func_signature)

        eval_ = self.init_noreg(name=name,
                                ref_model='None',
                                func='vector_poisson_t_w_test')
        expected = [GriddedForecast, Sequence[GriddedForecast], CSEPCatalog]
        self.assertEqual(expected, eval_.func_signature)

        eval_ = self.init_noreg(name=name,
                                ref_model='None',
                                func='sequential_information_gain')
        expected = [Sequence[GriddedForecast], Sequence[GriddedForecast],
                    Sequence[CSEPCatalog]]
        self.assertEqual(expected, eval_.func_signature)

        eval_ = self.init_noreg(name=name,
                                ref_model='None',
                                func='sequential_likelihood')
        expected = [Sequence[GriddedForecast], Sequence[CSEPCatalog]]
        self.assertEqual(expected, eval_.func_signature)

        def test_func(a: CSEPCatalog, b: str, c: List[GriddedForecast]):
            print(a, b, c)

        eval_ = self.init_noreg(name=name, func=test_func)
        expected = [CSEPCatalog, str, List[GriddedForecast]]
        self.assertEqual(expected, eval_.func_signature)

    def test_discrete_args(self):
        pass

    def test_sequential_args(self):
        pass

    def test_prepare_catalog(self):
        from unittest.mock import MagicMock, Mock, patch

        def read_cat(_):
            cat = Mock()
            cat.name = 'csep'
            return cat

        with patch('csep.core.catalogs.CSEPCatalog.load_json', read_cat):
            region = 'CSEPRegion'
            forecast = MagicMock(name='forecast', region=region)

            catt = Evaluation.get_catalog('path_to_cat', forecast)
            self.assertEqual('csep', catt.name)
            self.assertEqual(region, catt.region)

            region2 = 'definitelyNotCSEPregion'
            forecast2 = Mock(name='forecast', region=region2)
            cats = Evaluation.get_catalog(['path1', 'path2'],
                                          [forecast, forecast2])

            self.assertIsInstance(cats, list)
            self.assertEqual(cats[0].name, 'csep')
            self.assertEqual(cats[0].region, 'CSEPRegion')
            self.assertEqual(cats[1].region, 'definitelyNotCSEPregion')

            with self.assertRaises(AttributeError):
                Evaluation.get_catalog('path1', [forecast, forecast2])
            with self.assertRaises(IndexError):
                Evaluation.get_catalog(['path1', 'path2'],
                                       forecast)
        assert True

    def test_write_result(self):
        pass

    def to_dict(self):
        pass

    @classmethod
    def tearDownClass(cls) -> None:
        pass
