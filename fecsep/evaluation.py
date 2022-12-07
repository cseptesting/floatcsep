import json
from datetime import datetime
from typing import Dict, Callable, Union, Sequence

import csep.models
from csep.core.catalogs import CSEPCatalog
from fecsep.model import Model
from fecsep.utils import parse_csep_func
from functools import singledispatchmethod


def prepare_test_args(func):
    def funcwrap(*args, **kwargs):
        if 'Sequential' in args[0].type:
            if not isinstance(kwargs.get('cat_path'), list):
                raise TypeError(
                    'A list of catalog paths should be provided'
                    ' for a sequential evaluation')
            if not isinstance(kwargs.get('time_window'), list):
                raise TypeError('A list of time_window pairs should be'
                                ' provided for a sequential evaluation')
        if 'Comparative' in args[0].type:
            if kwargs.get('ref_model') is None:
                raise TypeError('None is not valid as reference model')
        if 'Batch' in args[0].type:

            if not isinstance(kwargs.get('model'), list):
                raise TypeError('Model batch should be passed as a list of '
                                'models')
        return func(*args, **kwargs)

    return funcwrap


class Evaluation:
    """

    Class representing a Scoring Test, which wraps the evaluation function,
    its arguments, parameters and metaparameters.

    Args:
        name (str): Name of the Test
        func (str, ~collections.abc.Callable): Test function/callable
        func_args (list): Positional arguments of the test function
        func_kwargs (dict): Keyword arguments of the test function
        plot_func (str, ~collections.abc.Callable): Test's plotting function
        plot_args (list): Positional arguments of the plotting function
        plot_kwargs (dict): Keyword arguments of the plotting function

    """

    '''
    Evaluation Typology
        Class >> Not distinguished in fecsep:
            Score (Single metric)
            Test (Metric and p-val)
        Mapping:
            Absolute (Single model)
            Comparative (Relative to a model)
            Batch (Relative to a model batch)
        Temporality: 
            Discrete (Metric for the entire time span):
            Incremental (Metrics of sequential time windows):  
            Sequential: 
    '''

    # todo: Typology characterization should be done within pycsep
    _TYPES = {
        'number_test': ['Absolute', 'Discrete'],
        'spatial_test': ['Absolute', 'Discrete'],
        'magnitude_test': ['Absolute', 'Discrete'],
        'likelihood_test': ['Absolute', 'Discrete'],
        'conditional_likelihood_test': ['Absolute', 'Discrete'],
        'negative_binomial_number_test': ['Absolute', 'Discrete'],
        'binary_spatial_test': ['Absolute', 'Discrete'],
        'binomial_spatial_test': ['Absolute', 'Discrete'],
        'brier_score': ['Absolute', 'Discrete'],
        'binary_conditional_likelihood_test': ['Absolute', 'Discrete'],
        'paired_t_test': ['Comparative', 'Discrete'],
        'w_test': ['Comparative', 'Discrete'],
        'binary_paired_t_test': ['Comparative', 'Discrete'],
        'vector_poisson_t_w_test': ['Batch', 'Discrete'],
        'sequential_likelihood': ['Absolute', 'Sequential'],
        'sequential_information_gain': ['Comparative', 'Sequential']
    }

    def __init__(self, name: str, func: Union[str, Callable],
                 func_kwargs: Dict = None,
                 ref_model: (str, Model) = None,
                 plot_func: Callable = None,
                 plot_args: Sequence = None,
                 plot_kwargs: Dict = None,
                 markdown: str = '') -> None:

        self.name = name

        self.func = parse_csep_func(func)
        self.func_kwargs = func_kwargs or {}  # todo set default args from exp?
        self.ref_model = ref_model

        self.plot_func = parse_csep_func(plot_func)
        self.plot_args = plot_args or []  # todo default args from exp?
        self.plot_kwargs = plot_kwargs or {}

        self.markdown = markdown

        self.type = Evaluation._TYPES[self.func.__name__]

    @property
    def type(self):
        return self._type

    @type.setter
    def type(self, type_list):
        if 'Comparative' in type_list and self.ref_model is None:
            raise TypeError('A comparative-type test should have a'
                            ' reference model assigned')
        self._type = type_list

    def is_type(self, test_type: str):
        return (test_type in self.type) or (
                test_type in [i.lower() for i in self.type])

    @singledispatchmethod
    def prepare_args(self, timewindow, **__):

        # If arguments does not match dispatch patterns:
        raise NotImplementedError('Test type not implemented')

    @prepare_args.register
    def discrete_args(self, time_window: str, cat_path: str,
                      model: Union[Model, list],
                      ref_model: Model = None) -> tuple:

        catalog = CSEPCatalog.load_json(cat_path)
        if isinstance(model, Model):
            # Single Forecast
            forecast = model.forecasts[time_window]
            region = forecast.region
        else:
            # Forecast Batch
            forecast = [model_i.forecasts[time_window] for model_i in model]
            region = forecast[0].region

        # One Catalog
        catalog.region = region  # F.Reg -> Cat.Reg

        if isinstance(ref_model, Model):
            if self.is_type('Absolute'):
                raise AttributeError('Absolute/Single test does not require '
                                     'reference model')
            # Args: (Fc, RFc, Cat)
            ref_forecast = ref_model.forecasts[time_window]
            test_args = (forecast, ref_forecast, catalog)
        else:
            # Args: (Fc, Cat)
            test_args = (forecast, catalog)

        return test_args

    @prepare_args.register
    def sequential_args(self, time_windows: list, cat_path: list,
                        model: list, ref_model: list = None) -> tuple:
        forecasts = [model.forecasts[i] for i in time_windows]
        catalogs = [CSEPCatalog.load_json(i) for i in cat_path]

        for i in catalogs:
            i.region = forecasts[0].region

        # Comparative Model
        if isinstance(ref_model, Model) and self.is_type('Comparative'):
            # Args: ([Fc_i], [RFc_i], [Cat_i])
            ref_forecasts = [ref_model.forecasts[i] for i in time_windows]
            test_args = (forecasts, ref_forecasts, catalogs, time_windows)
        else:
            # Args: ([Fc_i], [Cat_i])
            test_args = (forecasts, catalogs, time_windows)
        return test_args

    @prepare_test_args
    def compute(self, time_window: Union[str, list],
                cat_path: str, model: Union[Model, Sequence[Model]],
                path: str, ref_model: Model = None) -> None:
        """

        Runs the test, structuring the arguments according to the test typology

        Args:
            time_window (list[datetime, datetime]): Pair of datetime objects
             representing the testing time span
            cat_path (str):  Path to the filtered catalog
            model (Model, list[Model]): Model(s) to be evaluated
            ref_model: Model to be used as reference
            path: Path to store the Evaluation result

        Returns:

        """

        test_args = self.prepare_args(time_window,
                                      cat_path=cat_path,
                                      model=model,
                                      ref_model=ref_model)

        evaluation_result = self.func(*test_args, **self.func_kwargs)
        self.write_result(evaluation_result, path)

    @staticmethod
    def write_result(result: csep.models.EvaluationResult,
                     path: str) -> None:
        with open(path, 'w') as _file:
            json.dump(result.to_dict(), _file, indent=4)

    def to_dict(self):
        out = {}
        included = ['name', 'model', 'ref_model', 'path', 'func_kwargs']
        for k, v in self.__dict__.items():
            if k in included and v is not None:
                out[k] = v
        return out

    def __str__(self):
        return (
            f"name: {self.name}\n"
            f"reference model: {self.ref_model}\n"
            f"kwargs: {self.func_kwargs}\n"
            f"path: {self.path}"
        )

    @classmethod
    def from_dict(cls, record):
        if len(record) != 1:
            raise IndexError('A single test has not been passed')
        name = next(iter(record))
        return cls(name=name, **record[name])
