import json
from datetime import datetime
from typing import List, Dict, Callable, Union

import csep.models
from csep.core.catalogs import CSEPCatalog
from fecsep.model import Model
from fecsep.utils import parse_csep_func


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
        Class:
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
        'number_test': ['Test', 'Absolute', 'Discrete'],
        'spatial_test': ['Test', 'Absolute', 'Discrete'],
        'magnitude_test': ['Test', 'Absolute', 'Discrete'],
        'likelihood_test': ['Test', 'Absolute', 'Discrete'],
        'conditional_likelihood_test': ['Test', 'Absolute', 'Discrete'],
        'negative_binomial_number_test': ['Test', 'Absolute', 'Discrete'],
        'binary_spatial_test': ['Test', 'Absolute', 'Discrete'],
        'binomial_spatial_test': ['Test', 'Absolute', 'Discrete'],
        'brier_score': ['Score', 'Absolute', 'Discrete'],
        'binary_conditional_likelihood_test': ['Test', 'Absolute', 'Discrete'],
        'paired_t_test': ['Test', 'Comparative', 'Discrete'],
        'w_test': ['Test', 'Comparative', 'Discrete'],
        'binary_paired_t_test': ['Test', 'Comparative', 'Discrete'],
        'vector_poisson_t_w_test': ['Test', 'Batch', 'Discrete'],
        'sequential_likelihood': ['Score', 'Absolute', 'Sequential'],
        'sequential_information_gain': ['Score', 'Comparative', 'Sequential']
    }

    def __init__(self, name: str, func: Union[str, Callable],
                 func_args: List = None, func_kwargs: Dict = None,
                 ref_model: (str, Model) = None, plot_func: Callable = None,
                 plot_args: List = None, plot_kwargs: Dict = None,
                 markdown: str = '') -> None:

        self.name = name

        self.func = parse_csep_func(func)
        self.func_args = func_args
        self.func_kwargs = func_kwargs  # todo set default args from exp?
        self.ref_model = ref_model

        self.plot_func = parse_csep_func(plot_func)
        self.plot_args = plot_args or {}  # todo default args from exp?
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

    @staticmethod
    def discrete_args(time_window, cat_path, model, ref_model=None):

        if isinstance(model, Model):
            forecast = model.forecasts[time_window]  # One Forecast
            reg = forecast.region
        else:
            forecast = [model_i.forecasts[time_window] for model_i in model]
            reg = forecast[0].region

        catalog = CSEPCatalog.load_json(cat_path)  # One Catalog

        catalog.region = reg  # F.Reg > Cat.Reg

        if isinstance(ref_model, Model):
            ref_forecast = ref_model.forecasts[time_window]  # REF.FORECAST
            test_args = (forecast, ref_forecast, catalog)  # Args: (Fc, Cat)
        else:
            test_args = (forecast, catalog)
        return test_args

    @staticmethod
    def sequential_args(time_windows, cat_paths, model, ref_model=None):
        forecasts = [model.forecasts[i] for i in time_windows]
        catalogs = [CSEPCatalog.load_json(i) for i in cat_paths]
        for i in catalogs:
            i.region = forecasts[0].region
        if ref_model:
            ref_forecasts = [ref_model.forecasts[i] for i in time_windows]
            test_args = (forecasts, ref_forecasts, catalogs, time_windows)
        else:
            test_args = (forecasts, catalogs, time_windows)
        return test_args

    @prepare_test_args
    def compute(self, time_window: Union[List[datetime], List[List[datetime]]],
                cat_path: str, model: Union[Model, List[Model]],
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

        test_args = None
        if 'Discrete' in self.type:
            test_args = self.discrete_args(time_window, cat_path,
                                           model, ref_model)
        elif 'Sequential' in self.type:
            test_args = self.sequential_args(time_window, cat_path,
                                             model, ref_model)

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
