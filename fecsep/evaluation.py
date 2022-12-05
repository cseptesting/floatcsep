import json
from datetime import datetime
from typing import List, Dict, Callable, Union

import csep.models
from csep.core.catalogs import CSEPCatalog
from fecsep.model import Model
from fecsep.utils import parse_csep_func


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

    _types = {'consistency': ['number_test', 'spatial_test', 'magnitude_test',
                              'likelihood_test', 'conditional_likelihood_test',
                              'negative_binomial_number_test',
                              'binary_spatial_test', 'binomial_spatial_test',
                              'brier_score',
                              'binary_conditional_likelihood_test'],
              'comparative': ['paired_t_test', 'w_test',
                              'binary_paired_t_test'],
              'batch': ['vector_poisson_t_w_test'],
              'sequential': ['sequential_likelihood'],
              'seqcomp': ['sequential_information_gain']}
    _funcs = {key: f'compute_{key}' for key in _types.keys()}

    def __init__(self, name: str, func: Union[str, Callable],
                 func_args: List = None, func_kwargs: Dict = None,
                 plot_func: Callable = None, plot_args: List = None,
                 plot_kwargs: Dict = None) -> None:

        self.name = name

        self.func = parse_csep_func(func)
        self.func_args = func_args
        self.func_kwargs = func_kwargs  # todo set default args from exp?

        self.plot_func = parse_csep_func(plot_func)
        self.plot_args = plot_args or {}  # todo default args?
        self.plot_kwargs = plot_kwargs or {}

        self._type = None

    def compute(self, timewindow: List[datetime], catpath: str,
                model: Union[Model, List[Model]], path: str,
                ref_model: Model = None) -> None:
        """

        Runs the test, structuring the arguments according to the test typology

        Args:
            timewindow (list[datetime, datetime]): Pair of datetime objects
             representing the testing time span
            catpath (str):  Path to the filtered catalog
            model (Model, list[Model]): Model(s) to be evaluated
            ref_model: Model to be used as reference
            path: Path to store the Evaluation result

        Returns:

        """

        #  If Comparative  >>>> Ref_model
        #  If Sequential >>>>> Forecasts = [List of FC]
        #  If Batch   >>>> List of Models

        if self.type == 'consistency':
            forecast = model.forecasts[timewindow]  # One Forecast
            catalog = CSEPCatalog.load_json(catpath)  # One Catalog
            catalog.region = forecast.region  # F.Reg > Cat.Reg
            test_args = (forecast, catalog)  # Args: (Fc, Cat)

        if self.type == 'comparative':
            forecast = model.forecasts[timewindow]  # One Forecast
            catalog = CSEPCatalog.load_json(catpath)  # One Catalog
            catalog.region = forecast.region  # F.reg >  Cat.Reg
            ref_forecast = ref_model.forecasts[timewindow]  # REF.FORECAST
            test_args = (
                forecast, ref_forecast, catalog)  # Args: (Fc, RefFC, Cat)

        elif self.type == 'batch':
            ref_forecast = ref_model.forecasts[timewindow]  # One Ref. Forecast
            catalog = CSEPCatalog.load_json(catpath)  # One Catalog
            catalog.region = ref_forecast.region  # RefF.Reg >  Cat.reg
            forecast_batch = [model_i.forecasts[timewindow] for model_i in
                              # Multiple FCS
                              model]
            test_args = (ref_forecast, forecast_batch,
                         catalog)  # Args (RefFC, FCBatch, Cat)

        elif self.type == 'sequential':
            forecasts = [model.forecasts[i] for i in timewindow]
            catalogs = [CSEPCatalog.load_json(i) for i in catpath]
            for i in catalogs:
                i.region = forecasts[0].region
            test_args = (forecasts, catalogs, timewindow)

        elif self.type == 'seqcomp':
            forecasts = [model.forecasts[i] for i in timewindow]
            ref_forecasts = [ref_model.forecasts[i] for i in timewindow]
            catalogs = [CSEPCatalog.load_json(i) for i in catpath]
            for i in catalogs:
                i.region = forecasts[0].region
            test_args = (forecasts, ref_forecasts, catalogs, timewindow)

        self.write_result(self.func(*test_args, **self.func_kwargs))

    @staticmethod
    def write_result(self, result: csep.models.EvaluationResult,
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

    @property
    def type(self):

        self._type = None
        for ty, funcs in Evaluation._types.items():
            if self.func.__name__ in funcs:
                self._type = ty

        return self._type

    @classmethod
    def from_dict(cls, record):
        if len(record) != 1:
            raise IndexError('A single test has not been passed')
        name = next(iter(record))
        return cls(name=name, **record[name])
