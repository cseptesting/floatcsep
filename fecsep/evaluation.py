import json
from inspect import signature, Parameter
from datetime import datetime
from functools import singledispatchmethod
from typing import Dict, Callable, Union, Sequence

import csep.models
from csep.core.catalogs import CSEPCatalog
from csep.core.forecasts import GriddedForecast, CatalogForecast

from fecsep.registry import register
from fecsep.model import Model
from fecsep.utils import parse_csep_func

_ARGTYPES = {GriddedForecast: ['forecast',
                               'gridded_forecast',
                               'gridded_forecast1',
                               'gridded_forecast2',
                               'benchmark_forecast'],
             CSEPCatalog: ['catalog', 'observed_catalog'],
             Sequence[GriddedForecast]: ['forecasts',
                                         'gridded_forecasts',
                                         'benchmark_forecasts'],
             Sequence[CSEPCatalog]: ['observed_catalogs']}


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

    @register
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

        self.type = Evaluation._TYPES.get(self.func.__name__)

    @property
    def func_signature(self):
        """

        Finds the Evaluation function signature (type of the arguments).
        From this, the Experiment class can (1) Identify which args must be
        passed to the Evaluation.compute() (2) Determine the Task structure
        logic.

        Returns:
            An list with the Types representing the function positional
            arguments' types.

        """

        args = [param for param in signature(self.func).parameters.values()
                if param.default == Parameter.empty]
        names = [a.name for a in args]
        annotations = [a.annotation for a in args]
        func_sign = []
        for n, a in zip(names, annotations):
            if a == Parameter.empty:
                try:
                    argtype = [i for i, k in _ARGTYPES.items() if n in k][0]
                except IndexError:
                    raise TypeError(
                        f"The argument '{n}' of function "
                        f"'{self.func.__name__}' has no type specified,"
                        f" and was not found in 'evaluation._ARGTYPES'")
            else:
                argtype = a
            func_sign.append(argtype)
        return func_sign

    @property
    def type(self):
        return self._type

    @type.setter
    def type(self, type_list: Union[str, Sequence[str]]):

        if isinstance(type_list, Sequence):
            if ('Comparative' in type_list) and (self.ref_model is None):
                raise TypeError('A comparative-type test should have a'
                                ' reference model assigned')

        self._type = type_list

    def is_type(self, test_type: str):
        return (test_type in self.type) or (
                test_type in [i.lower() for i in self.type])

    @singledispatchmethod
    def prepare_args(self, timewindow, **__) -> tuple:
        """
        Discerns between argument `timewindow` types and dispatchs
        Args:
            timewindow (str, list(datetime), list(str), list(list(datetime))
            **__: Remaining args to pass directly to func.

        Returns:
            The arguments to pass to the testing functions
        """
        raise NotImplementedError('Test type not implemented')

    @prepare_args.register
    def single_tspan(self,
                     timewindow: str,
                     catalog: str,
                     model: Union[Model, Sequence[Model]],
                     ref_model: Model = None) -> tuple:

        #### Subtasks
        # Read Catalog
        # Get forecast from model
        #   todo: TI should get from memory (since it was already created)
        #    so check that it doesn't read/scale the forecast again
        # Share forecast region with catalog
        # Check if ref_model is None, Model or List[Model]

        catalog = CSEPCatalog.load_json(catalog)
        forecast = model.get_forecast(timewindow)
        catalog.region = forecast.region  # Add ref from F.Reg -> Cat.Reg

        if isinstance(ref_model, Model):
            # Args: (Fc, RFc, Cat)
            ref_forecast = ref_model.get_forecast(timewindow)
            test_args = (forecast, ref_forecast, catalog)
        elif isinstance(ref_model, list):
            # Args: (Fc, [RFc], Cat)
            ref_forecasts = [i.get_forecast(timewindow) for i in ref_model]
            test_args = (forecast, ref_forecasts, catalog)
        else:
            # Args: (Fc, Cat)
            test_args = (forecast, catalog)

        return test_args

    @prepare_args.register
    def sequential_tspan(
            self,
            timewindow: list,
            catalog: Sequence[str],
            model: Sequence[Model],
            ref_model: Sequence[Model] = None) -> tuple:
        """
        Subtasks
         - Get forecast for each timewindow_i from model
         - Read Catalog for each timewindow_i
         - Share forecast_i region with catalog_i
         - Check if ref_model is None, Model or List[Model]

        Args:
            timewindow:
            catalog:
            model:
            ref_model:

        Returns:

        """
        forecasts = [model.get_forecast(i) for i in timewindow]
        catalogs = [CSEPCatalog.load_json(i) for i in catalog]

        for i, j in zip(catalogs, forecasts):
            i.region = j.region

        if isinstance(ref_model, Model):
            # Args: ([Fc_i], [RFc_i], [Cat_i])
            ref_forecasts = [ref_model.forecasts[i] for i in timewindow]
            test_args = (forecasts, ref_forecasts, catalogs, timewindow)
        else:
            # Args: ([Fc_i], [Cat_i])
            test_args = (forecasts, catalogs, timewindow)
        return test_args

    def compute(self,
                timewindow: Union[str, list],
                catalog: str,
                model: Union[Model, Sequence[Model]],
                path: str,
                ref_model: Model = None) -> None:
        """

        Runs the test, structuring the arguments according to the
         test-typology/function-signature

        Args:
            timewindow (list[datetime, datetime]): Pair of datetime objects
             representing the testing time span
            catalog (str):  Path to the filtered catalog
            model (Model, list[Model]): Model(s) to be evaluated
            ref_model: Model to be used as reference
            path: Path to store the Evaluation result

        Returns:

        """

        test_args = self.prepare_args(timewindow,
                                      catalog=catalog,
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
        included = ['name', 'model', 'func', 'ref_model', 'path',
                    'func_kwargs']
        for k, v in self.__dict__.items():
            if k in included and v:
                out[k] = v
        return out

    def __str__(self):
        return (
            f"name: {self.name}\n"
            f"function: {self.func.__name__}\n"
            f"reference model: {self.ref_model}\n"
            f"kwargs: {self.func_kwargs}\n"
        )

    @classmethod
    def from_dict(cls, record):
        if len(record) != 1:
            raise IndexError('A single test has not been passed')
        name = next(iter(record))
        return cls(name=name, **record[name])
