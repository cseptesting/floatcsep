import json
from inspect import signature, Parameter
from datetime import datetime
from typing import Dict, Callable, Union, Sequence, List

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
        # todo deprecate
        return self._type

    @type.setter
    def type(self, type_list: Union[str, Sequence[str]]):
        # todo deprecate
        if isinstance(type_list, Sequence):
            if ('Comparative' in type_list) and (self.ref_model is None):
                raise TypeError('A comparative-type test should have a'
                                ' reference model assigned')

        self._type = type_list

    def is_type(self, test_type: str):
        return (test_type in self.type) or (
                test_type in [i.lower() for i in self.type])

    def prepare_args(self,
                     timewindow: Union[str, list],
                     catpath: Union[str, list],
                     model: Union[Model, Sequence[Model]],
                     ref_model: Union[Model, Sequence] = None) -> tuple:
        """

        Prepares the positional argument for the Evaluation function.

        Args:
            timewindow (str/list): Time window string (or list of str)
             formatted from `fecsep.utils.timewindow2str`
            catpath (str/list): Path(s) pointing to the filtered catalog (s)
            model (:class:`fecsep:model.Model`): Model to be evaluated
            ref_model (:class:`fecsep:model.Model`, list): Model (or models)
             reference for the evaluation.

        Returns:
            A tuple of the positional arguments required by the evaluation
            function `Evaluation.func`.

        """
        # Subtasks
        # ========
        # Get forecast from model
        # Read Catalog
        # Share forecast region with catalog
        # Check if ref_model is None, Model or List[Model]
        # Prepare argument tuple

        forecast = model.get_forecast(timewindow)
        catalog = self.get_catalog(catpath, forecast)

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

    @staticmethod
    def get_catalog(
            catalog_path: Union[str, Sequence[str]],
            forecast: Union[GriddedForecast, Sequence[GriddedForecast]]
    ) -> Union[CSEPCatalog, List[CSEPCatalog]]:
        """

        Reads the catalog(s) from given path(s). Reference the catalog region
        to the forecast region.

        Args:
            catalog_path (str, list(str)): Path to the existing catalog
            forecast (:class:`~csep.core.forecasts.GriddedForecast`): Forecast
             object, onto which the catalog will be confronted.

        Returns:

        """
        if isinstance(catalog_path, str):
            eval_cat = CSEPCatalog.load_json(catalog_path)
            eval_cat.region = getattr(forecast, 'region')
        else:
            eval_cat = [CSEPCatalog.load_json(i) for i in catalog_path]
            if (len(forecast) != len(eval_cat)) or (not isinstance(forecast,
                                                                   Sequence)):
                raise IndexError('Amount of passed catalogs and forecats must '
                                 'be the same')
            for cat, fc in zip(eval_cat, forecast):
                cat.region = getattr(fc, 'region', None)

        return eval_cat

    def compute(self,
                timewindow: Union[str, list],
                catalog: str,
                model: Model,
                path: str,
                ref_model: Union[Model, Sequence[Model]] = None) -> None:
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
                                      catpath=catalog,
                                      model=model,
                                      ref_model=ref_model)

        evaluation_result = self.func(*test_args, **self.func_kwargs)
        print(evaluation_result)
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
