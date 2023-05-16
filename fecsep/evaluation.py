import json
import numpy
from typing import Dict, Callable, Union, Sequence, List

import csep.models
from csep.core.catalogs import CSEPCatalog
from csep.core.forecasts import GriddedForecast

from fecsep.model import Model
from fecsep.utils import parse_csep_func


class Evaluation:
    """

    Class representing a Scoring Test, which wraps the evaluation function,
    its arguments, parameters and hyper-parameters.

    Args:
        name (str): Name of the Test
        func (str, ~typing.Callable): Test function/callable
        func_kwargs (dict): Keyword arguments of the test function
        ref_model (str): String of the reference model, if any
        plot_func (str, ~typing.Callable): Test's plotting function
        plot_args (list): Positional arguments of the plotting function
        plot_kwargs (dict): Keyword arguments of the plotting function

    """

    _TYPES = {
        'number_test': 'consistency',
        'spatial_test': 'consistency',
        'magnitude_test': 'consistency',
        'likelihood_test': 'consistency',
        'conditional_likelihood_test': 'consistency',
        'negative_binomial_number_test': 'consistency',
        'binary_spatial_test': 'consistency',
        'binomial_spatial_test': 'consistency',
        'brier_score': 'consistency',
        'binary_conditional_likelihood_test': 'consistency',
        'paired_t_test': 'comparative',
        'w_test': 'comparative',
        'binary_paired_t_test': 'comparative',
        'vector_poisson_t_w_test': 'batch',
        'sequential_likelihood': 'sequential',
        'sequential_information_gain': 'sequential_comparative'
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
        self.plot_args = plot_args or {}      # todo default args from exp?
        self.plot_kwargs = plot_kwargs or {}

        self.markdown = markdown
        self.type = Evaluation._TYPES.get(self.func.__name__)

    @property
    def type(self):
        """
        Returns the type of the test, mapped from the class attribute
        Evaluation._TYPES
        """
        return self._type

    @type.setter
    def type(self, type_list: Union[str, Sequence[str]]):
        if isinstance(type_list, Sequence):
            if ('Comparative' in type_list) and (self.ref_model is None):
                raise TypeError('A comparative-type test should have a'
                                ' reference model assigned')

        self._type = type_list

    def prepare_args(self,
                     timewindow: Union[str, list],
                     catpath: Union[str, list],
                     model: Union[Model, Sequence[Model]],
                     ref_model: Union[Model, Sequence] = None,
                     region = None ) -> tuple:
        """

        Prepares the positional argument for the Evaluation function.

        Args:
            timewindow (str, list): Time window string (or list of str)
             formatted from :meth:`fecsep.utils.timewindow2str`
            catpath (str,list): Path(s) pointing to the filtered catalog(s)
            model (:class:`fecsep:model.Model`): Model to be evaluated
            ref_model (:class:`fecsep:model.Model`, list): Reference model (or
             models) reference for the evaluation.

        Returns:
            A tuple of the positional arguments required by the evaluation
            function :meth:`Evaluation.func`.

        """
        # Subtasks
        # ========
        # Get forecast from model
        # Read Catalog
        # Share forecast region with catalog
        # Check if ref_model is None, Model or List[Model]
        # Prepare argument tuple

        forecast = model.get_forecast(timewindow, region)
        catalog = self.get_catalog(catpath, forecast)

        if isinstance(ref_model, Model):
            # Args: (Fc, RFc, Cat)
            ref_forecast = ref_model.get_forecast(timewindow, region)
            test_args = (forecast, ref_forecast, catalog)
        elif isinstance(ref_model, list):
            # Args: (Fc, [RFc], Cat)
            ref_forecasts = [i.get_forecast(timewindow, region)
                             for i in ref_model]
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

        Reads the catalog(s) from the given path(s). References the catalog
        region to the forecast region.

        Args:
            catalog_path (str, list(str)): Path to the existing catalog
            forecast (:class:`~csep.core.forecasts.GriddedForecast`): Forecast
             object, onto which the catalog will be confronted for testing.

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
                ref_model: Union[Model, Sequence[Model]] = None,
                region=None) -> None:
        """

        Runs the test, structuring the arguments according to the
         test-typology/function-signature

        Args:
            timewindow (list[~datetime.datetime, ~datetime.datetime]): Pair of
             datetime objects representing the testing time span
            catalog (str):  Path to the filtered catalog
            model (Model, list[Model]): Model(s) to be evaluated
            ref_model: Model to be used as reference
            path: Path to store the Evaluation result

        Returns:

        """
        test_args = self.prepare_args(timewindow,
                                      catpath=catalog,
                                      model=model,
                                      ref_model=ref_model,
                                      region=region)

        evaluation_result = self.func(*test_args, **self.func_kwargs)
        self.write_result(evaluation_result, path)

    @staticmethod
    def write_result(result: csep.models.EvaluationResult,
                     path: str) -> None:
        """
        Dumps a test result into a json file.
        """

        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, numpy.integer):
                    return int(obj)
                if isinstance(obj, numpy.floating):
                    return float(obj)
                if isinstance(obj, numpy.ndarray):
                    return obj.tolist()
                return json.JSONEncoder.default(self, obj)

        with open(path, 'w') as _file:
            json.dump(result.to_dict(), _file, indent=4, cls=NumpyEncoder)

    def to_dict(self) -> dict:
        """
        Represents an Evaluation instance as a dictionary, which can be
        serialized and then parsed
        """
        out = {}
        included = ['model', 'ref_model', 'func_kwargs',
                    'plot_args', 'plot_kwargs']
        for k, v in self.__dict__.items():
            if k in included and v:
                out[k] = v
        func_str = f'{self.func.__module__}.{self.func.__name__}'
        plotfunc_str = f'{self.plot_func.__module__}.{self.plot_func.__name__}'
        return {self.name: {**out, 'func': func_str,
                            'plot_func': plotfunc_str}}

    def __str__(self):
        return (
            f"name: {self.name}\n"
            f"function: {self.func.__name__}\n"
            f"reference model: {self.ref_model}\n"
            f"kwargs: {self.func_kwargs}\n"
        )

    @classmethod
    def from_dict(cls, record):
        """
        Parses a dictionary and re-instantiate an Evaluation object
        """
        if len(record) != 1:
            raise IndexError('A single test has not been passed')
        name = next(iter(record))
        return cls(name=name, **record[name])
