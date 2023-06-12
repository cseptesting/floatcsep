import os
import json
import numpy
from matplotlib import pyplot
from typing import Dict, Callable, Union, Sequence, List

from csep.core.catalogs import CSEPCatalog
from csep.core.forecasts import GriddedForecast
from csep.models import EvaluationResult

from floatcsep.model import Model
from floatcsep.utils import parse_csep_func, timewindow2str
from floatcsep.registry import PathTree


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
        plot_args (list,dict): Positional arguments of the plotting function
        plot_kwargs (list,dict): Keyword arguments of the plotting function

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
        'paired_ttest_point_process': 'comparative',
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

        self.plot_func = None
        self.plot_args = None
        self.plot_kwargs = None

        self.parse_plots(plot_func, plot_args, plot_kwargs)

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

    def parse_plots(self, plot_func, plot_args, plot_kwargs):

        if isinstance(plot_func, str):

            self.plot_func = [parse_csep_func(plot_func)]
            self.plot_args = [plot_args] if plot_args else [{}]
            self.plot_kwargs = [plot_kwargs] if plot_kwargs else [{}]

        elif isinstance(plot_func, (list, dict)):
            if isinstance(plot_func, dict):
                plot_func = [{i: j} for i, j in plot_func.items()]

            if plot_args is not None or plot_kwargs is not None:
                raise ValueError('If multiple plot functions are passed,'
                                 'each func should be a dictionary with '
                                 'plot_args and plot_kwargs passed as '
                                 'dictionaries beneath each func.')

            func_names = [list(i.keys())[0] for i in plot_func]
            self.plot_func = [parse_csep_func(func) for func in func_names]
            self.plot_args = [i[j].get('plot_args', {})
                              for i, j in zip(plot_func, func_names)]
            self.plot_kwargs = [i[j].get('plot_kwargs', {})
                                for i, j in zip(plot_func, func_names)]


    def prepare_args(self,
                     timewindow: Union[str, list],
                     catpath: Union[str, list],
                     model: Union[Model, Sequence[Model]],
                     ref_model: Union[Model, Sequence] = None,
                     region = None) -> tuple:
        """

        Prepares the positional argument for the Evaluation function.

        Args:
            timewindow (str, list): Time window string (or list of str)
             formatted from :meth:`floatcsep.utils.timewindow2str`
            catpath (str,list): Path(s) pointing to the filtered catalog(s)
            model (:class:`floatcsep:model.Model`): Model to be evaluated
            ref_model (:class:`floatcsep:model.Model`, list): Reference model (or
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
            region: region to filter a catalog forecast.

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
    def write_result(result: EvaluationResult,
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

    def read_results(self, window: str, models: List[Model],
                     tree: PathTree) -> List:
        """
        Reads an Evaluation result for a given time window and returns a list
        of the results for all tested models.
        """
        test_results = []

        if not isinstance(window, str):
            wstr_ = timewindow2str(window)
        else:
            wstr_ = window

        for i in models:
            eval_path = tree(wstr_, 'evaluations', self, i.name)
            with open(eval_path, 'r') as file_:
                model_eval = EvaluationResult.from_dict(json.load(file_))
            test_results.append(model_eval)

        return test_results

    def plot_results(self,
                     timewindow: Union[str, List],
                     models: List[Model],
                     tree: PathTree,
                     dpi: int = 300,
                     show: bool = False) -> None:
        """

        Plots all evaluation results

        Args:
            dpi: Figure resolution with which to save
            show: show in runtime

        """
        if isinstance(timewindow, str):
            timewindow = [timewindow]

        for func, fargs, fkwargs in zip(self.plot_func, self.plot_args,
                                        self.plot_kwargs):
            if self.type in ['consistency', 'comparative']:

                try:
                    for time_str in timewindow:
                        fig_path = tree(time_str, 'figures', self.name)
                        results = self.read_results(time_str, models, tree)
                        ax = func(results, plot_args=fargs, **fkwargs)
                        if 'code' in fargs:
                            exec(fargs['code'])
                        pyplot.savefig(fig_path, dpi=dpi)
                        if show:
                            pyplot.show()

                except AttributeError as msg:
                    if self.type in ['consistency', 'comparative']:
                        for time_str in timewindow:
                            results = self.read_results(time_str, models, tree)
                            for result, model in zip(results, models):
                                fig_name = f'{self.name}_{model.name}'

                                tree.paths[time_str]['figures'][fig_name] =\
                                    os.path.join(time_str, 'figures', fig_name)
                                fig_path = tree(time_str, 'figures', fig_name)
                                ax = func(result, plot_args=fargs, **fkwargs,
                                          show=False)
                                if 'code' in fargs:
                                    exec(fargs['code'])
                                pyplot.savefig(fig_path, dpi=dpi)
                                if show:
                                    pyplot.show()

            elif self.type in ['sequential', 'sequential_comparative', 'batch']:
                fig_path = tree(timewindow[-1], 'figures', self.name)
                results = self.read_results(timewindow[-1], models, tree)
                ax = func(results, plot_args=fargs, **fkwargs)

                if 'code' in fargs:
                    exec(fargs['code'])
                pyplot.savefig(fig_path, dpi=dpi)
                if show:
                    pyplot.show()

    def as_dict(self) -> dict:
        """
        Represents an Evaluation instance as a dictionary, which can be
        serialized and then parsed
        """
        out = {}
        included = ['model', 'ref_model', 'func_kwargs']
        for k, v in self.__dict__.items():
            if k in included and v:
                out[k] = v
        func_str = f'{self.func.__module__}.{self.func.__name__}'

        plot_func_str = []
        for i, j, k in zip(self.plot_func, self.plot_args, self.plot_kwargs):
            pfunc = {f'{i.__module__}.{i.__name__}': {'plot_args': j,
                                                      'plot_kwargs': k}}
            plot_func_str.append(pfunc)

        return {self.name: {**out,
                            'func': func_str,
                            'plot_func': plot_func_str}}

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
