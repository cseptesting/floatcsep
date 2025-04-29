import datetime
import os
from typing import Dict, Callable, Union, Sequence, List, Any

from csep.core.catalogs import CSEPCatalog
from csep.core.forecasts import GriddedForecast
from matplotlib import pyplot

from floatcsep.model import Model
from floatcsep.infrastructure.registries import ExperimentRegistry
from floatcsep.utils.helpers import parse_csep_func


class Evaluation:
    """
    Class representing a Scoring Test, which wraps the evaluation function, its arguments,
    parameters and hyperparameters.

    Args:
        name (str): Name of the Test.
        func (str, ~typing.Callable): Test function/callable.
        func_kwargs (dict): Keyword arguments of the test function.
        ref_model (str): String of the reference model, if any.
        plot_func (str, ~typing.Callable): Test's plotting function.
        plot_args (list,dict): Positional arguments of the plotting function.
        plot_kwargs (list,dict): Keyword arguments of the plotting function.
        markdown (str): The caption to be placed beneath the result figure.
    """

    _TYPES = {
        "number_test": "consistency",
        "spatial_test": "consistency",
        "magnitude_test": "consistency",
        "likelihood_test": "consistency",
        "conditional_likelihood_test": "consistency",
        "negative_binomial_number_test": "consistency",
        "binary_spatial_test": "consistency",
        "binomial_spatial_test": "consistency",
        "brier_score": "consistency",
        "binary_conditional_likelihood_test": "consistency",
        "paired_t_test": "comparative",
        "paired_ttest_point_process": "comparative",
        "w_test": "comparative",
        "binary_paired_t_test": "comparative",
        "vector_poisson_t_w_test": "batch",
        "sequential_likelihood": "sequential",
        "sequential_information_gain": "sequential_comparative",
    }

    def __init__(
        self,
        name: str,
        func: Union[str, Callable],
        func_kwargs: Dict = None,
        ref_model: (str, Model) = None,
        plot_func: Callable = None,
        plot_args: Sequence = None,
        plot_kwargs: Dict = None,
        markdown: str = "",
    ) -> None:

        self.name = name

        self.func = parse_csep_func(func)
        self.func_kwargs = func_kwargs or {}
        self.ref_model = ref_model

        self.plot_func = None
        self.plot_args = None
        self.plot_kwargs = None

        self.parse_plots(plot_func, plot_args, plot_kwargs)

        self.markdown = markdown
        self.type = Evaluation._TYPES.get(self.func.__name__)

        self.results_repo = None
        self.catalog_repo = None

    @property
    def type(self):
        """
        Returns the type of the test, mapped from the class attribute Evaluation._TYPES.
        """
        return self._type

    @type.setter
    def type(self, type_list: Union[str, Sequence[str]]):
        if isinstance(type_list, Sequence):
            if ("Comparative" in type_list) and (self.ref_model is None):
                raise TypeError(
                    "A comparative-type test should have a" " reference model assigned"
                )

        self._type = type_list

    def parse_plots(
        self,
        plot_func: Any,
        plot_args: Any,
        plot_kwargs: Any,
    ) -> None:
        """
        It parses the plot function(s) and its(their) arguments from the test configuration
        file. The plot function can belong to :mod:`csep.utils.plots` or a custom function.
        Each plotting function is parsed by using the function
        :func:`~floatcsep.utils.helpers.parse_csep_function`, and assigned to its respective
        `args` and `kwargs`

        Args:
            plot_func: The name of the plotting function
            plot_args: The arguments of the plotting function
            plot_kwargs: The keyword arguments of the plotting function


        """
        if isinstance(plot_func, str):

            self.plot_func = [parse_csep_func(plot_func)]
            self.plot_args = [plot_args] if plot_args else [{}]
            self.plot_kwargs = [plot_kwargs] if plot_kwargs else [{}]

        elif isinstance(plot_func, (list, dict)):
            if isinstance(plot_func, dict):
                plot_func = [{i: j} for i, j in plot_func.items()]

            if plot_args is not None or plot_kwargs is not None:
                raise ValueError(
                    "If multiple plot functions are passed,"
                    "each func should be a dictionary with "
                    "plot_args and plot_kwargs passed as "
                    "dictionaries beneath each func."
                )

            func_names = [list(i.keys())[0] for i in plot_func]
            self.plot_func = [parse_csep_func(func) for func in func_names]
            self.plot_args = [i[j].get("plot_args", {}) for i, j in zip(plot_func, func_names)]
            self.plot_kwargs = [
                i[j].get("plot_kwargs", {}) for i, j in zip(plot_func, func_names)
            ]

    def prepare_args(
        self,
        timewindow: Union[str, list],
        model: Union[Model, Sequence[Model]],
        ref_model: Union[Model, Sequence] = None,
        region=None,
    ) -> tuple:
        """
        Prepares the positional argument for the Evaluation function.

        Args:
            timewindow (str, list): Time window string (or list of str)
             formatted from :meth:`floatcsep.utils.timewindow2str`
            model (:class:`floatcsep:model.Model`): Model to be evaluated
            ref_model (:class:`floatcsep:model.Model`, list): Reference model (or
             models) reference for the evaluation.
            region (:class:`csep:core.regions.CartesianGrid2D`): Experiment region

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
        catalog = self.get_catalog(timewindow, forecast)

        if isinstance(ref_model, Model):
            # Args: (Fc, RFc, Cat)
            ref_forecast = ref_model.get_forecast(timewindow, region)
            test_args = (forecast, ref_forecast, catalog)
        elif isinstance(ref_model, list):
            # Args: (Fc, [RFc], Cat)
            ref_forecasts = [i.get_forecast(timewindow, region) for i in ref_model]
            test_args = (forecast, ref_forecasts, catalog)
        else:
            # Args: (Fc, Cat)
            test_args = (forecast, catalog)

        return test_args

    def get_catalog(
        self,
        timewindow: Union[str, Sequence[str]],
        forecast: Union[GriddedForecast, Sequence[GriddedForecast]],
    ) -> Union[CSEPCatalog, List[CSEPCatalog]]:
        """
        Reads the catalog(s) from the given path(s). References the catalog region to the
        forecast region.

        Args:
            timewindow (str): Time window of the testing catalog
            forecast (:class:`~csep.core.forecasts.GriddedForecast`): Forecast
             object, onto which the catalog will be confronted for testing.

        Returns:
        """

        if isinstance(timewindow, str):
            # eval_cat = CSEPCatalog.load_json(catalog_path)
            eval_cat = self.catalog_repo.get_test_cat(timewindow)
            eval_cat.region = getattr(forecast, "region")

        else:
            eval_cat = [self.catalog_repo.get_test_cat(i) for i in timewindow]
            if (len(forecast) != len(eval_cat)) or (not isinstance(forecast, Sequence)):
                raise IndexError("Amount of passed catalogs and forecasts must " "be the same")
            for cat, fc in zip(eval_cat, forecast):
                cat.region = getattr(fc, "region", None)

        return eval_cat

    def compute(
        self,
        timewindow: Union[str, list],
        model: Model,
        ref_model: Union[Model, Sequence[Model]] = None,
        region=None,
    ) -> None:
        """
        Runs the test, structuring the arguments according to the
        test-typology/function-signature

        Args:
            timewindow (list[~datetime.datetime, ~datetime.datetime]): A pair of datetime
             objects representing the testing time span
            catalog (str):  Path to the filtered catalog
            model (Model, list[Model]): Model(s) to be evaluated
            ref_model: Model to be used as reference
            region: region to filter a catalog forecast.

        Returns:
        """
        test_args = self.prepare_args(
            timewindow, model=model, ref_model=ref_model, region=region
        )

        evaluation_result = self.func(*test_args, **self.func_kwargs)

        if self.type in ["sequential", "sequential_comparative"]:
            self.results_repo.write_result(evaluation_result, self, model, timewindow[-1])
        else:
            self.results_repo.write_result(evaluation_result, self, model, timewindow)

    def read_results(
        self, window: Union[str, Sequence[datetime.datetime]], models: Union[Model, List[Model]]
    ) -> List:
        """
        Reads an Evaluation result for a given time window and returns a list of the results for
        all tested models.
        """

        test_results = self.results_repo.load_results(self, window, models)

        return test_results

    def plot_results(
        self,
        timewindow: Union[str, List],
        models: List[Model],
        registry: ExperimentRegistry,
        dpi: int = 300,
        show: bool = False,
    ) -> None:
        """
        Plots all evaluation results.

        Args:
            timewindow: string representing the desired timewindow to plot
            models: a list of :class:`floatcsep:models.Model`
            registry: a :class:`floatcsep:models.PathTree` containing path of the results
            dpi: Figure resolution with which to save
            show: show in runtime
        """
        if isinstance(timewindow, str):
            timewindow = [timewindow]

        for func, fargs, fkwargs in zip(self.plot_func, self.plot_args, self.plot_kwargs):
            if self.type in ["consistency", "comparative"]:
                # Regular consistency/comparative test plots (e.g., many models)
                try:
                    for time_str in timewindow:
                        fig_path = registry.get_figure_key(time_str, self.name)
                        results = self.read_results(time_str, models)
                        ax = func(results, plot_args=fargs, **fkwargs)
                        if "code" in fargs:
                            exec(fargs["code"])
                        pyplot.savefig(fig_path, dpi=dpi)
                        if show:
                            pyplot.show()
                # Single model test plots (e.g., test distribution)
                # todo: handle this more elegantly
                except AttributeError:
                    if self.type in ["consistency", "comparative"]:
                        for time_str in timewindow:
                            results = self.read_results(time_str, models)
                            for result, model in zip(results, models):
                                fig_name = f"{self.name}_{model.name}"

                                registry.figures[time_str][fig_name] = os.path.join(
                                    time_str, "figures", fig_name
                                )
                                fig_path = registry.get_figure_key(time_str, fig_name)
                                ax = func(result, plot_args=fargs, **fkwargs, show=False)
                                if "code" in fargs:
                                    exec(fargs["code"])
                                fig = ax.get_figure()
                                fig.savefig(fig_path, dpi=dpi)

                                if show:
                                    pyplot.show()

            elif self.type in ["sequential", "sequential_comparative", "batch"]:
                fig_path = registry.get_figure_key(timewindow[-1], self.name)
                results = self.read_results(timewindow[-1], models)
                ax = func(results, plot_args=fargs, **fkwargs)

                if "code" in fargs:
                    exec(fargs["code"])
                pyplot.savefig(fig_path, dpi=dpi)
                if show:
                    pyplot.show()

    def as_dict(self) -> dict:
        """
        Represents an Evaluation instance as a dictionary, which can be serialized and then
        parsed
        """
        out = {}
        included = ["model", "ref_model", "func_kwargs"]
        for k, v in self.__dict__.items():
            if k in included and v:
                out[k] = v
        func_str = f"{self.func.__module__}.{self.func.__name__}"

        plot_func_str = []
        for i, j, k in zip(self.plot_func, self.plot_args, self.plot_kwargs):
            pfunc = {f"{i.__module__}.{i.__name__}": {"plot_args": j, "plot_kwargs": k}}
            plot_func_str.append(pfunc)

        return {self.name: {**out, "func": func_str, "plot_func": plot_func_str}}

    def __str__(self):
        return (
            f"name: {self.name}\n"
            f"function: {self.func.__name__}\n"
            f"reference model: {self.ref_model}\n"
            f"kwargs: {self.func_kwargs}\n"
        )

    @classmethod
    def from_dict(cls, record):
        """Parses a dictionary and re-instantiate an Evaluation object."""
        if len(record) != 1:
            raise IndexError("A single test has not been passed")
        name = next(iter(record))
        return cls(name=name, **record[name])
