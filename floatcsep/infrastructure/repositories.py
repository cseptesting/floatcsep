import datetime
import json
import logging
from abc import ABC, abstractmethod
from os.path import isfile, exists
from typing import Sequence, Union, List, TYPE_CHECKING, Callable

import csep
import numpy
from csep.core.catalogs import CSEPCatalog
from csep.core.forecasts import GriddedForecast, CatalogForecast
from csep.models import EvaluationResult
from csep.utils.time_utils import decimal_year

from floatcsep.utils.readers import ForecastParsers
from floatcsep.infrastructure.registries import ForecastRegistry, ExperimentRegistry
from floatcsep.utils.helpers import str2timewindow, parse_csep_func
from floatcsep.utils.helpers import timewindow2str

log = logging.getLogger("floatLogger")

if TYPE_CHECKING:
    from floatcsep.evaluation import Evaluation
    from floatcsep.model import Model


class ForecastRepository(ABC):

    @abstractmethod
    def __init__(self, registry: ForecastRegistry):
        self.registry = registry
        self.lazy_load = False
        self.forecasts = {}

    @abstractmethod
    def load_forecast(self, tstring: Union[str, Sequence[str]], **kwargs):
        pass

    @abstractmethod
    def _load_single_forecast(self, tstring: str, **kwargs):
        pass

    @abstractmethod
    def remove(self, tstring: Union[str, Sequence[str]]):
        pass

    def __eq__(self, other) -> bool:

        if not isinstance(other, ForecastRepository):
            return False

        if len(self.forecasts) != len(other.forecasts):
            return False

        for key in self.forecasts.keys():
            if key not in other.forecasts.keys():
                return False
            if self.forecasts[key] != other.forecasts[key]:
                return False
        return True

    @classmethod
    def factory(
        cls, registry: ForecastRegistry, model_class: str, forecast_type: str = None, **kwargs
    ) -> "ForecastRepository":
        """Factory method. Instantiate first on explicit option provided in the model
        configuration. Then, defaults to gridded forecast for TimeIndependentModel and catalog
        forecasts for TimeDependentModel
        """

        if forecast_type == "catalog":
            return CatalogForecastRepository(registry, **kwargs)
        elif forecast_type == "gridded":
            return GriddedForecastRepository(registry, **kwargs)

        if model_class == "TimeIndependentModel":
            return GriddedForecastRepository(registry, **kwargs)
        elif model_class == "TimeDependentModel":
            return CatalogForecastRepository(registry, **kwargs)
        else:
            raise ValueError(f"Unknown forecast type: {forecast_type}")


class CatalogForecastRepository(ForecastRepository):
    """
    The class is responsible to access (or store in memory) the catalog-based forecasts of a
    model. The flag `lazy_load` can be set to False so the catalogs are stored in memory and
    reduce the time required to parse files.

    """

    def __init__(self, registry: ForecastRegistry, **kwargs):
        """

        Args:
            registry (ForecastRegistry): The registry containing the keys/path to the forecasts
             given their time-windows.
            **kwargs:
        """
        self.registry = registry
        self.lazy_load = kwargs.get("lazy_load", True)
        self.forecasts = {}

    def load_forecast(
        self, tstring: Union[str, list], region=None
    ) -> Union[CatalogForecast, list[CatalogForecast]]:
        """
        Returns a forecast object or a sequence of them for a set of time window strings.

        Args:
            tstring (str, list): String representing the time-window
            region (optional): A region, in case the forecast requires to be filtered lazily.

        Returns:
            The CSEP CatalogForecast object or a list of them.
        """
        if isinstance(tstring, str):
            return self._load_single_forecast(tstring, region)
        else:
            return [self._load_single_forecast(t, region) for t in tstring]

    def _load_single_forecast(self, t: str, region=None):
        fc_path = self.registry.get_forecast(t)
        return csep.load_catalog_forecast(
            fc_path, region=region, apply_filters=True, filter_spatial=True
        )

    def remove(self, tstring: Union[str, Sequence[str]]):
        pass


class GriddedForecastRepository(ForecastRepository):
    """
    The class is responsible to access (or store in memory) the gridded-based forecasts of a
    model. A keyword `lazy_load` can be set to False so the catalogs are stored in memory and
    avoid parsing files repeatedly (Skip for large files).

    """
    def __init__(self, registry: ForecastRegistry, **kwargs):
        """

        Args:
            registry (ForecastRegistry): The registry containing the keys/path to the forecasts
             given their time-windows.
            **kwargs:
        """
        self.registry = registry
        self.lazy_load = kwargs.get("lazy_load", False)
        self.forecasts = {}

    def load_forecast(
        self, tstring: Union[str, list] = None, name="", region=None, forecast_unit=1
    ) -> Union[GriddedForecast, Sequence[GriddedForecast]]:
        """
        Returns a forecast object or a sequence of them for a set of time window strings.

        Args:
            tstring (str, list): String representing the time-window
            name (str): Forecast name
            region (optional): A region, in case the forecast requires to be filtered lazily.
            forecast_unit (float): The time unit (in decimal years) that the forecast represents

        Returns:
            The CSEP CatalogForecast object or a list of them.
        """
        if isinstance(tstring, str):
            return self._get_or_load_forecast(tstring, name, forecast_unit)
        else:
            return [self._get_or_load_forecast(tw, name, forecast_unit) for tw in tstring]

    def _get_or_load_forecast(
        self, tstring: str, name: str, forecast_unit: float
    ) -> GriddedForecast:
        """Helper method to get or load a single forecast."""
        if tstring in self.forecasts:
            log.debug(f"Using {name} forecast for {tstring} from memory")
            return self.forecasts[tstring]
        else:
            log.debug(f"Loading {name} forecast for {tstring}")
            forecast = self._load_single_forecast(tstring, forecast_unit, name)
            if not self.lazy_load:
                self.forecasts[tstring] = forecast
            return forecast

    def _load_single_forecast(self, tstring: str, fc_unit: float = 1, name_=""):

        start_date, end_date = str2timewindow(tstring)

        time_horizon = decimal_year(end_date) - decimal_year(start_date)
        tstring_ = timewindow2str([start_date, end_date])

        f_path = self.registry.get_forecast(tstring_)
        f_parser = getattr(ForecastParsers, self.registry.fmt)

        rates, region, mags = f_parser(f_path)

        forecast_ = GriddedForecast(
            name=f"{name_}",
            data=rates,
            region=region,
            magnitudes=mags,
            start_time=start_date,
            end_time=end_date,
        )

        scale = time_horizon / fc_unit
        if scale != 1.0:
            forecast_ = forecast_.scale(scale)

        log.debug(
            f"\tForecast expected count: {forecast_.event_count:.2f}"
            f" with scaling parameter: {scale:.1f}"
        )

        return forecast_

    def remove(self, tstring: Union[str, Sequence[str]]):
        pass


class ResultsRepository:
    """
    The class is responsible to access, read and write the results of a given evaluation
    """
    def __init__(self, registry: ExperimentRegistry):
        """

        Args:
            registry (ExperimentRegistry): The registry of an experiment, which keeps track
             of the filepaths of each result.
        """
        self.registry = registry

    def _load_result(
        self,
        test: "Evaluation",
        window: Union[str, Sequence[datetime.datetime]],
        model: "Model",
    ) -> EvaluationResult:

        if not isinstance(window, str):
            wstr_ = timewindow2str(window)
        else:
            wstr_ = window

        eval_path = self.registry.get_result(wstr_, test, model)

        with open(eval_path, "r") as file_:
            model_eval = EvaluationResult.from_dict(json.load(file_))

        return model_eval

    def load_results(
        self,
        test: "Evaluation",
        window: Union[str, Sequence[datetime.datetime]],
        models: Union[list["Model"], "Model"],
    ) -> Union[List, EvaluationResult]:
        """
        Reads an Evaluation result for a given time window and returns a list of the results for
        all tested models.

        Args:
            test (Evaluation): The tests for which the results are to be loaded
            window (str, list): The time-windows for which the results are to be loaded
            models (Model, list): The models for which the results are to be loaded
        """

        if isinstance(models, list):
            test_results = []
            for model in models:
                model_eval = self._load_result(test, window, model)
                test_results.append(model_eval)
            return test_results
        else:
            return self._load_result(test, window, models)

    def write_result(self, result: EvaluationResult, test, model, window) -> None:
        """
        Writes the evaluation results using their method .to_dict() as json file.


        Args:
            result: CSEP evaluation result
            test: Name of the test
            model: Name of the model
            window: Name of the time-window

        """
        path = self.registry.get_result(window, test, model)

        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, numpy.integer):
                    return int(obj)
                if isinstance(obj, numpy.floating):
                    return float(obj)
                if isinstance(obj, numpy.ndarray):
                    return obj.tolist()
                return json.JSONEncoder.default(self, obj)

        with open(path, "w") as _file:
            json.dump(result.to_dict(), _file, indent=4, cls=NumpyEncoder)


class CatalogRepository:
    """
    The class handles the main and sub-catalogs from the experiment. It is responsible of
    accessing, downloading, storing the main catalog, as well as filtering and storing the
    corresponding input-catalogs (e.g., input for a model to be run) and test-catalogs (catalogs
    for the model's forecasts to be evaluated against).
    """

    def __init__(self, registry: ExperimentRegistry):
        """

        Args:
            registry (ExperimentRegistry): The registry of the experiment

        """
        self.cat_path = None
        self._catalog = None
        self.registry = registry
        self.time_config = {}
        self.region_config = {}

    def __dir__(self):
        """Adds time and region configs keys to instance scope."""

        _dir = (
            list(super().__dir__()) + list(self.time_config.keys()) + list(self.region_config)
        )
        return sorted(_dir)

    def __getattr__(self, item: str) -> object:
        """
        Override built-in method to return attributes found within.
        :attr:`region_config` or :attr:`time_config`
        """

        try:
            return self.__dict__[item]
        except KeyError:
            try:
                return self.time_config[item]
            except KeyError:
                try:
                    return self.region_config[item]
                except KeyError:
                    raise AttributeError(
                        f"Experiment '{self.name}'" f" has no attribute '{item}'"
                    ) from None

    def as_dict(self):
        return

    def set_main_catalog(
        self, catalog: Union[str, Callable, CSEPCatalog], time_config: dict, region_config: dict
    ):
        """
        Sets the catalog to be used for the experiment.

        Args:
            catalog: Experiment's main catalog.
            region_config: Experiment instantiation
            time_config:
        """
        self.catalog = catalog
        self.time_config = time_config
        self.region_config = region_config

    @property
    def catalog(self) -> CSEPCatalog:
        """
        Returns a CSEP catalog loaded from the given query function or a stored file if it
        exists.
        """
        cat_path = self.registry.abs(self.cat_path)

        if callable(self._catalog):
            if isfile(self.cat_path):
                return CSEPCatalog.load_json(self.cat_path)
            bounds = {
                "start_time": min([item for sublist in self.timewindows for item in sublist]),
                "end_time": max([item for sublist in self.timewindows for item in sublist]),
                "min_magnitude": self.magnitudes.min(),
                "max_depth": self.depths.max(),
            }
            if self.region:
                bounds.update(
                    {
                        i: j
                        for i, j in zip(
                            ["min_longitude", "max_longitude", "min_latitude", "max_latitude"],
                            self.region.get_bbox(),
                        )
                    }
                )

            catalog = self._catalog(catalog_id="catalog", **bounds)

            if self.region:
                catalog.filter_spatial(region=self.region, in_place=True)
                catalog.region = None
            catalog.write_json(self.cat_path)

            return catalog

        elif isfile(cat_path):
            try:
                return CSEPCatalog.load_json(cat_path)
            except json.JSONDecodeError:
                return csep.load_catalog(cat_path)

    @catalog.setter
    def catalog(self, cat: Union[Callable, CSEPCatalog, str]) -> None:

        if cat is None:
            self._catalog = None
            self.cat_path = None

        elif isfile(self.registry.abs(cat)):
            log.info(f"\tCatalog: '{cat}'")
            self._catalog = self.registry.rel(cat)
            self.cat_path = self.registry.rel(cat)

        else:
            # catalog can be a function
            self._catalog = parse_csep_func(cat)
            self.cat_path = self.registry.abs("catalog.json")
            if isfile(self.cat_path):
                log.info(f"\tCatalog: stored " f"'{self.cat_path}' " f"from '{cat}'")
            else:
                log.info(f"\tCatalog: '{cat}'")

    def get_test_cat(self, tstring: str = None) -> CSEPCatalog:
        """
        Filters the complete experiment catalog to a test sub-catalog bounded by the test
        time-window. Writes it to filepath defined in :attr:`Experiment.registry`

        Args:
            tstring (str): Time window string
        """

        if tstring:
            start, end = str2timewindow(tstring)
        else:
            start = self.start_date
            end = self.end_date

        sub_cat = self.catalog.filter(
            [
                f"origin_time < {end.timestamp() * 1000}",
                f"origin_time >= {start.timestamp() * 1000}",
                f"magnitude >= {self.mag_min}",
                f"magnitude < {self.mag_max}",
            ],
            in_place=False,
        )
        if self.region:
            sub_cat.filter_spatial(region=self.region, in_place=True)

        return sub_cat

    def set_test_cat(self, tstring: str) -> None:
        """
        Filters the complete experiment catalog to a test sub-catalog bounded by the test
        time-window. Writes it to filepath defined in :attr:`Experiment.registry`

        Args:
            tstring (str): Time window string
        """

        testcat_name = self.registry.get_test_catalog(tstring)
        if not exists(testcat_name):
            log.debug(
                f"Filtering testing catalog and saving to {self.registry.rel(testcat_name)}"
            )
            start, end = str2timewindow(tstring)
            sub_cat = self.catalog.filter(
                [
                    f"origin_time < {end.timestamp() * 1000}",
                    f"origin_time >= {start.timestamp() * 1000}",
                    f"magnitude >= {self.mag_min}",
                    f"magnitude < {self.mag_max}",
                ],
                in_place=False,
            )
            if self.region:
                sub_cat.filter_spatial(region=self.region, in_place=True)
            sub_cat.write_json(filename=testcat_name)
        else:
            log.debug(f"Using test catalog from {self.registry.rel(testcat_name)}")

    def set_input_cat(self, tstring: str, model: "Model") -> None:
        """
        Filters the complete experiment catalog to input sub-catalog filtered to the beginning
        of the test time-window.

        Args:
            tstring (str): Time window string
            model (:class:`~floatcsep.model.Model`): Model to give the input
             catalog
        """
        start, end = str2timewindow(tstring)
        sub_cat = self.catalog.filter([f"origin_time < {start.timestamp() * 1000}"])
        sub_cat.write_ascii(filename=model.registry.get("input_cat"))
