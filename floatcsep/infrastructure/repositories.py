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

from floatcsep.infrastructure.registries import (
    ExperimentRegistry,
    ModelRegistry,
    ExperimentFileRegistry,
    ModelFileRegistry,
)
from floatcsep.utils.file_io import (
    GriddedForecastParsers,
    CatalogForecastParsers,
    CatalogSerializer,
    CatalogParser,
)
from floatcsep.utils.helpers import str2timewindow, parse_csep_func
from floatcsep.utils.helpers import timewindow2str

log = logging.getLogger("floatLogger")

if TYPE_CHECKING:
    from floatcsep.evaluation import Evaluation
    from floatcsep.model import Model


class CatalogRepository:
    """
    The class handles the main and sub-catalogs from the experiment. It is responsible for
    accessing, downloading, storing the main catalog, as well as filtering and storing the
    corresponding input-catalogs (e.g., input for a model to be run) and test-catalogs (catalogs
    for the model's forecasts to be evaluated against).
    """

    def __init__(self, registry: ExperimentFileRegistry):
        """

        Args:
            registry (ExperimentRegistry): The registry of the experiment

        """
        self.name = "CatalogRepository"
        self.cat_path = None
        self._catalog = None
        self.registry = registry
        self.time_config = {}
        self.region_config = {}
        self.input_config = {}

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
        self,
        catalog: Union[str, Callable, CSEPCatalog],
        time_config: dict,
        region_config: dict,
        input_config: dict = None,
    ):
        """
        Sets the catalog to be used for the experiment.

        Args:
            catalog: Experiment's main catalog.
            time_config: Experiment temporal configuration
            region_config: Experiment region configuration (testing domain)
            input_config: Input data configuration (see
                :func:`~floatcsep.utils.helpers.read_input_cfg`). Its ``catalog`` block
                sets how the models' input catalogs are filtered.
        """
        self.time_config = time_config
        self.region_config = region_config
        self.input_config = input_config or {}
        self.catalog = catalog

    @property
    def input_cat_config(self) -> dict:
        """
        Returns:
            The settings used to filter the models' input catalogs. Defaults to the testing
            magnitude range, without spatial, depth or lower time bounds.
        """
        cfg = self.input_config.get("catalog") or {}
        return {
            "region": cfg.get("region", None),
            "mag_min": cfg.get("mag_min", self.region_config.get("mag_min")),
            "mag_max": cfg.get("mag_max", self.region_config.get("mag_max")),
            "depth_min": cfg.get("depth_min", None),
            "depth_max": cfg.get("depth_max", None),
            "start_date": cfg.get("start_date", None),
        }

    @property
    def catalog(self) -> CSEPCatalog:
        """
        Returns a CSEP catalog loaded from the given query function or a stored file if it
        exists.
        """
        return self._catalog

    @catalog.setter
    def catalog(self, cat: Union[Callable, CSEPCatalog, str]) -> None:
        if cat is None:
            self._catalog = None
            self.cat_path = None

        elif isfile(self.registry.abs(cat)):
            log.info(f"\tCatalog: {cat}")
            try:
                reader = getattr(CatalogParser, "json")
                self._catalog = reader(self.registry.abs(cat))
            except json.JSONDecodeError:
                self._catalog = csep.load_catalog(self.registry.abs(cat))
            self.cat_path = self.registry.rel(cat)
            self.name = cat
        else:
            query_function = parse_csep_func(cat)
            bounds = self.query_bounds()

            self._catalog = query_function(catalog_id="catalog", **bounds)
            self.cat_path = self.registry.rel("catalog.json")
            self.name = cat
            writer = getattr(CatalogSerializer, "json")
            writer(catalog=self._catalog, filename=self.registry.abs(self.cat_path))

            if isfile(self.registry.abs(self.cat_path)):
                log.info(f"\tCatalog: stored " f"'{self.cat_path}' " f"from '{cat}'")
            else:
                log.info(f"\tCatalog: '{cat}'")

    def query_bounds(self) -> dict:
        """
        Bounds to query the main catalog from a network API, wide enough to cover both the
        testing settings (``region_config``, ``time_config``) and the input catalog settings
        (``input_config``).

        Returns:
            Keyword arguments for the catalog query function.
        """
        inp = self.input_cat_config
        start = min([item for sublist in self.time_windows for item in sublist])
        if inp["start_date"]:
            start = min(start, inp["start_date"])
        mag_min = self.magnitudes.min()
        if inp["mag_min"] is not None:
            mag_min = min(mag_min, inp["mag_min"])
        depth_max = self.depths.max()
        if inp["depth_max"] is not None:
            depth_max = max(depth_max, inp["depth_max"])
        bounds = {
            "start_time": start,
            "end_time": max([item for sublist in self.time_windows for item in sublist]),
            "min_magnitude": mag_min,
            "max_depth": depth_max,
        }
        regions = [r for r in (self.region, inp["region"]) if r is not None]
        if regions:
            bboxes = numpy.array([r.get_bbox() for r in regions])
            bounds.update(
                {
                    "min_longitude": bboxes[:, 0].min(),
                    "max_longitude": bboxes[:, 1].max(),
                    "min_latitude": bboxes[:, 2].min(),
                    "max_latitude": bboxes[:, 3].max(),
                }
            )
        return bounds

    def get_test_cat(self, tstring: str = None, fmt: str = "json") -> CSEPCatalog:
        """
        Filters the complete experiment catalog to a test sub-catalog bounded by the test
        time-window. Writes it to filepath defined in :attr:`Experiment.registry`

        Args:
            tstring (str): Time window string
            fmt (str): Format of the catalog to be used
        """

        test_cat_name = self.registry.get_test_catalog_key(tstring)
        reader = getattr(CatalogParser, fmt)
        test_catalog = reader(filename=test_cat_name)

        return test_catalog

    def get_input_cat(self, tstring: str) -> CSEPCatalog:
        """
        Filters the main catalog to the input catalog of a time window: all the events before
        the window start, within the settings of ``input_config.catalog`` (region, magnitude,
        depth and lower time bound). See :func:`~floatcsep.utils.helpers.read_input_cfg`.

        Args:
            tstring (str): Time window string

        Returns:
            The input catalog as a :class:`csep.core.catalogs.CSEPCatalog`
        """
        start, end = str2timewindow(tstring)
        cfg = self.input_cat_config
        filters = [f"origin_time < {start.timestamp() * 1000}"]
        if cfg["start_date"]:
            filters.append(f"origin_time >= {cfg['start_date'].timestamp() * 1000}")
        if cfg["mag_min"] is not None:
            filters.append(f"magnitude >= {cfg['mag_min']}")
        if cfg["mag_max"] is not None:
            filters.append(f"magnitude < {cfg['mag_max']}")
        if cfg["depth_min"] is not None:
            filters.append(f"depth >= {cfg['depth_min']}")
        if cfg["depth_max"] is not None:
            filters.append(f"depth < {cfg['depth_max']}")
        sub_cat = self.catalog.filter(filters, in_place=False)
        if cfg["region"] is not None and sub_cat.get_number_of_events() > 0:
            sub_cat.filter_spatial(region=cfg["region"], in_place=True)
        return sub_cat

    def set_input_cats(self, tstring: str, models: List["Model"], fmt: str = "ascii") -> None:
        """
        Filters the complete experiment catalog to the input sub-catalog of a time window (see
        :meth:`get_input_cat`) and writes it to the input directory of each model.

        Args:
            tstring (str): Time window string
            models (list of :class:`~floatcsep.model.Model`): Models to give the input catalog
            fmt (str): Output catalog format
        """
        log.debug("[Catalogs] Filtering input catalog and saving to models' input directory")
        sub_cat = self.get_input_cat(tstring)
        writer = getattr(CatalogSerializer, fmt)
        for model in models:
            writer(catalog=sub_cat, filename=model.registry.get_input_catalog_key(tstring))

    def set_test_cats(self, tstring: str, fmt: str = "json") -> None:
        """
        Filters the complete experiment catalog to a test sub-catalog bounded by the test
        time-window. Writes it to filepath defined in :attr:`Experiment.registry`

        Args:
            tstring (str): Time window string
            fmt (str): Output catalog format
        """

        test_cat_name = self.registry.get_test_catalog_key(tstring)

        log.debug(
            f"[Catalogs] Filtering testing catalog and saving to "
            f"{self.registry.rel(test_cat_name)}"
        )
        start, end = str2timewindow(tstring)
        filters = [
            f"origin_time < {end.timestamp() * 1000}",
            f"origin_time >= {start.timestamp() * 1000}",
            f"magnitude >= {self.mag_min}",
            f"magnitude < {self.mag_max}",
        ]
        depth_min = self.region_config.get("depth_min", None)
        depth_max = self.region_config.get("depth_max", None)
        if depth_min is not None:
            filters.append(f"depth >= {depth_min}")
        if depth_max is not None:
            filters.append(f"depth < {depth_max}")
        sub_cat = self.catalog.filter(filters, in_place=False)
        if self.region and sub_cat.get_number_of_events() > 0:
            sub_cat.filter_spatial(region=self.region, in_place=True)

        writer = getattr(CatalogSerializer, fmt)
        writer(catalog=sub_cat, filename=test_cat_name)

    def filter_catalog(
        self,
        start_date=None,
        end_date=None,
        min_mag=None,
        max_mag=None,
        min_depth=None,
        max_depth=None,
        region=None,
    ) -> CSEPCatalog:
        """
        Wrapper for pyCSEP catalog filters, to constrain a catalog to a given time and magnitude
        range, as well to a spatial region.

        Args:
            start_date (datetime.datetime): Initial datetime
            end_date (datetime.datetime): Final datetime
            min_mag (float): Minimum magnitude to filter
            max_mag (float): Maximum magnitude to filter
            min_depth (float): Minimum depth to filter (positive downwards from surface)
            max_depth (float): Maximum depth to filter (positive downwards from surface)
            region (csep.core.regions.CartesianGrid2D): Spatial domain to filter

        Returns:
            Filtered catalog
        """
        filters = []
        if start_date:
            filters.append(
                f"origin_time >= {csep.utils.time_utils.datetime_to_utc_epoch(start_date)}"
            )
        if end_date:
            filters.append(
                f"origin_time <= {csep.utils.time_utils.datetime_to_utc_epoch(end_date)}"
            )
        if min_mag:
            filters.append(f"magnitude >= {min_mag}")
        if max_mag:
            filters.append(f"magnitude <= {max_mag}")
        if min_depth:
            filters.append(f"depth >= {min_depth}")
        if max_depth:
            filters.append(f"depth <= {max_depth}")
        filtered_catalog = self.catalog.filter(filters, in_place=False)
        if region:
            filtered_catalog.filter_spatial(region=region, in_place=True)

        return filtered_catalog


class ForecastRepository(ABC):

    @abstractmethod
    def __init__(self, registry: ModelRegistry):
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
        cls, registry: ModelRegistry, model_class: str, forecast_type: str = None, **kwargs
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

    def __init__(self, registry: ModelFileRegistry, **kwargs):
        """

        Args:
            registry (ModelRegistry): The registry containing the keys/path to the forecasts
             given their time-windows.
            **kwargs:
        """
        self.registry = registry
        self.lazy_load = kwargs.get("lazy_load", True)
        self.forecasts = {}

    def load_forecast(
        self,
        tstring: Union[str, list],
        name=None,
        region=None,
        n_sims=None,
    ) -> Union[CatalogForecast, list[CatalogForecast]]:
        """
        Returns a forecast object or a sequence of them for a set of time window strings.

        Args:
            tstring (str, list): String representing the time-window.
            name (str): Name of the forecast model.
            region (optional): A region, in case the forecast requires to be filtered lazily.
            n_sims (optional: The number of simulations/synthetic catalogs of the forecast.

        Returns:
            The CSEP CatalogForecast object or a list of them.
        """
        if isinstance(tstring, str):
            return self._load_single_forecast(tstring, name=name, region=region, n_sims=n_sims)
        else:
            return [self._load_single_forecast(t, region) for t in tstring]

    def _load_single_forecast(self, tstring: str, name=None, region=None, n_sims=None):
        start_date, end_date = str2timewindow(tstring)

        fc_path = self.registry.get_forecast_key(tstring)
        fmt = self.registry.fmt
        f_parser = getattr(CatalogForecastParsers, fmt[1:] if fmt.endswith(".") else fmt)

        forecast_ = f_parser(
            fc_path,
            name=name,
            start_time=start_date,
            end_time=end_date,
            n_cat=n_sims,
            region=region,
            apply_filters=True,
            filter_spatial=True,
        )
        return forecast_

    def remove(self, tstring: Union[str, Sequence[str]]):
        pass


class GriddedForecastRepository(ForecastRepository):
    """
    The class is responsible to access (or store in memory) the gridded-based forecasts of a
    model. A keyword `lazy_load` can be set to False so the catalogs are stored in memory and
    avoid parsing files repeatedly (Skip for large files).

    """

    def __init__(self, registry: ModelFileRegistry, **kwargs):
        """

        Args:
            registry (ModelRegistry): The registry containing the keys/path to the forecasts
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

        f_path = self.registry.get_forecast_key(tstring_)
        fmt = self.registry.fmt
        f_parser = getattr(GriddedForecastParsers, fmt[1:] if fmt.startswith(".") else fmt)

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

        eval_path = self.registry.get_result_key(wstr_, test, model)

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
        path = self.registry.get_result_key(window, test, model)

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
