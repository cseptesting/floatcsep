import logging
from abc import ABC, abstractmethod
from typing import Sequence, Union

import csep
from csep.core.forecasts import GriddedForecast
from csep.utils.time_utils import decimal_year

from floatcsep.readers import ForecastParsers
from floatcsep.registry import ForecastRegistry
from floatcsep.utils import str2timewindow
from floatcsep.utils import timewindow2str

log = logging.getLogger("floatLogger")


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

    def __init__(self, registry: ForecastRegistry, **kwargs):
        self.registry = registry
        self.lazy_load = kwargs.get("lazy_load", True)
        self.forecasts = {}

    def load_forecast(self, tstring: Union[str, list], region=None):

        if isinstance(tstring, str):
            return self._load_single_forecast(tstring, region)
        else:
            return [self._load_single_forecast(t, region) for t in tstring]

    def _load_single_forecast(self, t: str, region=None):
        fc_path = self.registry.get_path("forecasts", t)
        return csep.load_catalog_forecast(
            fc_path, region=region, apply_filters=True, filter_spatial=True
        )

    def remove(self, tstring: Union[str, Sequence[str]]):
        pass


class GriddedForecastRepository(ForecastRepository):

    def __init__(self, registry: ForecastRegistry, **kwargs):
        self.registry = registry
        self.lazy_load = kwargs.get("lazy_load", False)
        self.forecasts = {}

    def load_forecast(
        self, tstring: Union[str, list] = None, name="", region=None, forecast_unit=1
    ) -> Union[GriddedForecast, Sequence[GriddedForecast]]:
        """Returns a forecast when requested."""
        if isinstance(tstring, str):
            return self._get_or_load_forecast(tstring, name, forecast_unit)
        else:
            return [self._get_or_load_forecast(tw, name, forecast_unit) for tw in tstring]

    def _get_or_load_forecast(
        self, tstring: str, name: str, forecast_unit: int
    ) -> GriddedForecast:
        """Helper method to get or load a single forecast."""
        if tstring in self.forecasts:
            log.debug(f"Loading {name} forecast for {tstring} from memory")
            return self.forecasts[tstring]
        else:
            log.debug(f"Loading {name} forecast for {tstring} on the fly")
            forecast = self._load_single_forecast(tstring, forecast_unit, name)
            if not self.lazy_load:
                self.forecasts[tstring] = forecast
            return forecast

    def _load_single_forecast(self, tstring: str, fc_unit=1, name_=""):

        start_date, end_date = str2timewindow(tstring)

        time_horizon = decimal_year(end_date) - decimal_year(start_date)
        tstring_ = timewindow2str([start_date, end_date])

        f_path = self.registry.get_path("forecasts", tstring_)
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
            f"Model {name_}:\n"
            f"\tForecast expected count: {forecast_.event_count:.2f}"
            f" with scaling parameter: {time_horizon:.1f}"
        )

        return forecast_

    def remove(self, tstring: Union[str, Sequence[str]]):
        pass
