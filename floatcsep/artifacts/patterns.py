from typing import List
from csep.core.forecasts import GriddedForecast as Forecast
from csep.core.catalogs import CSEPCatalog as Catalog


def consistency_test(
        gridded_forecast: Forecast,
        observed_catalog: Catalog):
    pass


def comparative_test(
        forecast: Forecast,
        benchmark_forecast: Forecast,
        observed_catalog: Catalog):
    pass


def sequential_score(
        gridded_forecasts: List[Forecast],
        observed_catalogs: List[Catalog]):
    pass


def sequential_relative_score(
        gridded_forecasts: List[Forecast],
        reference_forecasts: List[Forecast],
        observed_catalogs: List[Catalog]):
    pass


def vector_poisson_t_w_test(
        forecasts: Forecast,
        benchmark_forecast: List[Forecast],
        catalog: Catalog):
    pass
