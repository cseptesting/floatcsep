# python libraries
import copy
import functools
import logging
import os
import re
from datetime import datetime, date
from typing import Union, Mapping, Sequence, Any, Optional

import matplotlib

# third-party libraries
import numpy
import pandas
import scipy.stats
import seaborn
import yaml
from matplotlib import pyplot
from matplotlib.lines import Line2D

# pyCSEP libraries
import csep.core
import csep.utils
from csep.core.catalogs import CSEPCatalog
from csep.core.exceptions import CSEPCatalogException
from csep.core.forecasts import GriddedForecast
from csep.core.poisson_evaluations import (
    paired_t_test,
    w_test,
    _poisson_likelihood_test,
)
from csep.core.regions import CartesianGrid2D
from csep.models import EvaluationResult
from csep.utils.calc import cleaner_range
from matplotlib.dates import AutoDateLocator, DateFormatter
from csep.utils.constants import SECONDS_PER_DAY, SECONDS_PER_HOUR

# floatCSEP libraries
import floatcsep.utils.accessors
import floatcsep.utils.file_io

_UNITS = ["years", "months", "weeks", "days"]
_PD_FORMAT = ["YS", "MS", "W", "D"]


log = logging.getLogger("floatLogger")


def parse_csep_func(func):
    """
    Search in pyCSEP and floatCSEP a function or method whose name matches the provided string.

    Args:
        func (str, obj) : representing the name of the pycsep/floatcsep function
            or method

    Returns:

        The callable function/method object. If it was already callable,
        returns the same input
    """

    def recgetattr(obj, attr, *args):
        def _getattr(obj_, attr_):
            return getattr(obj_, attr_, *args)

        return functools.reduce(_getattr, [obj] + attr.split("."))

    if callable(func):
        return func
    elif func is None:
        return func
    else:
        _target_modules = [
            csep,
            csep.utils,
            csep.plots,
            csep.core.regions,
            floatcsep.utils.helpers,
            floatcsep.utils.accessors,
            floatcsep.utils.file_io.HDF5Serializer,
            floatcsep.utils.file_io.GriddedForecastParsers,
            floatcsep.utils.file_io.CatalogForecastParsers,
        ]
        for module in _target_modules:
            try:
                return recgetattr(module, func)
            except AttributeError:
                pass
        raise AttributeError(
            f"Evaluation/Plot/Region function {func} has not yet been"
            f" implemented in floatcsep or pycsep"
        )


def parse_timedelta_string(window, exp_class="ti"):
    """
    Parses a float or string representing the testing time window length.

    Note:

        Time-independent experiments defaults to `year` as time unit whereas
        time-dependent to `days`

    Args:

        window (str, int): length of the time window
        exp_class (str): experiment class

    Returns:

        Formatted :py:class:`str` representing the length and
        unit (year, month, week, day) of the time window
    """

    if isinstance(window, str):
        try:
            n, unit_ = [i for i in re.split(r"(\d+)", window) if i]
            unit = [i for i in [j[:-1] for j in _UNITS] if i in unit_.lower()][0]
            return f"{n}-{unit}s"

        except (ValueError, IndexError):
            raise ValueError(
                "Time window is misspecified. "
                "Try the amount followed by the time unit "
                "(e.g. 1 day, 1 months, 3 years)"
            )
    elif isinstance(window, float):
        n = window
        unit = "year" if exp_class == "ti" else "day"
        return f"{n}-{unit}s"


def read_time_cfg(time_config, **kwargs):
    """
    Builds the temporal configuration of an experiment.

    Args:
        time_config (dict): Dictionary containing the explicit temporal
            attributes of the experiment (see `_attrs` local variable)
        **kwargs: Only the keywords contained in the local variable `_attrs`
            are captured. This ensures to capture the keywords passed
            to an :class:`~floatcsep.core.Experiment` object

    Returns:
        A dictionary containing the experiment time attributes and the time
        windows to be evaluated
    """
    _attrs = ["start_date", "end_date", "intervals", "horizon", "offset", "growth", "exp_class"]
    time_config = copy.deepcopy(time_config)
    if time_config is None:
        time_config = {}

    experiment_class = time_config.get("exp_class", kwargs.get("exp_class", "ti"))
    time_config["exp_class"] = experiment_class

    time_config.update({i: j for i, j in kwargs.items() if i in _attrs})
    if "horizon" in time_config.keys():
        time_config["horizon"] = parse_timedelta_string(time_config["horizon"])
    if "offset" in time_config.keys():
        time_config["offset"] = parse_timedelta_string(time_config["offset"])

    if not time_config.get("time_windows"):
        if experiment_class == "ti":
            time_config["time_windows"] = time_windows_ti(**time_config)
        elif experiment_class == "td":
            time_config["time_windows"] = time_windows_td(**time_config)
    else:
        time_config["start_date"] = time_config["time_windows"][0][0]
        time_config["end_date"] = time_config["time_windows"][-1][-1]

    return time_config


def read_region_cfg(region_config, **kwargs):
    """
    Builds the region configuration of an experiment.

    Args:
        region_config (dict): Dictionary containing the explicit region
            attributes of the experiment (see `_attrs` local variable)
        **kwargs: Only the keywords contained in the local variable `_attrs`
            are captured. This ensures to capture the keywords passed
            to an :class:`~floatcsep.core.Experiment` object

    Returns:
        A dictionary containing the region attributes of the experiment
    """
    region_config = copy.deepcopy(region_config)
    _attrs = ["region", "mag_min", "mag_max", "mag_bin", "magnitudes", "depth_min", "depth_max"]
    if region_config is None:
        region_config = {}
    region_config.update({i: j for i, j in kwargs.items() if i in _attrs})

    dmin = region_config.get("depth_min", -2)
    dmax = region_config.get("depth_max", 6000)
    depths = cleaner_range(dmin, dmax, dmax - dmin)
    magnitudes = region_config.get("magnitudes", None)
    if magnitudes is None:
        magmin = region_config["mag_min"]
        magmax = region_config["mag_max"]
        magbin = region_config["mag_bin"]
        magnitudes = cleaner_range(magmin, magmax, magbin)

    region_data = region_config.get("region", None)
    try:
        region = (
            parse_csep_func(region_data)(name=region_data, magnitudes=magnitudes)
            if region_data
            else None
        )
    except AttributeError:
        if isinstance(region_data, str):
            filename = os.path.join(kwargs.get("path", ""), region_data)
            with open(filename, "r") as file_:
                parsed_region = file_.readlines()
                try:
                    data = numpy.array(
                        [re.split(r"\s+|,", i.strip()) for i in parsed_region], dtype=float
                    )
                except ValueError:
                    data = numpy.array(
                        [re.split(r"\s+|,", i.strip()) for i in parsed_region[1:]], dtype=float
                    )
                dh1 = scipy.stats.mode(numpy.diff(numpy.unique(data[:, 0]))).mode
                dh2 = scipy.stats.mode(numpy.diff(numpy.unique(data[:, 1]))).mode
                dh = numpy.nanmin([dh1, dh2])
                region = CartesianGrid2D.from_origins(
                    data, name=region_data, magnitudes=magnitudes, dh=dh
                )
                region_config.update({"path": region_data})
        else:
            region_data["magnitudes"] = magnitudes
            region = CartesianGrid2D.from_dict(region_data)

    region_config.update({"depths": depths, "magnitudes": magnitudes, "region": region})

    return region_config


def timewindow2str(datetimes: Sequence) -> Union[str, list[str]]:
    """
    Converts a time window (list/tuple of datetimes) to a string that represents it.  Can be a
    single timewindow or a list of time windows.

    Args:
        datetimes: A sequence (of sequences) of datetimes, representing a list of time_windows

    Returns:
        A sequence of strings for each time window
    """
    if all(isinstance(i, datetime) for i in datetimes):
        return "_".join([j.date().isoformat() for j in datetimes])

    elif all(isinstance(i, (list, tuple)) for i in datetimes):
        return ["_".join([j.date().isoformat() for j in i]) for i in datetimes]


def str2timewindow(
    tw_string: Union[str, Sequence[str]],
) -> Union[Sequence[datetime], Sequence[Sequence[datetime]]]:
    """
    Converts a string representation of a time window into a list of datetimes representing the
    time window edges.

    Args:
        tw_string: A string representing the time window ('{datetime}_{datetime}')

    Returns:
        A list (of list) containing the pair of datetimes objects
    """
    if isinstance(tw_string, str):
        start_date, end_date = [datetime.fromisoformat(i) for i in tw_string.split("_")]
        return start_date, end_date

    elif isinstance(tw_string, (list, tuple)):
        datetimes = []
        for twstr in tw_string:
            start_date, end_date = [datetime.fromisoformat(i) for i in twstr.split("_")]
            datetimes.append([start_date, end_date])
        return datetimes


def time_windows_ti(
    start_date=None, end_date=None, intervals=None, horizon=None, growth="incremental", **_
):
    """
    Creates the testing intervals for a time-independent experiment.

    Note:

        The following argument combinations are possible:
            - (start_date, end_date)
            - (start_date, end_date, timeintervals)
            - (start_date, end_date, horizon)
            - (start_date, timeintervals, horizon)

    Args:
        start_date (datetime.datetime): Start of the experiment
        end_date  (datetime.datetime): End of the experiment
        intervals (int): number of intervals to discretize the time span
        horizon (str): time length of each interval
        growth (str): incremental or cumulative time windows

    Returns:

        List of tuples containing the lower and upper boundaries of each
        testing window, as :py:class:`datetime.datetime`.
    """
    frequency = None

    if (intervals is None) and (horizon is None):
        intervals = 1
    elif horizon:
        n, unit = horizon.split("-")
        frequency = f"{n}{_PD_FORMAT[_UNITS.index(unit)]}"

    periods = intervals + 1 if intervals else intervals
    try:
        timelimits = pandas.date_range(
            start=start_date, end=end_date, periods=periods, freq=frequency
        )
        timelimits = timelimits.to_pydatetime()
    except ValueError as e_:
        raise ValueError(
            "The following experiment parameters combinations are possible:\n"
            "   (start_date, end_date)\n"
            "   (start_date, end_date, intervals)\n"
            "   (start_date, end_date, timewindow)\n"
            "   (start_date, intervals, timewindow)\n",
            e_,
        )

    if growth == "incremental":
        return [(i, j) for i, j in zip(timelimits[:-1], timelimits[1:])]

    elif growth == "cumulative":
        return [(timelimits[0], i) for i in timelimits[1:]]


def time_windows_td(
    start_date=None, end_date=None, timeintervals=None, horizon=None, timeoffset=None, **_
):
    """
    Creates the testing intervals for a time-dependent experiment.

    Note:
        The following are combinations are possible:
            - (start_date, end_date, timeintervals)
            - (start_date, end_date, horizon)
            - (start_date, timeintervals, horizon)
            - (start_date,  end_date, horizon, timeoffset)
            - (start_date,  timeinvervals, horizon, timeoffset)

    Args:
        start_date (datetime.datetime): Start of the experiment
        end_date  (datetime.datetime): End of the experiment
        timeintervals (int): number of intervals to discretize the time span
        horizon (str): time length of each time window
        timeoffset (str): Offset between consecutive forecast.
                          if None or timeoffset=horizon, windows are
                          non-overlapping

    Returns:
        List of tuples containing the lower and upper boundaries of each
        testing window, as :py:class:`datetime.datetime`.
    """

    frequency = parse_timedelta_string(horizon)
    offset = parse_timedelta_string(timeoffset)

    if frequency:
        n, unit = frequency.split("-")
        frequency = f"{n}{_PD_FORMAT[_UNITS.index(unit)]}"
    if offset:
        n, unit = offset.split("-")
        offset = f"{n}{_PD_FORMAT[_UNITS.index(unit)]}"

    periods = timeintervals + 1 if timeintervals else timeintervals

    windows = []

    if start_date and end_date and horizon and timeoffset:

        current_start = start_date
        current_end = start_date
        while current_end < end_date:
            next_window = pandas.date_range(
                start=current_start, periods=2, freq=frequency
            ).tolist()

            current_end = next_window[1]

            windows.append((current_start, current_end))

            current_start = pandas.date_range(start=current_start, periods=2, freq=offset)[
                1
            ].to_pydatetime()

    elif start_date and timeintervals and horizon and timeoffset:

        for _ in range(timeintervals):
            next_window = pandas.date_range(
                start=start_date, periods=2, freq=frequency
            ).tolist()
            lower_bound = start_date
            upper_bound = next_window[1]
            windows.append((lower_bound, upper_bound))
            start_date = pandas.date_range(start=start_date, periods=2, freq=offset)[
                1
            ].to_pydatetime()

    elif start_date and end_date and timeintervals:
        if timeintervals == 1:
            log.warning("Only 1 time window is presentL. Consider using exp_class: ti")

        timelimits = pandas.date_range(
            start=start_date, end=end_date, periods=periods, freq=frequency
        ).tolist()
        windows = [(i, j) for i, j in zip(timelimits[:-1], timelimits[1:])]

    # Case 2: (start_date, end_date, horizon)
    elif start_date and end_date and horizon:
        timelimits = pandas.date_range(
            start=start_date, end=end_date, periods=periods, freq=frequency
        ).tolist()
        windows = [(i, j) for i, j in zip(timelimits[:-1], timelimits[1:])]

    # Case 3: (start_date, timeintervals, horizon)
    elif start_date and timeintervals and horizon:
        timelimits = pandas.date_range(
            start=start_date, end=end_date, periods=periods, freq=frequency
        ).tolist()
        windows = [(i, j) for i, j in zip(timelimits[:-1], timelimits[1:])]

    else:
        raise ValueError(
            "The following experiment parameters combinations are possible:\n"
            "   (start_date, end_date, timeintervals)\n"
            "   (start_date, end_date, horizon)\n"
            "   (start_date, timeintervals, horizon)\n"
            "   (start_date, end_date, horizon, timeoffset)\n"
            "   (start_date, timeinvervals, horizon, timeoffset)\n"
        )
    return windows


def parse_nested_dicts(nested_dict: dict) -> dict:
    """
    Parses nested dictionaries to return appropriate parsing on each element
    """

    def _get_value(x):
        # For each element type, transforms to desired string/output
        if hasattr(x, "as_dict"):
            # e.g. model, test, etc.
            o = x.as_dict()
        else:
            try:
                try:
                    o = getattr(x, "__name__")
                except AttributeError:
                    o = getattr(x, "name")
            except AttributeError:
                if isinstance(x, numpy.ndarray):
                    o = x.tolist()
                else:
                    o = x
        return o

    def iter_attr(val):
        # recursive iter through nested dicts/lists
        if isinstance(val, Mapping):
            return {item: iter_attr(val_) for item, val_ in val.items()}
        elif isinstance(val, Sequence) and not isinstance(val, str):
            return [iter_attr(i) for i in val]
        else:
            return _get_value(val)

    return iter_attr(nested_dict)


class NoAliasLoader(yaml.Loader):
    @staticmethod
    def ignore_aliases(self):
        return True


#######################
# Perhaps add to pycsep
#######################


def sequential_likelihood(
    gridded_forecasts: Sequence[GriddedForecast],
    observed_catalogs: Sequence[CSEPCatalog],
    seed: int = None,
    random_numbers=None,
):
    """
    Performs the likelihood test on Gridded Forecast using an Observed Catalog.

    Note: The forecast and the observations should be scaled to the same time period before
    calling this function. This increases transparency as no assumptions are being made about
    the length of the forecasts. This is particularly important for gridded forecasts that
    supply their forecasts as rates.

    Args:
        gridded_forecasts: list csep.core.forecasts.GriddedForecast
        observed_catalogs: list csep.core.catalogs.Catalog
        seed (int): used fore reproducibility, and testing
        random_numbers (numpy.ndarray): random numbers used to override the random number
         generation injection point for testing.

    Returns:
        evaluation_result: csep.core.evaluations.EvaluationResult
    """

    # grid catalog onto spatial grid
    # grid catalog onto spatial grid

    likelihoods = []

    for gridded_forecast, observed_catalog in zip(gridded_forecasts, observed_catalogs):
        try:
            _ = observed_catalog.region.magnitudes
        except CSEPCatalogException:
            observed_catalog.region = gridded_forecast.region

        gridded_catalog_data = observed_catalog.spatial_magnitude_counts()

        # simply call likelihood test on catalog and forecast
        qs, obs_ll, simulated_ll = _poisson_likelihood_test(
            gridded_forecast.data,
            gridded_catalog_data,
            num_simulations=1,
            seed=seed,
            random_numbers=random_numbers,
            use_observed_counts=False,
            normalize_likelihood=False,
        )
        likelihoods.append(obs_ll)
        # populate result data structure

    result = EvaluationResult()
    result.test_distribution = numpy.arange(len(gridded_forecasts))
    result.name = "Sequential Likelihood"
    result.observed_statistic = likelihoods
    result.quantile = 1
    result.sim_name = gridded_forecast.name
    result.obs_name = observed_catalog.name
    result.status = "normal"
    result.min_mw = numpy.min(gridded_forecast.magnitudes)

    return result


def sequential_information_gain(
    gridded_forecasts: Sequence[GriddedForecast],
    benchmark_forecasts: Sequence[GriddedForecast],
    observed_catalogs: Sequence[CSEPCatalog],
    seed: int = None,
    random_numbers: numpy.ndarray = None,
):
    """
    Evaluates the Information Gain for multiple time-windows.

    Args:

        gridded_forecasts: list csep.core.forecasts.GriddedForecast
        benchmark_forecasts: list csep.core.forecasts.GriddedForecast
        observed_catalogs: list csep.core.catalogs.Catalog
        seed (int): used fore reproducibility, and testing
        random_numbers (numpy.ndarray): random numbers used to override the random number
            generation injection point for testing.

    Returns:
        evaluation_result: csep.core.evaluations.EvaluationResult
    """

    information_gains = []

    gridded_forecast = None
    observed_catalog = None

    for gridded_forecast, reference_forecast, observed_catalog in zip(
        gridded_forecasts, benchmark_forecasts, observed_catalogs
    ):
        try:
            _ = observed_catalog.region.magnitudes
        except CSEPCatalogException:
            observed_catalog.region = gridded_forecast.region

        gridded_catalog_data = observed_catalog.spatial_magnitude_counts()

        # simply call likelihood test on catalog and forecast
        qs, obs_ll, simulated_ll = _poisson_likelihood_test(
            gridded_forecast.data,
            gridded_catalog_data,
            num_simulations=1,
            seed=seed,
            random_numbers=random_numbers,
            use_observed_counts=False,
            normalize_likelihood=False,
        )
        qs, ref_ll, simulated_ll = _poisson_likelihood_test(
            reference_forecast.data,
            gridded_catalog_data,
            num_simulations=1,
            seed=seed,
            random_numbers=random_numbers,
            use_observed_counts=False,
            normalize_likelihood=False,
        )

        information_gains.append(obs_ll - ref_ll)

    result = EvaluationResult()
    result.test_distribution = numpy.arange(len(gridded_forecasts))
    result.name = "Sequential Likelihood"
    result.observed_statistic = information_gains
    result.quantile = 1
    result.sim_name = gridded_forecast.name
    result.obs_name = observed_catalog.name
    result.status = "normal"
    result.min_mw = numpy.min(gridded_forecast.magnitudes)

    return result


def vector_poisson_t_w_test(
    forecast: GriddedForecast,
    benchmark_forecasts: Sequence[GriddedForecast],
    catalog: CSEPCatalog,
):
    """
    Computes Student's t-test for the information gain per earthquake over.

    a list of forecasts and w-test for normality

    Uses all ref_forecasts to perform pair-wise t-tests against the
    forecast provided to the function.

    Args:
        forecast (csep.GriddedForecast): forecast to evaluate
        benchmark_forecasts (list of csep.GriddedForecast): list of forecasts to evaluate
        catalog (csep.AbstractBaseCatalog): evaluation catalog filtered consistent with forecast
        **kwargs: additional default arguments

    Returns:
        results (list of csep.EvaluationResult): iterable of evaluation results
    """
    results_t = []
    results_w = []

    for bmf_j in benchmark_forecasts:
        results_t.append(paired_t_test(forecast, bmf_j, catalog))
        results_w.append(w_test(forecast, bmf_j, catalog))
    result = EvaluationResult()
    result.name = "Paired T-Test"
    result.test_distribution = "normal"
    result.observed_statistic = [t.observed_statistic for t in results_t]
    result.quantile = (
        [numpy.abs(t.quantile[0]) - t.quantile[1] for t in results_t],
        [w.quantile for w in results_w],
    )
    result.sim_name = forecast.name
    result.obs_name = catalog.name
    result.status = "normal"
    result.min_mw = numpy.min(forecast.magnitudes)

    return result


def plot_sequential_likelihood(evaluation_results, plot_args=None):
    """
    Plot of likelihood against time.

    Args:
        evaluation_results (list): An evaluation result containing the likelihoods
        plot_args (dict): A configuration dictionary for the plotting.

    Returns:
        Ax object

    """
    if plot_args is None:
        plot_args = {}
    title = plot_args.get("title", None)
    titlesize = plot_args.get("titlesize", None)
    ylabel = plot_args.get("ylabel", None)
    colors = plot_args.get("colors", [None] * len(evaluation_results))
    linestyles = plot_args.get("linestyles", [None] * len(evaluation_results))
    markers = plot_args.get("markers", [None] * len(evaluation_results))
    markersize = plot_args.get("markersize", 1)
    linewidth = plot_args.get("linewidth", 0.5)
    figsize = plot_args.get("figsize", (6, 4))
    timestrs = plot_args.get("timestrs", None)
    if timestrs:
        startyear = [date.fromisoformat(j.split("_")[0]) for j in timestrs][0]
        endyears = [date.fromisoformat(j.split("_")[1]) for j in timestrs]
        years = [startyear] + endyears
    else:
        startyear = 0
        years = numpy.arange(0, len(evaluation_results[0].observed_statistic) + 1)

    seaborn.set_style("white", {"axes.facecolor": ".9", "font.family": "Ubuntu"})
    pyplot.rcParams.update(
        {
            "xtick.bottom": True,
            "axes.labelweight": "bold",
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 9,
        }
    )

    if isinstance(colors, list):
        assert len(colors) == len(evaluation_results)
    elif isinstance(colors, str):
        colors = [colors] * len(evaluation_results)
    if isinstance(linestyles, list):
        assert len(linestyles) == len(evaluation_results)
    elif isinstance(linestyles, str):
        linestyles = [linestyles] * len(evaluation_results)
    if isinstance(markers, list):
        assert len(markers) == len(evaluation_results)
    elif isinstance(markers, str):
        markers = [markers] * len(evaluation_results)

    fig, ax = pyplot.subplots(figsize=figsize)
    for i, result in enumerate(evaluation_results):
        data = [0] + result.observed_statistic
        ax.plot(
            years,
            data,
            color=colors[i],
            linewidth=linewidth,
            linestyle=linestyles[i],
            marker=markers[i],
            markersize=markersize,
            label=result.sim_name,
        )
        ax.set_ylabel(ylabel)
        ax.set_xlim([startyear, None])
        ax.set_title(title, fontsize=titlesize)
        ax.grid(True)
    ax.legend(loc=(1.04, 0), fontsize=7)
    fig.tight_layout()


def magnitude_vs_time(
    catalog: "CSEPCatalog",
    color: str = "steelblue",
    size: int = 15,
    max_size: int = 300,
    power: int = 4,
    alpha: float = 0.5,
    reset_times: bool = False,
    ax: Optional[matplotlib.axes.Axes] = None,
    show: bool = False,
    **kwargs: Any,
) -> matplotlib.axes.Axes:
    """
    Scatter plot of the catalog magnitudes and origin times. The size of each event is scaled
    exponentially by its magnitude using the parameters ``size``, ``max_size`` and ``power``.


    Args:
        catalog (CSEPCatalog): Catalog of seismic events to be plotted.
        color (str, optional): Color of the scatter plot. Defaults to `'steelblue'`.
        size (int, optional): Marker size for the event with the minimum magnitude. Defaults
            to `4`.
        max_size (int, optional): Marker size for the event with the maximum magnitude.
            Defaults to `300`.
        power (int, optional): Power scaling of the scatter sizing. Defaults to `4`.
        alpha (float, optional): Transparency level for the scatter points. Defaults to `0.5`.
        reset_times (bool, optional): If True, x-axis shows time in days since first event.
            Defaults to False.
        ax (matplotlib.axes.Axes, optional): Axis object on which to plot. If not provided, a
            new figure and axis are created. Defaults to `None`.
        show (bool, optional): Whether to display the plot. Defaults to `False`.
        **kwargs:
            Additional keyword arguments to customize the plot:

            - **figsize** (`tuple`): The size of the figure.
            - **title** (`str`): Plot title. Defaults to `None`.
            - **title_fontsize** (`int`): Font size for the plot title.
            - **xlabel** (`str`): Label for the X-axis. Defaults to `'Datetime'`.
            - **xlabel_fontsize** (`int`): Font size for the X-axis label.
            - **ylabel** (`str`): Label for the Y-axis. Defaults to `'Magnitude'`.
            - **ylabel_fontsize** (`int`): Font size for the Y-axis label.
            - **datetime_locator** (`matplotlib.dates.Locator`): Locator for the
              X-axis datetime ticks.
            - **datetime_formatter** (`str` or `matplotlib.dates.Formatter`):
              Formatter for the datetime axis. Defaults to `'%Y-%m-%d'`.
            - **grid** (`bool`): Whether to show grid lines. Defaults to `True`.
            - **tight_layout** (`bool`): Whether to use a tight layout for the figure.
              Defaults to `True`.

    Returns:
        matplotlib.axes.Axes: The Matplotlib axes object with the plotted data.

    """
    # Initialize plot

    DEFAULT_PLOT_ARGS = {
        # General figure/axes handling
        "figsize": (8, 4),
        "tight_layout": True,
        "grid": True,
        "title": None,
        "title_fontsize": 16,
        "xlabel": None,
        "ylabel": None,
        "xlabel_fontsize": 12,
        "ylabel_fontsize": 12,
        # Line/Scatter parameters
        "color": "steelblue",
        "alpha": 0.8,
        "size": 5,
        "marker": "o",
        "markersize": 5,
        "markercolor": "steelblue",
        "markeredgecolor": "black",
        "datetime_locator": AutoDateLocator(),
        "datetime_formatter": DateFormatter("%Y-%m-%d"),
    }
    plot_args = {**DEFAULT_PLOT_ARGS, **kwargs.get("plot_args", {}), **kwargs}
    fig, ax = pyplot.subplots(figsize=plot_args["figsize"]) if ax is None else (ax.figure, ax)

    # Get data
    mag = catalog.data["magnitude"]
    datetimes = catalog.get_datetimes()

    if reset_times:
        # Convert to days since first event
        timestamps = numpy.array([dt.timestamp() for dt in datetimes])
        xdata = (timestamps - timestamps[0]) / SECONDS_PER_DAY
        xlabel = plot_args["xlabel"] or "Days since first event"
    else:
        xdata = datetimes
        xlabel = plot_args["xlabel"] or "Datetime"

    # Plot data
    ax.scatter(
        xdata,
        mag,
        marker="o",
        c=color,
        s=_autosize_scatter(mag, min_size=size, max_size=max_size, power=power),
        alpha=alpha,
    )

    # Set labels and title
    ax.set_xlabel(xlabel, fontsize=plot_args["xlabel_fontsize"])
    ax.set_ylabel(plot_args["ylabel"] or "Magnitude", fontsize=plot_args["ylabel_fontsize"])
    ax.set_title(plot_args["title"], fontsize=plot_args["title_fontsize"])

    # Autoformat ticks and labels
    if not reset_times:
        ax.xaxis.set_major_locator(plot_args["datetime_locator"])
        ax.xaxis.set_major_formatter(plot_args["datetime_formatter"])
        fig.autofmt_xdate()
    ax.grid(plot_args["grid"])

    if plot_args["tight_layout"]:
        fig.tight_layout()
    if show:
        pyplot.show()

    return ax


def plot_matrix_comparative_test(evaluation_results, p=0.05, order=True, plot_args={}):
    """Produces matrix plot for comparative tests for all models (TBI in pyCSEP)

    Args:
        evaluation_results (list of result objects): paired t-test results
        p (float): significance level
        order (bool): columns/rows ordered by ranking

    Returns:
        ax (matplotlib.Axes): handle for figure
    """
    names = [i.sim_name for i in evaluation_results]

    t_value = numpy.array([Tw_i.observed_statistic for Tw_i in evaluation_results])
    t_quantile = numpy.array([Tw_i.quantile[0] for Tw_i in evaluation_results])
    w_quantile = numpy.array([Tw_i.quantile[1] for Tw_i in evaluation_results])
    score = numpy.sum(t_value, axis=1) / t_value.shape[0]

    if order:
        arg_ind = numpy.flip(numpy.argsort(score))
    else:
        arg_ind = numpy.arange(len(score))

    # Flip rows/cols if ordered by value
    data_t = t_value[arg_ind, :][:, arg_ind]
    data_w = w_quantile[arg_ind, :][:, arg_ind]
    data_tq = t_quantile[arg_ind, :][:, arg_ind]
    fig, ax = pyplot.subplots(1, 1, figsize=(7, 6))

    cmap = seaborn.diverging_palette(220, 20, as_cmap=True)
    seaborn.heatmap(
        data_t,
        vmin=-3,
        vmax=3,
        center=0,
        cmap=cmap,
        ax=ax,
        cbar_kws={
            "pad": 0.01,
            "shrink": 0.7,
            "label": "Information Gain",
            "anchor": (0.0, 0.0),
        },
    ),
    ax.set_yticklabels([names[i] for i in arg_ind], rotation="horizontal")
    ax.set_xticklabels([names[i] for i in arg_ind], rotation="vertical")
    for n, i in enumerate(data_tq):
        for m, j in enumerate(i):
            if j > 0 and data_w[n, m] < p:
                ax.scatter(n + 0.5, m + 0.5, marker="o", s=5, color="black")

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            lw=0,
            label=r"T and W significant",
            markerfacecolor="black",
            markeredgecolor="black",
            markersize=4,
        )
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower right",
        bbox_to_anchor=(0.75, 0.0, 0.2, 0.2),
        handletextpad=0,
    )
    pyplot.tight_layout()


def _autosize_scatter(
    values: numpy.ndarray,
    min_size: float = 50.0,
    max_size: float = 400.0,
    power: float = 3.0,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
) -> numpy.ndarray:
    """
    Auto-sizes scatter plot markers based on values.

    Args:
        values (numpy.ndarray): The data values (e.g., magnitude) to base the sizing on.
        min_size (float): The minimum marker size.
        max_size (float): The maximum marker size.
        power (float): The power to apply for scaling.
        min_val (Optional[float]): The minimum value (e.g., magnitude) for normalization.
        max_val (Optional[float]): The maximum value (e.g., magnitude) for normalization.

    Returns:
        numpy.ndarray: The calculated marker sizes.
    """
    min_val = min_val or numpy.min(values)
    max_val = max_val or numpy.max(values)
    normalized_values = ((values - min_val) / (max_val - min_val)) ** power
    marker_sizes = min_size + normalized_values * (max_size - min_size) * bool(power)
    return marker_sizes
