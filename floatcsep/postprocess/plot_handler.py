import importlib.util
import logging
import os
from typing import TYPE_CHECKING, Union

from cartopy import crs as ccrs
from matplotlib import pyplot

from floatcsep.utils.helpers import (
    timewindow2str,
    magnitude_vs_time,
)

if TYPE_CHECKING:
    from floatcsep.experiment import Experiment

log = logging.getLogger("floatLogger")


def plot_results(experiment: "Experiment", dpi: int = 300, show: bool = False) -> None:
    """
    Plots all evaluation results, according to the plotting functions
    given in the tests configuration file.

    Args:
        experiment: The experiment instance, whose results were already calculated.
        dpi: The resolution of the plots.
        show: Whether to show the plots.
    """
    log.info("Plotting evaluation results")
    time_windows = timewindow2str(experiment.time_windows)
    models = experiment.models
    registry = experiment.registry

    for test in experiment.tests:
        log.info("Plotting results for test '%s'", test.name)
        for func, fargs, fkwargs, mode in zip(
            test.plot_func,
            test.plot_args,
            test.plot_kwargs,
            test.plot_modes,
        ):
            if mode in ("aggregate", "per_model") and test.type in [
                "consistency",
                "comparative",
            ]:
                for time_str in time_windows:
                    results = test.read_results(time_str, models)

                    if mode == "aggregate":
                        fig_path = registry.get_figure_key(time_str, test.name)
                        ax = func(results, plot_args=fargs, **(fkwargs or {}))
                        if "code" in (fargs or {}):
                            exec(fargs["code"])
                        pyplot.savefig(fig_path, dpi=dpi)
                        if show:
                            pyplot.show()

                    elif mode == "per_model":

                        for result, model in zip(results, models):
                            fig_key = f"{test.name}_{model.name}"
                            fig_path = registry.get_figure_key(time_str, fig_key)
                            ax = func(result, plot_args=fargs, **(fkwargs or {}), show=False)
                            if "code" in (fargs or {}):
                                exec(fargs["code"])
                            fig = ax.get_figure()
                            fig.savefig(fig_path, dpi=dpi)
                            if show:
                                pyplot.show()

            elif mode == "sequential" or test.type in [
                "sequential",
                "sequential_comparative",
                "batch",
            ]:
                time_key = time_windows[-1]
                results = test.read_results(time_key, models)
                fig_path = registry.get_figure_key(time_key, test.name)
                ax = func(results, plot_args=fargs, **(fkwargs or {}))
                if "code" in (fargs or {}):
                    exec(fargs["code"])
                pyplot.savefig(fig_path, dpi=dpi)
                if show:
                    pyplot.show()


def plot_forecasts(experiment: "Experiment") -> None:
    """
    Plots and saves all the generated forecasts.

    It can be set specified in the experiment ``config.yml`` as:
    ::

        postprocess:
            plot_forecasts: True



    or by specifying arguments as:
    ::

        postprocess:
            plot_forecasts:
                projection: Mercator
                basemap: google-satellite
                cmap: magma

    The default is ``plot_forecasts: True``

    Args:
        experiment: The experiment instance, whose models were already run and their forecast
            are located in the filesystem/database

    """

    # Parsing plot configuration file
    plot_forecast_config: dict = parse_plot_config(
        experiment.postprocess.get("plot_forecasts", {})
    )
    if not isinstance(plot_forecast_config, dict):
        return

    #####################################
    # Default forecast plotting function.
    #####################################
    log.info("Plotting forecasts")

    # Get the time windows to be plotted. Defaults to only the last time window.
    time_windows = (
        timewindow2str(experiment.time_windows)
        if plot_forecast_config.get("all_time_windows")
        else [timewindow2str(experiment.time_windows[-1])]
    )

    # Get the projection of the plots
    plot_forecast_config["projection"]: ccrs.Projection = parse_projection(
        plot_forecast_config.get("projection")
    )
    plot_forecast_config["title"] = None

    for model in experiment.models:
        for window in time_windows:
            forecast = model.get_forecast(window, region=experiment.region)
            forecast.plot(plot_args=plot_forecast_config,
                           plot_region=plot_forecast_config.get("plot_region", True)
                          )

            if plot_forecast_config.get("catalog"):
                cat_args = plot_forecast_config.get("catalog", {})
                if cat_args is True:
                    cat_args = {}
                overlay_args = {
                    **cat_args,
                    "basemap": plot_forecast_config.get("basemap", None),
                }
                ax = experiment.catalog_repo.get_test_cat(window).plot(
                    ax=ax,
                    extent=ax.get_extent(),
                    plot_args=overlay_args,
                )

            fig = ax.get_figure()
            fig.canvas.draw()

            dpi = plot_forecast_config.get("dpi", 300)
            png_path = experiment.registry.get_figure_key(window, "forecasts", model.name)
            fig.savefig(
                png_path,
                dpi=dpi,
                bbox_inches="tight",
                pad_inches=0.02,
                facecolor="white",
            )
            pyplot.close(fig)


def plot_catalogs(experiment: "Experiment") -> None:
    """
    Plots and saves the testing catalogs.

    It can be set specified in the experiment ``config.yml`` as:
    ::

        postprocess:
            plot_catalog: True



    or by specifying arguments as:
    ::

        postprocess:
            plot_catalog:
                projection: Mercator
                basemap: google-satellite
                markersize: 2

    The default is ``plot_catalog: True``


    Args:
        experiment: The experiment instance, whose catalogs were already accessed and filtered.

    """
    # Parsing plot configuration file
    plot_catalog_config: dict = parse_plot_config(
        experiment.postprocess.get("plot_catalog", {})
    )
    if not isinstance(plot_catalog_config, dict):
        return

    ####################################
    # Default catalog plotting function.
    ####################################
    log.info("Plotting catalogs")

    # Get the projection of the plots
    plot_catalog_config["projection"]: ccrs.Projection = parse_projection(
        plot_catalog_config.get("projection")
    )
    # Get the start and end dates of the experiment (as a string)
    experiment_timewindow = timewindow2str([experiment.start_date, experiment.end_date])

    # Get the catalog for the entire duration of the experiment
    test_catalog = experiment.catalog_repo.filter_catalog(
        start_date=experiment.start_date,
        end_date=experiment.end_date,
        min_mag=experiment.mag_min,
        max_mag=experiment.mag_max,
        min_depth=experiment.depth_min,
        max_depth=experiment.depth_max,
        region=experiment.region,
    )
    # Skip plotting if no events
    if test_catalog.get_number_of_events() == 0:
        log.debug(f"Catalog has zero events in {experiment_timewindow}")
        return
    dpi = plot_catalog_config.get("dpi", 300)

    # Plot catalog map
    ax = test_catalog.plot(plot_args=plot_catalog_config)
    fig = ax.get_figure()
    fig.canvas.draw()
    cat_map_path = experiment.registry.get_figure_key("main_catalog_map")
    fig.savefig(
        cat_map_path,
        dpi=dpi,
        bbox_inches="tight",  # <— trim outer margins
        pad_inches=0.02,  # <— tiny padding to avoid clipping
        facecolor="white",
    )
    pyplot.close(fig)

    # Plot catalog time series vs. magnitude
    ax = magnitude_vs_time(test_catalog)
    fig = ax.get_figure()
    fig.canvas.draw()
    cat_time_path = experiment.registry.get_figure_key("main_catalog_time")
    fig.savefig(
        cat_time_path,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.02,
        facecolor="white",
    )
    pyplot.close(fig)

    # If selected, plot the test catalogs for each of the time windows
    if plot_catalog_config.get("all_time_windows"):
        for tw in experiment.time_windows:
            tw_str = timewindow2str(tw)
            test_catalog = experiment.catalog_repo.get_test_cat(tw_str)

            if test_catalog.get_number_of_events() == 0:
                log.debug(f"Catalog has zero events in {tw_str}. Skip plotting")
                continue

            # Map
            ax = test_catalog.plot(plot_args=plot_catalog_config)
            fig = ax.get_figure()
            fig.canvas.draw()
            cat_map_path = experiment.registry.get_figure_key(tw_str, "catalog_map") + ".png"
            fig.savefig(
                cat_map_path,
                dpi=dpi,
                bbox_inches="tight",
                pad_inches=0.02,
                facecolor="white",
            )
            pyplot.close(fig)

            # Time series
            ax = magnitude_vs_time(test_catalog)
            fig = ax.get_figure()
            fig.canvas.draw()
            cat_time_path = experiment.registry.get_figure_key(tw_str, "catalog_time") + ".png"
            fig.savefig(
                cat_time_path,
                dpi=dpi,
                bbox_inches="tight",
                pad_inches=0.02,
                facecolor="white",
            )
            pyplot.close(fig)


def plot_custom(experiment: "Experiment"):
    """
    Hook for user-based plotting functions. It corresponds to a function within a python file,
    specified in the experiment ``config.yml`` as:
    ::

        postprocess:
            plot_custom: {module}.py:{function}

    Args:
        experiment: The experiment instance, whose models were already run and their forecast
         are located in the filesystem/database

    """
    plot_config = parse_plot_config(experiment.postprocess.get("plot_custom", False))
    if plot_config is None:
        return
    script_path, func_name = plot_config

    log.info(f"Plotting from script {script_path} and function {func_name}")
    script_abs_path = experiment.registry.abs(script_path)
    allowed_directory = os.path.dirname(experiment.registry.abs(experiment.config_file))

    if not os.path.isfile(script_path) or (
        os.path.dirname(script_abs_path) != os.path.realpath(allowed_directory)
    ):

        log.error(f"Script {script_path} is not in the configuration file directory.")
        log.info(
            "\t Skipping plotting. Script can be reallocated and re-run the plotting only"
            " by typing 'floatcsep plot {config}'"
        )
        return

    module_name = os.path.splitext(os.path.basename(script_abs_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, script_abs_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Execute the script securely
    try:
        func = getattr(module, func_name)

    except AttributeError:
        log.error(f"Function {func_name} not found in {script_path}")
        log.info(
            "\t Skipping plotting. Plot script can be modified and re-run the plotting only"
            " by typing 'floatcsep plot {config}'"
        )
        return

    try:
        func(experiment)
    except Exception as e:
        log.error(f"Error executing {func_name} from {script_path}: {e}")
        log.info(
            "\t Skipping plotting. Plot script can be modified and re-run the plotting only"
            " by typing 'floatcsep plot {config}'"
        )
    return


def parse_plot_config(plot_config: Union[dict, str, bool]):
    """
    Parses the configuration of a given plot directive, usually gotten from the experiment
    ``config.yml`` as:
    ::

        postprocess:
            {plot_config}

    Args:
        plot_config: The plotting directive, which can be a dictionary, a boolean, or a string.
            If it is a dictionary, then it is directly returned. If it is a boolean, then
            the default plotting configuration is used. If it is a string, then it is
            expected to be of the form ``{script_path}.py:{func_name}``.


    """
    if plot_config is True:
        return {}

    elif plot_config in (None, False):
        return

    elif isinstance(plot_config, dict):
        return plot_config

    elif isinstance(plot_config, str):
        # Parse the script path and function name
        try:
            script_path, func_name = plot_config.split(".py:")
            script_path += ".py"
            return script_path, func_name
        except ValueError:
            log.error(
                f"Invalid format for custom plot function: {plot_config}. "
                "Try {script_name}.py:{func}"
            )
            log.info(
                "\t Skipping plotting. The script can be modified and re-run the plotting only "
                "by typing 'floatcsep plot {config}'"
            )
            return

    else:
        log.error("Plot configuration not understood. Skipping plotting")
        return


def parse_projection(proj_config: Union[dict, str, bool]):
    """
    Retrieve projection configuration.
    e.g., as defined in the config file:
    ::

        projection:
            Mercator:
                central_longitude: 0.0

    """
    if proj_config is None:
        return ccrs.PlateCarree(central_longitude=0.0)

    if isinstance(proj_config, dict):
        proj_name, proj_args = next(iter(proj_config.items()))
    else:
        proj_name, proj_args = proj_config, {}

    if not isinstance(proj_name, str):
        return ccrs.PlateCarree(central_longitude=0.0)

    return getattr(ccrs, proj_name, ccrs.PlateCarree)(**proj_args)
