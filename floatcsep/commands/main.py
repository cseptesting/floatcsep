import argparse
import logging

from floatcsep import __version__
from floatcsep.experiment import Experiment, ExperimentComparison
from floatcsep.infrastructure.logger import setup_logger, set_console_log_level
from floatcsep.postprocess.plot_handler import (
    plot_results,
    plot_forecasts,
    plot_catalogs,
    plot_custom,
)
from floatcsep.postprocess.reporting import generate_report, reproducibility_report

setup_logger()
log = logging.getLogger("floatLogger")


def stage(config: str, **_) -> None:
    """
    This function is a preliminary step that stages the models before the experiment is run. It
    is helpful to deal with models that generate forecasts from a source code. Staging means to
    get a source code from a web repository (e.g., zenodo, github) or locate the model in the
    filesystem. It will build the computational environment, install each model dependencies and
    build the source codes.

    Example usage from a terminal:
    ::

        floatcsep stage <config>

    Args:
        config (str): Path to the experiment configuration file (YAML format).
        **_: Additional keyword arguments are not used.

    Returns:
        None
    """
    log.info(f"floatCSEP v{__version__} | Stage")
    exp = Experiment.from_yml(config_yml=config)
    exp.stage_models()

    log.info("Finalized")
    log.debug("")


def run(config: str, **kwargs) -> None:
    """
    Core routine of the floatCSEP workflow. It runs the experiment using the specified YAML
    configuration file. The main steps are:

    1) An Experiment is initialized from the configuration parameters, setting
       the time window, region, testing catalogs, models and evaluations.
    2) Stages the models by accessing the model's forecast files or source code, or by
       detecting them in the filesystem. If necessary, the computational environment is built
       for each model.
    3) According to the experiment and model characteristics (e.g., time-dependent,
       time-windows, evaluations) a set of tasks is created to create/load the forecasts, filter
       testing catalogs, and evaluate each forecasts with its corresponding test catalog.
    4) The tasks are executed according to the experiment logic. Soon to be parallelized.
    5) Postprocessing, such as plotting the catalogs, forecasts, results and user-based
       functions is carried out, as well as creating a human-readable report.
    6) Makes the experiment reproducible, by creating a new configuration file that can be run
       in the future and then compared to old results.

    Example usage from a terminal:
    ::

        floatcsep run <config>

    Args:
        config (str): Path to the experiment configuration file (YAML format).
        **kwargs: Additional configuration parameters to pass to the experiment.

    Returns:
        None
    """
    log.info(f"floatCSEP v{__version__} | Run")
    exp = Experiment.from_yml(config_yml=config, **kwargs)
    exp.stage_models()
    exp.set_tasks()
    exp.run()

    plot_catalogs(experiment=exp)
    plot_forecasts(experiment=exp)
    plot_results(experiment=exp)
    plot_custom(experiment=exp)

    generate_report(experiment=exp)
    exp.make_repr()

    log.info("Finalized")
    log.debug("-------- END OF RUN --------")


def plot(config: str, **kwargs) -> None:
    """
    Generates plots for an already executed experiment. It will not create any forecasts nor run
    any evaluation.

    This function loads the experiment configuration, stages the models to identify the required
    time-windows and results to be plotted.

    Example usage from a terminal:
    ::

        floatcsep plot <config>


    Args:
        config (str): Path to the experiment configuration file (YAML format).
        **kwargs: Additional configuration parameters to pass to the experiment.

    Returns:
        None
    """
    log.info(f"floatCSEP v{__version__} | Plot")

    exp = Experiment.from_yml(config_yml=config, **kwargs)
    exp.stage_models()
    exp.set_tasks()

    plot_catalogs(experiment=exp)
    plot_forecasts(experiment=exp)
    plot_results(experiment=exp)
    plot_custom(experiment=exp)

    generate_report(experiment=exp)

    log.debug("")


def reproduce(config: str, **kwargs) -> None:
    """
    Reproduces the results of a previously run experiment.

    This function re-runs an experiment based on its original configuration and compares the new
    results with the original run. It generates a reproducibility report by comparing the two
    sets of results.

    Example usage from a terminal:
    ::

        floatcsep reproduce <config>

    Args:
        config (str): Path to the experiment configuration file (YAML format).
        **kwargs: Additional configuration parameters to pass to the experiment.

    Returns:
        None
    """
    log.info(f"floatCSEP v{__version__} | Reproduce")

    reproduced_exp = Experiment.from_yml(config, repr_dir="reproduced", **kwargs)
    reproduced_exp.stage_models()
    reproduced_exp.set_tasks()
    reproduced_exp.run()

    original_config = reproduced_exp.original_config
    original_exp = Experiment.from_yml(original_config, rundir=reproduced_exp.original_run_dir)
    original_exp.stage_models()
    original_exp.set_tasks()

    comp = ExperimentComparison(original_exp, reproduced_exp)
    comp.compare_results()

    reproducibility_report(exp_comparison=comp)
    log.info("Finalized")
    log.debug("")


def floatcsep() -> None:
    """
    Entry point for the floatCSEP command-line interface (CLI).

    This function parses command-line arguments and executes the appropriate function
    (`run`, `stage`, `plot`, or `reproduce`) based on the user's input. It also supports
    logging and debugging options

    Example usage from a terminal:
    ::

        floatcsep run <config>

    Args:
        None (arguments are parsed via the command-line interface).

    Returns:
        None
    """
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument(
        "func",
        type=str,
        choices=["run", "stage", "plot", "reproduce"],
        help="Run a calculation",
    )
    parser.add_argument("config", type=str, help="Experiment Configuration file")
    parser.add_argument(
        "-l", "--logging", action="store_true", default=False, help="Don't log experiment"
    )
    parser.add_argument(
        "-t", "--timestamp", action="store_true", default=False, help="Timestamp results"
    )

    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Set the logging level to DEBUG for console output.",
    )
    args = parser.parse_args()

    if hasattr(args, "debug") and args.debug:
        set_console_log_level("DEBUG")

    try:
        func = globals()[args.func]
        args.__delattr__("func")

    except AttributeError:
        raise AttributeError("Function not implemented")
    func(**vars(args))
