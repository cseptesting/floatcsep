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


def stage(config, **_):

    log.info(f"floatCSEP v{__version__} | Stage")
    exp = Experiment.from_yml(config_yml=config)
    exp.stage_models()

    log.info("Finalized")
    log.debug("")


def run(config, **kwargs):

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
    log.debug(f"-------- END OF RUN --------")


def plot(config, **kwargs):

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


def reproduce(config, **kwargs):

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


def floatcsep():
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
