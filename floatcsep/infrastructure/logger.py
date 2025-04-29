import sys
import os
import logging.config
import warnings

LOG_NAME = "experiment.log"
LOGGING_CONFIG = {
    "version": 1,
    "formatters": {
        "default": {
            "format": "%(asctime)s %(levelname)s - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "console": {
            "formatter": "default",
            "level": "INFO",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        }
    },
    "loggers": {"floatLogger": {"level": "DEBUG", "handlers": ["console"], "propagate": False}},
    "root": {"level": "INFO", "handlers": ["console"]},
}


def add_fhandler(filename):
    formatter = logging.Formatter(
        fmt=LOGGING_CONFIG["formatters"]["default"]["format"],
        datefmt=LOGGING_CONFIG["formatters"]["default"]["datefmt"],
    )
    fhandler = logging.FileHandler(filename)
    fhandler.setFormatter(formatter)
    fhandler.setLevel(logging.DEBUG)
    logging.getLogger("floatLogger").addHandler(fhandler)


def is_sphinx_build():
    # Check if Sphinx is running
    return "sphinx" in sys.argv[0] or os.getenv("SPHINX_BUILD") is not None


def setup_logger():
    if is_sphinx_build():
        # Reduce logging or disable it during Sphinx builds
        logging.basicConfig(level=logging.WARNING)
    else:
        logging.config.dictConfig(LOGGING_CONFIG)
        logging.getLogger("numexpr").setLevel(logging.WARNING)
        logging.getLogger("matplotlib").setLevel(logging.CRITICAL)
        # numpy.seterr(all="ignore")
        warnings.filterwarnings("ignore")


def set_console_log_level(log_level):
    """Set the console log level based on the user's CLI input."""
    logger = logging.getLogger("floatLogger")
    # Update the log level for the console handler
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.setLevel(log_level)




def log_models_tree(log, experiment_registry, time_windows):
    """
    Logs the forecasts for all models managed by this ExperimentFileRegistry.
    """
    log.debug("===================")
    log.debug(f" Total Time Windows: {len(time_windows)}")
    for model_name, registry in experiment_registry.model_registries.items():
        log.debug(f"  Model: {model_name}")
        exists_group = []
        not_exist_group = []

        for timewindow, filepath in registry.forecasts.items():
            if registry.forecast_exists(timewindow):
                exists_group.append(timewindow)
            else:
                not_exist_group.append(timewindow)

        log.debug(f"    Existing forecasts: {len(exists_group)}")
        log.debug(f"    Missing forecasts: {len(not_exist_group)}")
        for timewindow in not_exist_group:
            log.debug(f"      Time Window: {timewindow}")
    log.debug("===================")


def log_results_tree(log, experiment_registry):
    """
    Logs a summary of the results dictionary, sorted by test.
    For each test and time window, it logs whether all models have results,
    or if some results are missing, and specifies which models are missing.
    """
    log.debug("===================")

    total_results = results_exist_count = results_not_exist_count = 0

    # Get all unique test names and sort them
    all_tests = sorted(
        {test_name for tests in experiment_registry.results.values() for test_name in tests}
    )

    for test_name in all_tests:
        log.debug(f"Test: {test_name}")
        for timewindow, tests in experiment_registry.results.items():
            if test_name in tests:
                models = tests[test_name]
                missing_models = []

                for model_name, result_path in models.items():
                    total_results += 1
                    result_full_path = experiment_registry.get_result_key(timewindow, test_name, model_name)
                    if os.path.exists(result_full_path):
                        results_exist_count += 1
                    else:
                        results_not_exist_count += 1
                        missing_models.append(model_name)

                if not missing_models:
                    log.debug(f"  Time Window: {timewindow} - All models evaluated.")
                else:
                    log.debug(
                        f"  Time Window: {timewindow} - Missing results for models: "
                        f"{', '.join(missing_models)}"
                    )

    log.debug(f"Total Results: {total_results}")
    log.debug(f"Results that Exist: {results_exist_count}")
    log.debug(f"Results that Do Not Exist: {results_not_exist_count}")
    log.debug("===================")