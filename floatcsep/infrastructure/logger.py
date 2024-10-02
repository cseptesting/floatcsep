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
