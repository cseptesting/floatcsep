from floatcsep import evaluation
from floatcsep import experiment
from floatcsep import model
from floatcsep.infrastructure import engine, environments, registries, repositories, logger
from floatcsep.utils import readers, accessors, helpers
from floatcsep.postprocess import reporting, plot_handler

from importlib.metadata import version, PackageNotFoundError


try:
    __version__ = version("floatcsep")
except PackageNotFoundError:
    __version__ = "0.0.0"
