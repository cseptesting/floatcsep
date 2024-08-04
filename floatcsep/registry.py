import logging
import os
from abc import ABC, abstractmethod
from datetime import datetime
from os.path import join, abspath, relpath, normpath, dirname, exists
from typing import Sequence, Union

from floatcsep.utils import timewindow2str

log = logging.getLogger("floatLogger")


class BaseFileRegistry(ABC):

    def __init__(self, workdir: str):
        self.workdir = workdir

    @staticmethod
    def _parse_arg(arg):
        if isinstance(arg, (list, tuple)):
            return timewindow2str(arg)
        elif isinstance(arg, str):
            return arg
        elif hasattr(arg, "name"):
            return arg.name
        elif hasattr(arg, "__name__"):
            return arg.__name__
        else:
            raise Exception("Arg is not found")

    @abstractmethod
    def as_dict(self):
        pass

    @abstractmethod
    def build_tree(self, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def get_path(self, *args):
        pass

    def abs(self, *paths: Sequence[str]) -> str:
        _path = normpath(abspath(join(self.workdir, *paths)))
        return _path

    def abs_dir(self, *paths: Sequence[str]) -> str:
        _path = normpath(abspath(join(self.workdir, *paths)))
        _dir = dirname(_path)
        return _dir

    def rel(self, *paths: Sequence[str]) -> str:
        """Gets the relative path of a file, when it was defined relative to.

        the experiment working dir.
        """

        _abspath = normpath(abspath(join(self.workdir, *paths)))
        _relpath = relpath(_abspath, self.workdir)
        return _relpath

    def rel_dir(self, *paths: Sequence[str]) -> str:
        """Gets the absolute path of a file, when it was defined relative to.

        the experiment working dir.
        """

        _path = normpath(abspath(join(self.workdir, *paths)))
        _dir = dirname(_path)

        return relpath(_dir, self.workdir)

    def file_exists(self, *args):
        file_abspath = self.get_path(*args)
        return exists(file_abspath)


class ForecastRegistry(BaseFileRegistry):
    def __init__(
        self,
        workdir: str,
        path: str,
        database: str = None,
        args_file: str = None,
        input_cat: str = None,
    ):
        super().__init__(workdir)

        self.path = path
        self.database = database
        self.args_file = args_file
        self.input_cat = input_cat
        self.forecasts = {}
        self.inventory = {}

    def get_path(self, *args):
        val = self.__dict__
        for i in args:
            parsed_arg = self._parse_arg(i)
            val = val[parsed_arg]
        return self.abs(val)

    @property
    def dir(self) -> str:
        """
        Returns:

            The directory containing the model source.
        """
        if os.path.isdir(self.get_path("path")):
            return self.get_path("path")
        else:
            return os.path.dirname(self.get_path("path"))

    @property
    def fmt(self) -> str:
        if self.database:
            return os.path.splitext(self.database)[1][1:]
        else:
            return os.path.splitext(self.path)[1][1:]

    def as_dict(self):
        return {
            "workdir": self.workdir,
            "path": self.path,
            "database": self.database,
            "args_file": self.args_file,
            "input_cat": self.input_cat,
            "forecasts": self.forecasts,
            "inventory": self.inventory,
        }

    def forecast_exists(self, timewindow: Union[str, list]):

        if isinstance(timewindow, str):
            return self.file_exists("forecasts", timewindow)
        else:
            return [self.file_exists("forecasts", i) for i in timewindow]

    def build_tree(
        self,
        timewindows: Sequence[Sequence[datetime]] = None,
        model_class: str = "TimeIndependentModel",
        prefix=None,
        args_file=None,
        input_cat=None,
    ) -> None:
        """
        Creates the run directory, and reads the file structure inside.

        Args:
            timewindows (list(str)): List of time windows or strings.
            model_class (str): Model's class name
            prefix (str): prefix of the model forecast filenames if TD
            args_file (str, bool): input arguments path of the model if TD
            input_cat (str, bool): input catalog path of the model if TD

        Returns:
            run_folder: Path to the run.
             exists: flag if forecasts, catalogs and test_results if they
             exist already
             target_paths: flag to each element of the gefe (catalog and
             evaluation results)
        """

        windows = timewindow2str(timewindows)

        if model_class == "TimeIndependentModel":
            fname = self.database if self.database else self.path
            self.forecasts = {win: fname for win in windows}
            self.inventory = {win: exists(self.forecasts[win]) for win in windows}

        elif model_class == "TimeDependentModel":

            args = args_file if args_file else join("input", "args.txt")
            self.args_file = join(self.path, args)
            input_cat = input_cat if input_cat else join("input", "catalog.csv")
            self.input_cat = join(self.path, input_cat)
            # grab names for creating directories
            subfolders = ["input", "forecasts"]
            dirtree = {folder: self.abs(self.path, folder) for folder in subfolders}

            # create directories if they don't exist
            for _, folder_ in dirtree.items():
                os.makedirs(folder_, exist_ok=True)

            # set forecast names
            self.forecasts = {
                win: join(dirtree["forecasts"], f"{prefix}_{win}.csv") for win in windows
            }

            self.inventory = {
                win: any(file for file in list(os.listdir(dirtree["forecasts"])))
                for win in windows
            }


class ExperimentRegistry(BaseFileRegistry):
    def __init__(self, workdir: str, rundir: str = "results"):
        super().__init__(workdir)
        self.rundir = rundir
        self.paths = {}
        self.result_exists = {}

    def get_path(self, *args):
        pass

    def __call__(self, *args):
        val = self.paths
        for i in args:
            parsed_arg = self._parse_arg(i)
            val = val[parsed_arg]
        return self.abs(self.rundir, val)

    def as_dict(self):
        return self.workdir

    def build_tree(self, timewindows=None, models=None, tests=None) -> None:
        """
        Creates the run directory, and reads the file structure inside.

        Args:
            timewindows: List of time windows, or representing string.
            models: List of models or model names
            tests: List of tests or test names

        Returns:
            run_folder: Path to the run.
             exists: flag if forecasts, catalogs and test_results if they
             exist already
             target_paths: flag to each element of the gefe (catalog and
             evaluation results)
        """
        # grab names for creating directories
        windows = timewindow2str(timewindows)
        models = [i.name for i in models]
        tests = [i.name for i in tests]

        # todo create datetime parser for filenames
        # todo find better way to name paths

        # Determine required directory structure for run
        # results > time_window > cats / evals / figures

        run_folder = self.rundir

        subfolders = ["catalog", "evaluations", "figures", "forecasts"]
        dirtree = {
            win: {folder: self.abs(run_folder, win, folder) for folder in subfolders}
            for win in windows
        }

        # create directories if they don't exist
        for tw, tw_folder in dirtree.items():
            for _, folder_ in tw_folder.items():
                os.makedirs(folder_, exist_ok=True)

        # Check existing files
        files = {
            win: {name: list(os.listdir(path)) for name, path in windir.items()}
            for win, windir in dirtree.items()
        }

        file_exists = {
            win: {
                "forecasts": False,
                "catalog": any(file for file in files[win]["catalog"]),
                "evaluations": {
                    test: {
                        model: any(
                            f"{test}_{model}.json" in file for file in files[win]["evaluations"]
                        )
                        for model in models
                    }
                    for test in tests
                },
            }
            for win in windows
        }

        target_paths = {
            "config": "repr_config.yml",
            "catalog_figure": "catalog",
            "magnitude_time": "events",
            **{
                win: {
                    "catalog": join(win, "catalog", "test_catalog.json"),
                    "evaluations": {
                        test: {
                            model: join(win, "evaluations", f"{test}_{model}.json")
                            for model in models
                        }
                        for test in tests
                    },
                    "figures": {
                        **{test: join(win, "figures", f"{test}") for test in tests},
                        "catalog": join(win, "figures", "catalog"),
                        "magnitude_time": join(win, "figures", "magnitude_time"),
                    },
                    "forecasts": {
                        model: join(win, "forecasts", f"{model}") for model in models
                    },
                }
                for win in windows
            },
        }
        self.paths = target_paths
        self.result_exists = file_exists
