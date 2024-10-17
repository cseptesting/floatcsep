import json
import logging
import os
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Callable, Union, Sequence

import git
from csep.core.forecasts import GriddedForecast, CatalogForecast

from floatcsep.utils.accessors import from_zenodo, from_git
from floatcsep.infrastructure.environments import EnvironmentFactory
from floatcsep.utils.readers import ForecastParsers, HDF5Serializer
from floatcsep.infrastructure.registries import ForecastRegistry
from floatcsep.infrastructure.repositories import ForecastRepository
from floatcsep.utils.helpers import timewindow2str, str2timewindow, parse_nested_dicts

log = logging.getLogger("floatLogger")


class Model(ABC):
    """
    The Model class represents a forecast generating system. It can represent a source code, a
    collection or a single forecast, etc. A Model can be instantiated from either the filesystem
    or host repositories.

    Args:
        name (str): Name of the model
        model_path (str): Relative path of the model (file or code) to the work directory
        zenodo_id (int): Zenodo ID or record of the Model
        giturl (str): Link to a git repository
        repo_hash (str): Specific commit/branch/tag hash.
        authors (list[str]): Authors' names metadata
        doi: Digital Object Identifier metadata:
    """

    def __init__(
        self,
        name: str,
        zenodo_id: int = None,
        giturl: str = None,
        repo_hash: str = None,
        authors: List[str] = None,
        doi: str = None,
        **kwargs,
    ):

        self.name = name
        self.zenodo_id = zenodo_id
        self.giturl = giturl
        self.repo_hash = repo_hash
        self.authors = authors
        self.doi = doi

        self.registry = None
        self.forecasts = {}

        self.force_stage = False
        self.__dict__.update(**kwargs)

    @abstractmethod
    def stage(self, timewindows=None) -> None:
        """Prepares the stage for a model run."""
        pass

    @abstractmethod
    def get_forecast(self, tstring: str, region=None):
        """Retrieves the forecast based on a time window."""
        pass

    @abstractmethod
    def create_forecast(self, tstring: str, **kwargs) -> None:
        """Creates a forecast based on the model's logic."""
        pass

    def get_source(self, zenodo_id: int = None, giturl: str = None, **kwargs) -> None:
        """
        Search, download or clone the model source in the filesystem, zenodo.

        and git, respectively. Identifies if the instance path points to a file
        or to its parent directory

        Args:
            zenodo_id (int): Zenodo identifier of the repository. Usually as
             `https://zenodo.org/record/{zenodo_id}`
            giturl (str): git remote repository URL from which to clone the
             source
            **kwargs: see :func:`~floatcsep.utils.from_zenodo` and
             :func:`~floatcsep.utils.from_git`
        """

        if zenodo_id:
            log.info(f"Retrieving model {self.name} from zenodo id: " f"{zenodo_id}")
            try:
                from_zenodo(
                    zenodo_id,
                    self.registry.dir if self.registry.fmt else self.registry.path,
                    force=True,
                )
            except (KeyError, TypeError) as msg:
                raise KeyError(f"Zenodo identifier is not valid: {msg}")

        elif giturl:
            log.info(f"Retrieving model {self.name} from git url: " f"{giturl}")
            try:
                from_git(
                    giturl,
                    self.registry.dir if self.registry.fmt else self.registry.path,
                    **kwargs,
                )
            except (git.NoSuchPathError, git.CommandError) as msg:
                raise git.NoSuchPathError(f"git url was not found {msg}")
        else:
            raise FileNotFoundError("Model has no path or identified")

        if not os.path.exists(self.registry.dir) or not os.path.exists(
            self.registry.get("path")
        ):
            raise FileNotFoundError(
                f"Directory '{self.registry.dir}' or file {self.registry}' do not exist. "
                f"Please check the specified 'path' matches the repo "
                f"structure"
            )

    def as_dict(self, excluded=("name", "repository", "workdir")):
        """
        Returns:
            Dictionary with relevant attributes. Model can be re-instantiated from this dict
        """

        list_walk = [
            (i, j) for i, j in sorted(self.__dict__.items()) if not i.startswith("_") and j
        ]

        dict_walk = {i: j for i, j in list_walk if i not in excluded}
        dict_walk["path"] = dict_walk.pop("registry").path

        return {self.name: parse_nested_dicts(dict_walk)}

    @classmethod
    def from_dict(cls, record: dict, **kwargs):
        """
        Returns a Model instance from a dictionary containing the required attributes. Can be
        used to quickly instantiate from a .yml file.

        Args:
            record (dict): Contains the keywords from the ``__init__`` method.

                Note:
                    Must have either an explicit key `name`, or it must have
                    exactly one key with the model's name, whose values are
                    the remaining ``__init__`` keywords.

        Returns:
            A Model instance
        """

        if "name" in record.keys():
            return cls(**record)
        elif len(record) != 1:
            raise IndexError("A single model has not been passed")
        name = next(iter(record))
        return cls(name=name, **record[name], **kwargs)

    @classmethod
    def factory(cls, model_cfg: dict) -> "Model":
        """Factory method. Instantiate first on any explicit option provided in the model
        configuration.
        """
        model_path = [*model_cfg.values()][0]["model_path"]
        workdir = [*model_cfg.values()][0].get("workdir", "")
        model_class = [*model_cfg.values()][0].get("class", "")

        if model_class in ("ti", "time_independent"):
            return TimeIndependentModel.from_dict(model_cfg)

        elif model_class in ("td", "time_dependent"):
            return TimeDependentModel.from_dict(model_cfg)

        if os.path.isfile(os.path.join(workdir, model_path)):
            return TimeIndependentModel.from_dict(model_cfg)

        elif "func" in [*model_cfg.values()][0]:
            return TimeDependentModel.from_dict(model_cfg)

        else:
            return TimeIndependentModel.from_dict(model_cfg)


class TimeIndependentModel(Model):
    """
    A Model whose forecast is invariant in time. A TimeIndependentModel is commonly represented
    by a single forecast as static data.

    """

    def __init__(self, name: str, model_path: str, forecast_unit=1, store_db=False, **kwargs):
        """

        Args:
            name (str): The name of the model.
            model_path (str): The path to the model data.
            forecast_unit (float): The unit of time for the forecast.
            store_db (bool): flag to indicate whether to store the model in a database.

        """
        super().__init__(name, **kwargs)

        self.forecast_unit = forecast_unit
        self.store_db = store_db
        self.registry = ForecastRegistry(kwargs.get("workdir", os.getcwd()), model_path)
        self.repository = ForecastRepository.factory(
            self.registry, model_class=self.__class__.__name__, **kwargs
        )

    def stage(self, timewindows: Sequence[Sequence[datetime]] = None) -> None:
        """
        Acquire the forecast data if it is not in the file system. Sets the paths internally
        (or database pointers) to the forecast data.

        Args:
            timewindows (list): time_windows that the forecast data represents.
        """

        if self.force_stage or not self.registry.file_exists("path"):
            os.makedirs(self.registry.dir, exist_ok=True)
            self.get_source(self.zenodo_id, self.giturl, branch=self.repo_hash)

        if self.store_db:
            self.init_db()

        self.registry.build_tree(timewindows=timewindows, model_class=self.__class__.__name__)

    def init_db(self, dbpath: str = "", force: bool = False) -> None:
        """
        Initializes the database if `use_db` is True. If the model source is a file,
        serializes the forecast into a HDF5 file. If source is a generating function or code,
        creates an empty DB.

        Args:
            dbpath (str): Path to drop the HDF5 database. Defaults to same path
             replaced with an `hdf5` extension
            force (bool): Forces the serialization even if the DB already
             exists
        """

        parser = getattr(ForecastParsers, self.registry.fmt)
        rates, region, mag = parser(self.registry.get("path"))
        db_func = HDF5Serializer.grid2hdf5

        if not dbpath:
            dbpath = self.registry.path.replace(self.registry.fmt, "hdf5")
            self.registry.database = dbpath

        if not os.path.isfile(self.registry.abs(dbpath)) or force:
            log.info(f"Serializing model {self.name} into HDF5 format")
            db_func(
                rates,
                region,
                mag,
                hdf5_filename=self.registry.abs(dbpath),
                unit=self.forecast_unit,
            )

    def get_forecast(
        self, tstring: Union[str, list] = None, region=None
    ) -> Union[GriddedForecast, List[GriddedForecast]]:
        """Wrapper that just returns a forecast when requested."""

        return self.repository.load_forecast(
            tstring, name=self.name, region=region, forecast_unit=self.forecast_unit
        )

    def create_forecast(self, tstring: str, **kwargs) -> None:
        """
        Creates a forecast from the model source and a given time window.

        Note:
            Dummy function for this class, although eventually could also be a source
            code (e.g., a Smoothed-Seismicity-Model built from the input-catalog).

        """
        return


class TimeDependentModel(Model):
    """
    Model that creates varying forecasts depending on a time window. Requires either a
    collection of Forecasts or a function/source code that returns a Forecast.
    """

    def __init__(
        self,
        name: str,
        model_path: str,
        func: Union[str, Callable] = None,
        func_kwargs: dict = None,
        **kwargs,
    ) -> None:
        """

        Args:
            name: The name of the model
            model_path: The path to either the source code, or the folder containing static
                forecasts.
            func: A function/command that runs the model.
            func_kwargs: The keyword arguments to run the model. They are usually (over)written
                into the file `{model_path}/input/{args_file}`
            **kwargs: Additional keyword parameters, such as a ``prefix`` (str) for the
                resulting forecast file paths, ``args_file`` (str) as the path for the model
                arguments file or ``input_cat`` that indicates where the input catalog will be
                placed for the model.

        """
        super().__init__(name, **kwargs)

        self.func = func
        self.func_kwargs = func_kwargs or {}

        self.registry = ForecastRegistry(kwargs.get("workdir", os.getcwd()), model_path)
        self.repository = ForecastRepository.factory(
            self.registry, model_class=self.__class__.__name__, **kwargs
        )
        self.build = kwargs.get("build", None)
        self.force_build = kwargs.get("force_build", False)
        if self.func:
            self.environment = EnvironmentFactory.get_env(
                self.build, self.name, self.registry.abs(model_path)
            )

    def stage(self, timewindows=None) -> None:
        """
        Core method to interface a model with the experiment.

        1) Get the model from filesystem, Zenodo or Git. Prepares the directory
        2) If source code, creates the computational environment (conda, venv or Docker)
        3) Prepares the registry tree: filepaths/keys corresponding to existing forecasts
           and those to be generated, as well as input catalog and arguments file.

        """
        if self.force_stage or not self.registry.file_exists("path"):
            os.makedirs(self.registry.dir, exist_ok=True)
            self.get_source(self.zenodo_id, self.giturl, branch=self.repo_hash)

        if hasattr(self, "environment"):
            self.environment.create_environment(force=self.force_build)

        self.registry.build_tree(
            timewindows=timewindows,
            model_class=self.__class__.__name__,
            prefix=self.__dict__.get("prefix", self.name),
            args_file=self.__dict__.get("args_file", None),
            input_cat=self.__dict__.get("input_cat", None),
        )

    def get_forecast(
        self, tstring: Union[str, list] = None, region=None
    ) -> Union[GriddedForecast, CatalogForecast, List[GriddedForecast], List[CatalogForecast]]:
        """
        Wrapper that returns a forecast, by accessing the model's forecast repository.

        Note:
            The argument ``tstring`` is formatted according to how the Experiment
            handles timewindows, specified in the functions
            :func:`~floatcsep.utils.helpers.timewindow2str` and
            :func:`~floatcsep.utils.helpers.str2timewindow`

        Args:
            tstring: String representing the start and end of the forecast,
                formatted as 'YY1-MM1-DD1_YY2-MM2-DD2'.
            region: String representing the region for which to return a forecast.
                If None, will return a forecast for all regions.

        """
        return self.repository.load_forecast(tstring, region=region)

    def create_forecast(self, tstring: str, **kwargs) -> None:
        """
        Creates a forecast from the model source and a given time window.

        Note:
            The argument ``tstring`` is formatted according to how the Experiment
            handles timewindows, specified in the functions
            :func:`~floatcsep.utils.helpers.timewindow2str` and
            :func:`~floatcsep.utils.helpers.str2timewindow`

        Args:
            tstring: String representing the start and end of the forecast,
                formatted as 'YY1-MM1-DD1_YY2-MM2-DD2'.
            **kwargs:
        """
        start_date, end_date = str2timewindow(tstring)

        # Model src is a func or binary
        if not kwargs.get("force") and self.registry.forecast_exists(tstring):
            log.info(f"Forecast for {tstring} of model {self.name} already exists")
            return

        self.prepare_args(start_date, end_date, **kwargs)
        log.info(
            f"Running {self.name} using {self.environment.__class__.__name__}:"
            f" {timewindow2str([start_date, end_date])}"
        )
        self.environment.run_command(f"{self.func} \"{self.registry.get('args_file')}\"")

    def prepare_args(self, start: datetime, end: datetime, **kwargs) -> None:
        """
        When the model is a source code, the args file is a plain text file with the required
        input arguments. At minimum, it consists of the start and end of the forecast
        timewindow, but it can also contain other arguments (e.g., minimum magnitude, number of
        simulations, cutoff learning magnitude, etc.)

        Args:
            start: start date of the forecast timewindow
            end: end date of the forecast timewindow
            **kwargs: represents additional model arguments (name/value pair)

        """
        filepath = self.registry.get("args_file")
        fmt = os.path.splitext(filepath)[1]

        if fmt == ".txt":

            def replace_arg(arg, val, fp):
                with open(fp, "r") as filearg_:
                    lines = filearg_.readlines()

                pattern_exists = False
                for k, line in enumerate(lines):
                    if line.startswith(arg):
                        lines[k] = f"{arg} = {val}\n"
                        pattern_exists = True
                        break  # assume there's only one occurrence of the key
                if not pattern_exists:
                    lines.append(f"{arg} = {val}\n")
                with open(fp, "w") as file:
                    file.writelines(lines)

            replace_arg("start_date", start.isoformat(), filepath)
            replace_arg("end_date", end.isoformat(), filepath)
            for i, j in kwargs.items():
                replace_arg(i, j, filepath)

        elif fmt == ".json":
            with open(filepath, "r") as file_:
                args = json.load(file_)
            args["start_date"] = start.isoformat()
            args["end_date"] = end.isoformat()

            args.update(kwargs)

            with open(filepath, "w") as file_:
                json.dump(args, file_, indent=2)
