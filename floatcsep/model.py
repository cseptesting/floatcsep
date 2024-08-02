import json
import logging
import os
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Callable, Union, Mapping, Sequence

import csep
import git
import numpy
from csep.core.forecasts import GriddedForecast, CatalogForecast
from csep.utils.time_utils import decimal_year

from floatcsep.accessors import from_zenodo, from_git
from floatcsep.environments import EnvironmentFactory
from floatcsep.readers import ForecastParsers, HDF5Serializer
from floatcsep.registry import ModelTree
from floatcsep.utils import timewindow2str, str2timewindow

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
        model_path: str,
        zenodo_id: int = None,
        giturl: str = None,
        repo_hash: str = None,
        authors: List[str] = None,
        doi: str = None,
        **kwargs,
    ):

        self.name = name
        self.model_path = model_path
        self.zenodo_id = zenodo_id
        self.giturl = giturl
        self.repo_hash = repo_hash
        self.authors = authors
        self.doi = doi

        self.path = None
        self.forecasts = {}

        self.__dict__.update(**kwargs)

    @property
    def dir(self) -> str:
        """
        Returns:
            The directory containing the model source.
        """
        if os.path.isdir(self.path("path")):
            return self.path("path")
        else:
            return os.path.dirname(self.path("path"))

    @abstractmethod
    def stage(self, timewindows=None) -> None:
        """Prepares the stage for a model run. Can be"""
        pass

    @abstractmethod
    def get_forecast(self, tstring: str, region=None):
        """Retrieves the forecast based on a time window."""
        pass

    @abstractmethod
    def create_forecast(self, tstring: str, **kwargs) -> None:
        """Creates a forecast based on the model's logic."""
        pass

    def get_source(
        self, zenodo_id: int = None, giturl: str = None, force: bool = False, **kwargs
    ) -> None:
        """
        Search, download or clone the model source in the filesystem, zenodo.

        and git, respectively. Identifies if the instance path points to a file
        or to its parent directory

        Args:
            zenodo_id (int): Zenodo identifier of the repository. Usually as
             `https://zenodo.org/record/{zenodo_id}`
            giturl (str): git remote repository URL from which to clone the
             source
            force (bool): Forces to re-query the model from a web repository
            **kwargs: see :func:`~floatcsep.utils.from_zenodo` and
             :func:`~floatcsep.utils.from_git`
        """
        if os.path.exists(self.path("path")) and not force:
            return

        os.makedirs(self.dir, exist_ok=True)

        if zenodo_id:
            log.info(f"Retrieving model {self.name} from zenodo id: " f"{zenodo_id}")
            try:
                from_zenodo(
                    zenodo_id,
                    self.dir if self.path.fmt else self.path("path"),
                    force=force,
                )
            except (KeyError, TypeError) as msg:
                raise KeyError(f"Zenodo identifier is not valid: {msg}")

        elif giturl:
            log.info(f"Retrieving model {self.name} from git url: " f"{giturl}")
            try:
                from_git(giturl, self.dir if self.path.fmt else self.path("path"), **kwargs)
            except (git.NoSuchPathError, git.CommandError) as msg:
                raise git.NoSuchPathError(f"git url was not found {msg}")
        else:
            raise FileNotFoundError("Model has no path or identified")

        if not os.path.exists(self.dir) or not os.path.exists(self.path("path")):
            raise FileNotFoundError(
                f"Directory '{self.dir}' or file {self.path}' do not exist. "
                f"Please check the specified 'path' matches the repo "
                f"structure"
            )

    def as_dict(self, excluded=("name", "forecasts", "workdir")):
        """
        Returns:
            Dictionary with relevant attributes. Model can be re-instantiated from this dict
        """

        def _get_value(x):
            # For each element type, transforms to desired string/output
            if hasattr(x, "as_dict"):
                # e.g. model, evaluation, filetree, etc.
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
                return {
                    item: iter_attr(val_)
                    for item, val_ in val.items()
                    if ((item not in excluded) and val_)
                }
            elif isinstance(val, Sequence) and not isinstance(val, str):
                return [iter_attr(i) for i in val]
            else:
                return _get_value(val)

        list_walk = [
            (i, j) for i, j in sorted(self.__dict__.items()) if not i.startswith("_") and j
        ]

        dict_walk = {i: j for i, j in list_walk}

        return {self.name: iter_attr(dict_walk)}

    @classmethod
    def from_dict(cls, record: dict, **kwargs):
        """
        Returns a Model instance from a dictionary containing the required atrributes. Can be
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


class TimeIndependentModel(Model):
    """
    A Model that does not change in time, commonly represented by static data.

    Args
        name (str): The name of the model.
        model_path (str): The path to the model data.
        forecast_unit (float): The unit of time for the forecast.
        store_db (bool): flag to indicate whether to store the model in a database.
    """

    def __init__(self, name: str, model_path: str, forecast_unit=1, store_db=False, **kwargs):
        super().__init__(name, model_path, **kwargs)
        self.forecast_unit = forecast_unit
        self.store_db = store_db

        self.path = ModelTree(kwargs.get("workdir", os.getcwd()), model_path)

    def init_db(self, dbpath: str = "", force: bool = False) -> None:
        """
        Initializes the database if `use_db` is True.

        If the model source is a file, serializes the forecast into a HDF5 file. If source is a
        generating function or code, creates an empty DB

        Args:
            dbpath (str): Path to drop the HDF5 database. Defaults to same path
             replaced with an `hdf5` extension
            force (bool): Forces the serialization even if the DB already
             exists
        """

        parser = getattr(ForecastParsers, self.path.fmt)
        rates, region, mag = parser(self.path("path"))
        db_func = HDF5Serializer.grid2hdf5

        if not dbpath:
            dbpath = self.path.path.replace(self.path.fmt, "hdf5")
            self.path.database = dbpath

        if not os.path.isfile(self.path.abs(dbpath)) or force:
            log.info(f"Serializing model {self.name} into HDF5 format")
            db_func(
                rates,
                region,
                mag,
                hdf5_filename=self.path.abs(dbpath),
                unit=self.forecast_unit,
            )

    def rm_db(self) -> None:
        """Clean up the generated HDF5 File."""
        pass

    def stage(self, timewindows: Union[str, List[datetime]] = None) -> None:
        """
        Acquire the forecast data if it is not in the file system.
        Sets internally the paths (or database pointers) to the forecast data.

        Args:
            timewindows (str, list): time_windows that the forecast data represents.

        """
        self.get_source(self.zenodo_id, self.giturl, branch=self.repo_hash)
        if self.store_db:
            self.init_db()

        self.path.build_tree(
            timewindows=timewindows,
            model_class="ti",
            prefix=self.__dict__.get("prefix", self.name),
        )

    def get_forecast(
        self, tstring: Union[str, list] = None, region=None
    ) -> Union[GriddedForecast, CatalogForecast, List[GriddedForecast], List[CatalogForecast]]:
        """
        Wrapper that just returns a forecast when requested.
        """

        if isinstance(tstring, str):
            # If only one time_window string is passed
            try:
                # If they are retrieved from the Evaluation class
                return self.forecasts[tstring]
            except KeyError:
                # In case they are called from postprocess
                self.create_forecast(tstring)
                return self.forecasts[tstring]
        else:
            # If multiple time_window strings are passed
            forecasts = []
            for tw in tstring:
                if tw in self.forecasts.keys():
                    forecasts.append(self.forecasts[tw])
            if not forecasts:
                raise KeyError(f"Forecasts {*tstring,} have not been created yet")
            return forecasts

    def create_forecast(self, tstring: str, **kwargs) -> None:
        """
        Creates a forecast from the model source and a given time window.

        Note:
            The argument `tstring` is formatted according to how the Experiment
            handles timewindows, specified in the functions
            :func:'floatcsep.utils.timewindow2str` and
            :func:'floatcsep.utils.str2timewindow`

        Args:
            tstring: String representing the start and end of the forecast,
                formatted as 'YY1-MM1-DD1_YY2-MM2-DD2'.
            **kwargs:
        """
        start_date, end_date = str2timewindow(tstring)
        self.forecast_from_file(start_date, end_date, **kwargs)

    def forecast_from_file(self, start_date: datetime, end_date: datetime, **kwargs) -> None:
        """
        Generates a forecast from a file, by parsing and scaling it to.

        the desired time window. H

        Args:
            start_date (~datetime.datetime): Start of the forecast
            end_date (~datetime.datetime): End of the forecast
            **kwargs: Keyword arguments for
             :class:`csep.core.forecasts.GriddedForecast`
        """

        time_horizon = decimal_year(end_date) - decimal_year(start_date)
        tstring = timewindow2str([start_date, end_date])

        f_path = self.path("forecasts", tstring)
        f_parser = getattr(ForecastParsers, self.path.fmt)

        rates, region, mags = f_parser(f_path)

        forecast = GriddedForecast(
            name=f"{self.name}",
            data=rates,
            region=region,
            magnitudes=mags,
            start_time=start_date,
            end_time=end_date,
        )

        scale = time_horizon / self.forecast_unit
        if scale != 1.0:
            forecast = forecast.scale(scale)

        log.debug(
            f"Model {self.name}:\n"
            f"\tForecast expected count: {forecast.event_count:.2f}"
            f" with scaling parameter: {time_horizon:.1f}"
        )

        self.forecasts[tstring] = forecast


class TimeDependentModel(Model):
    """
    Model that creates varying forecasts depending on a time window. Requires either a
    collection of Forecasts or a function that returns a Forecast.

    """

    def __init__(
        self,
        name: str,
        model_path: str,
        func: Union[str, Callable] = None,
        func_kwargs: dict = None,
        **kwargs,
    ) -> None:

        super().__init__(name, model_path, **kwargs)

        self.func = func
        self.func_kwargs = func_kwargs or {}

        self.path = ModelTree(kwargs.get("workdir", os.getcwd()), model_path)
        self.build = kwargs.get("build", None)
        self.run_prefix = ""

        if self.func:
            self.environment = EnvironmentFactory.get_env(
                self.build, self.name, self.path.abs(self.model_path)
            )

    def stage(self, timewindows=None) -> None:
        """
        Pre-steps to make the model runnable before integrating.

            - Get from filesystem, Zenodo or Git
            - Pre-check model fileformat
            - Initialize database
            - Run model quality assurance (unit tests, runnable from floatcsep)
        """
        self.get_source(self.zenodo_id, self.giturl, branch=self.repo_hash)

        if hasattr(self, "environment"):
            self.environment.create_environment()

        self.path.build_tree(
            timewindows=timewindows,
            model_class="td",
            prefix=self.__dict__.get("prefix", self.name),
            args_file=self.__dict__.get("args_file", None),
            input_cat=self.__dict__.get("input_cat", None),
        )

    def get_forecast(
        self, tstring: Union[str, list] = None, region=None
    ) -> Union[GriddedForecast, CatalogForecast, List[GriddedForecast], List[CatalogForecast]]:
        """Wrapper that just returns a forecast, hiding the access method  under the hood"""

        if isinstance(tstring, str):
            # If one time window string is passed
            fc_path = self.path("forecasts", tstring)
            # A region must be given to the forecast
            return csep.load_catalog_forecast(
                fc_path, region=region, apply_filters=True, filter_spatial=True
            )

        else:
            forecasts = []
            for t in tstring:
                fc_path = self.path("forecasts", t)
                # A region must be given to the forecast
                forecasts.append(
                    csep.load_catalog_forecast(
                        fc_path, region=region, apply_filters=True, filter_spatial=True
                    )
                )
            return forecasts

    def create_forecast(self, tstring: str, **kwargs) -> None:
        """
        Creates a forecast from the model source and a given time window.

        Note:
            The argument `tstring` is formatted according to how the Experiment
            handles timewindows, specified in the functions
            :func:'floatcsep.utils.timewindow2str` and
            :func:'floatcsep.utils.str2timewindow`

        Args:
            tstring: String representing the start and end of the forecast,
                formatted as 'YY1-MM1-DD1_YY2-MM2-DD2'.
            **kwargs:
        """
        start_date, end_date = str2timewindow(tstring)

        # Model src is a func or binary

        fc_path = self.path("forecasts", tstring)
        if kwargs.get("force") or not os.path.exists(fc_path):
            self.forecast_from_func(start_date, end_date, **self.func_kwargs, **kwargs)
        else:
            log.info(f"Forecast of {tstring} of model {self.name} already " f"exists")

    def forecast_from_func(self, start_date: datetime, end_date: datetime, **kwargs) -> None:

        self.prepare_args(start_date, end_date, **kwargs)
        log.info(
            f"Running {self.name} using {self.environment.__class__.__name__}:"
            f" {timewindow2str([start_date, end_date])}"
        )

        self.run_model()

    def prepare_args(self, start, end, **kwargs):

        filepath = self.path("args_file")
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

    def run_model(self):

        self.environment.run_command(f'{self.func} {self.path("args_file")}')


class ModelFactory:
    @staticmethod
    def create_model(model_cfg) -> Model:

        model_path = [*model_cfg.values()][0]["model_path"]
        workdir = [*model_cfg.values()][0].get("workdir", "")
        model_class = [*model_cfg.values()][0].get("class", "")

        if model_class == "ti":
            return TimeIndependentModel.from_dict(model_cfg)

        elif model_class == "td":
            return TimeDependentModel.from_dict(model_cfg)

        if os.path.isfile(os.path.join(workdir, model_path)):
            return TimeIndependentModel.from_dict(model_cfg)

        elif "func" in [*model_cfg.values()][0]:
            return TimeDependentModel.from_dict(model_cfg)

        else:
            return TimeIndependentModel.from_dict(model_cfg)
