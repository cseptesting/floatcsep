import json
import logging
import os
import shutil
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import List, Callable, Union, Sequence

import yaml
from csep.core.forecasts import GriddedForecast, CatalogForecast

from floatcsep.infrastructure.environments import EnvironmentFactory
from floatcsep.infrastructure.registries import ModelRegistry
from floatcsep.infrastructure.repositories import ForecastRepository
from floatcsep.utils.accessors import from_zenodo, from_git
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
    def stage(self, time_windows=None) -> None:
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

    @abstractmethod
    def get_source(self):
        """Retrieves the model from a web repository"""
        pass

    def as_dict(self, excluded=("name", "repository", "workdir", "environment")):
        """
        Returns:
            Dictionary with relevant attributes. Model can be re-instantiated from this dict
        """

        list_walk = [
            (i, j) for i, j in sorted(self.__dict__.items()) if not i.startswith("_") and j
        ]

        dict_walk = {i: j for i, j in list_walk if i not in excluded}
        dict_walk["path"] = self.registry.rel(dict_walk.pop("registry").path).as_posix()

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
        self.registry = ModelRegistry.factory(
            model_name=name, workdir=kwargs.get("workdir", os.getcwd()), path=model_path
        )
        self.repository = ForecastRepository.factory(
            self.registry, model_class=self.__class__.__name__, **kwargs
        )

    def stage(self, time_windows: Sequence[Sequence[datetime]] = None, **kwargs) -> None:
        """
        Acquire the forecast data if it is not in the file system. Sets the paths internally
        (or database pointers) to the forecast data.

        Args:
            time_windows (list): time_windows that the forecast data represents.
        """

        if self.force_stage or not self.registry.file_exists("path"):
            os.makedirs(self.registry.dir, exist_ok=True)
            self.get_source()  # now the TI version above

        self.registry.build_tree(time_windows=time_windows, model_class=self.__class__.__name__)

    def get_source(self) -> None:
        """
        Fetch a single-file forecast into the model directory
        """

        container = self.registry.dir
        expected_file = self.registry.path

        os.makedirs(container, exist_ok=True)

        if expected_file.exists() and expected_file.is_file() and not self.force_stage:
            return

        os.makedirs(container, exist_ok=True)

        if expected_file.exists() and expected_file.is_file() and not self.force_stage:
            return

        if self.giturl:
            from_git(self.giturl, str(container), branch=self.repo_hash, force=self.force_stage)
        elif self.zenodo_id:
            from_zenodo(
                self.zenodo_id,
                str(container),
                force=self.force_stage,
                keys=[expected_file.name],
            )
        else:
            pass

        if not expected_file.exists() or not expected_file.is_file():
            raise FileNotFoundError(
                f"Expected TI model file at: {expected_file}\n" f"Fetched into: {container}"
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
        args_file: str = "args.txt",
        input_cat: str = "catalog.csv",
        fmt: str = "csv",
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
            args_file: Name of the arguments file that will be used to create forecasts
            input_cat: Name of the file that will be used as input catalog to create forecasts
            **kwargs: Additional keyword parameters, such as a ``prefix`` (str) for the
                resulting forecast file paths, ``args_file`` (str) as the path for the model
                arguments file or ``input_cat`` that indicates where the input catalog will be
                placed for the model.

        """
        super().__init__(name, **kwargs)

        self.func = func
        self.func_kwargs = func_kwargs or {}
        self.registry = ModelRegistry.factory(
            model_name=name,
            workdir=kwargs.get("workdir", os.getcwd()),
            path=model_path,
            fmt=fmt,
            args_file=args_file,
            input_cat=input_cat,
        )
        self.repository = ForecastRepository.factory(
            self.registry, model_class=self.__class__.__name__, **kwargs
        )
        self.build = kwargs.get("build", None)
        self.force_build = kwargs.get("force_build", False)
        if self.func:
            self.environment = EnvironmentFactory.get_env(
                self.build, self.name, self.registry.path.as_posix()
            )

    def stage(
        self, time_windows=None, run_mode="sequential", stage_dir="results", run_id="run"
    ) -> None:
        """
        Retrieve model artifacts and Set up its interface with the experiment.

        1) Get the model from filesystem, Zenodo or Git. Prepares the directory
        2) If source code, creates the computational environment (conda, venv or Docker)
        3) Prepares the registry tree: filepaths/keys corresponding to existing forecasts
           and those to be generated, as well as input catalog and arguments file.

        """
        need_source = (
                self.force_stage
                or not self.registry.path.exists()
                or (self.registry.path.is_dir() and not any(self.registry.path.iterdir()))
        )

        if need_source:
            os.makedirs(self.registry.dir, exist_ok=True)
            self.get_source(self.zenodo_id, self.giturl, branch=self.repo_hash)

        if hasattr(self, "environment"):
            self.environment.create_environment(force=self.force_build)

        self.registry.build_tree(
            time_windows=time_windows,
            model_class=self.__class__.__name__,
            prefix=self.__dict__.get("prefix", self.name),
            run_mode=run_mode,
            stage_dir=stage_dir,
            run_id=run_id,
        )

    def get_source(self, zenodo_id: int = None, giturl: str = None, **kwargs) -> None:
        """
        Search, download or clone the model source in the filesystem from git or zenodo, respectively.

        Args:
            zenodo_id (int): Zenodo identifier of the repository. Usually as
             `https://zenodo.org/record/{zenodo_id}`
            giturl (str): git remote repository URL from which to clone the
             source
            **kwargs: see :func:`~floatcsep.utils.from_zenodo` and
             :func:`~floatcsep.utils.from_git`
        """

        target_dir = self.registry.path  # TD expects a directory here

        # If forced, start clean so clone/download won’t fail on non-empty
        if self.force_stage and target_dir.exists():

            shutil.rmtree(target_dir)

        target_dir.mkdir(parents=True, exist_ok=True)

        if self.giturl:
            from_git(self.giturl, target_dir.as_posix(), branch=self.repo_hash, force=False)
        elif self.zenodo_id:
            from_zenodo(self.zenodo_id, target_dir.as_posix(), force=self.force_stage)
        else:
            pass

        if not target_dir.exists() or not target_dir.is_dir():
            raise FileNotFoundError(f"Expected TD model directory at: {target_dir}")

    def get_forecast(
        self, tstring: Union[str, list] = None, region=None
    ) -> Union[GriddedForecast, CatalogForecast, List[GriddedForecast], List[CatalogForecast]]:
        """
        Wrapper that returns a forecast, by accessing the model's forecast repository.

        Note:
            The argument ``tstring`` is formatted according to how the Experiment
            handles time_windows, specified in the functions
            :func:`~floatcsep.utils.helpers.timewindow2str` and
            :func:`~floatcsep.utils.helpers.str2timewindow`

        Args:
            tstring: String representing the start and end of the forecast,
                formatted as 'YY1-MM1-DD1_YY2-MM2-DD2'.
            region: String representing the region for which to return a forecast.
                If None, will return a forecast for all regions.

        """
        return self.repository.load_forecast(tstring, name=self.name, region=region)

    def create_forecast(self, tstring: str, **kwargs) -> None:
        """
        Creates a forecast from the model source and a given time window.

        Note:
            The argument ``tstring`` is formatted according to how the Experiment
            handles time_windows, specified in the functions
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
        self.prepare_extra_input(start_date, end_date, **kwargs)

        log.info(
            f"[Model] Running {self.name} using {self.environment.__class__.__name__}:"
            f" {timewindow2str([start_date, end_date])}"
        )
        input_dir = self.registry.get_input_dir(tstring)
        forecast_dir = self.registry.get_forecast_dir()
        run_label = f"{self.name}_{tstring}"
        self.environment.run_command(
            command=f"{self.func}",
            run_label=run_label,
            input_volume=input_dir,
            forecast_volume=forecast_dir,
        )

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
        window_str = timewindow2str([start, end])

        dest_path = Path(self.registry.get_args_key(window_str))
        tpl_path = self.registry.get_args_template_path()
        suffix = tpl_path.suffix.lower()

        if suffix == ".txt":

            def load_kv(fp: Path) -> dict:
                data = {}
                if fp.exists():
                    with open(fp, "r") as f:
                        for line in f:
                            line = line.strip()
                            if not line or line.startswith("#"):
                                continue
                            if "=" in line:
                                k, v = line.split("=", 1)
                                data[k.strip()] = v.strip()
                return data

            def dump_kv(fp: Path, data: dict) -> None:
                ordered_keys = []
                for k in ("start_date", "end_date"):
                    if k in data:
                        ordered_keys.append(k)
                ordered_keys += sorted(
                    k for k in data.keys() if k not in ("start_date", "end_date")
                )

                with open(fp, "w") as f:
                    for k in ordered_keys:
                        f.write(f"{k} = {data[k]}\n")

            data = load_kv(tpl_path)
            data["start_date"] = start.isoformat()
            data["end_date"] = end.isoformat()
            for k, v in (kwargs or {}).items():
                data[k] = v
            for k, v in (self.func_kwargs or {}).items():
                data[k] = v
            dump_kv(dest_path, data)

        elif suffix == ".json":
            base = {}
            if tpl_path.exists():
                with open(tpl_path, "r") as f:
                    base = json.load(f) or {}
            base["start_date"] = start.isoformat()
            base["end_date"] = end.isoformat()
            base.update(kwargs or {})
            base.update(self.func_kwargs or {})

            with open(dest_path, "w") as f:
                json.dump(base, f, indent=2)

        elif suffix in (".yml", ".yaml"):
            if tpl_path.exists():
                with open(tpl_path, "r") as f:
                    data = yaml.safe_load(f) or {}
            else:
                data = {}

            data["start_date"] = start.isoformat()
            data["end_date"] = end.isoformat()

            def nested_update(dest: dict, src: dict, max_depth: int = 3, _lvl: int = 1):
                for key, val in (src or {}).items():
                    if (
                        _lvl < max_depth
                        and key in dest
                        and isinstance(dest[key], dict)
                        and isinstance(val, dict)
                    ):
                        nested_update(dest[key], val, max_depth, _lvl + 1)
                    else:
                        dest[key] = val

            nested_update(data, self.func_kwargs or {})
            nested_update(data, kwargs or {})
            with open(dest_path, "w") as f:
                yaml.safe_dump(data, f, indent=2)

        else:
            raise ValueError(f"Unsupported args file format: {suffix}")

    def prepare_extra_input(self, start: datetime, end: datetime, **kwargs) -> None:
        """
        When the model is a source code, and the run is in parallel, additional data located in
         the model's input folder is copied onto the mounted folder (in /tmp or  /results)

        Args:
            start: start date of the forecast timewindow
            end: end date of the forecast timewindow
            **kwargs: represents additional model arguments (name/value pair)

        """
        window_str = timewindow2str([start, end])

        src_path = self.registry.model_dirtree["input"]
        dest_path = self.registry.get_input_dir(window_str)

        catalog_filename = Path(self.registry.get_input_catalog_key(window_str)).name
        args_filename = Path(self.registry.get_args_key(window_str)).name
        exclude = {catalog_filename, args_filename}

        if src_path.resolve() == dest_path.resolve():
            return

        dest_path.mkdir(parents=True, exist_ok=True)

        for root, dirs, files in os.walk(src_path):
            root_p = Path(root)
            rel = root_p.relative_to(src_path)
            out_dir = dest_path / rel
            out_dir.mkdir(parents=True, exist_ok=True)

            for fname in files:
                if fname in exclude:
                    continue
                src_file = root_p / fname
                dst_file = out_dir / fname

                shutil.copy2(src_file, dst_file)
