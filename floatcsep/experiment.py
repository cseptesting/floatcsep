import datetime
import filecmp
import hashlib
import logging
import os
import shutil
from os.path import join, abspath, relpath, dirname, isfile, split, exists
from typing import Union, List, Dict, Sequence

import numpy
import yaml
import scipy


from floatcsep.evaluation import Evaluation
from floatcsep.infrastructure.logger import add_fhandler
from floatcsep.model import Model, TimeDependentModel
from floatcsep.infrastructure.registries import ExperimentRegistry
from floatcsep.infrastructure.repositories import ResultsRepository, CatalogRepository
from floatcsep.utils.helpers import (
    NoAliasLoader,
    read_time_cfg,
    read_region_cfg,
    timewindow2str,
    parse_nested_dicts,
)
from floatcsep.infrastructure.engine import Task, TaskGraph

log = logging.getLogger("floatLogger")


class Experiment:
    """
    Main class that handles an Experiment's context. Contains all the specifications,
    instructions and methods to carry out an experiment.

    Args:
        name (str): Experiment name
        time_config (dict): Contains all the temporal specifications.
            It can contain the following keys:

            - start_date (:class:`datetime.datetime`):
              Experiment start date
            - end_date (:class:`datetime.datetime`):
              Experiment end date
            - exp_class (:class:`str`):
              `ti` (Time-Independent) or `td` (Time-Dependent)
            - intervals (:class:`int`): Number of testing intervals/windows
            - horizon (:class:`str`, :py:class:`float`): Time length of the
              forecasts (e.g. 1, 10, `1 year`, `2 days`). `ti` defaults to
              years, `td` to days.
            - growth (:class:`str`): `incremental` or `cumulative`
            - offset (:class:`float`): recurrence of forecast creation.

            For further details, see :func:`~floatcsep.utils.timewindows_ti`
            and :func:`~floatcsep.utils.timewindows_td`

        region_config (dict): Contains all the spatial and magnitude
            specifications. It must contain the following keys:

            - region (:py:class:`str`,
              :class:`csep.core.regions.CartesianGrid2D`): The geographical
              region, which can be specified as:
              (i) the name of a :mod:`csep`/:mod:`floatcsep` default region
              function (e.g. :func:`~csep.core.regions.california_relm_region`)
              (ii) the name of a user function or
              (iii) the path to a lat/lon file
            - mag_min (:class:`float`): Minimum magnitude of the experiment
            - mag_max (:class:`float`): Maximum magnitude of the experiment
            - mag_bin (:class:`float`): Magnitud bin size
            - magnitudes (:class:`list`, :class:`numpy.ndarray`): Explicit
              magnitude bins
            - depth_min (:class:`float`): Minimum depth. Defaults to -2
            - depth_max (:class:`float`): Maximum depth. Defaults to 6000

        model_config (str): Path to the models' configuration file
        test_config (str): Path to the evaluations' configuration file
        default_test_kwargs (dict): Default values for the testing
         (seed, number of simulations, etc.)
        postprocess (dict): Contains the instruction for postprocessing
         (e.g. plot forecasts, catalogs)
        **kwargs: see Note

    Note:
        Instead of using `time_config` and `region_config`, an Experiment can
        be instantiated using these dicts as keywords. (e.g. ``Experiment(
        **time_config, **region_config)``, ``Experiment(start_date=start_date,
        intervals=1, region='csep-italy', ...)``
    """

    def __init__(
        self,
        name: str = None,
        time_config: dict = None,
        region_config: dict = None,
        catalog: str = None,
        models: str = None,
        tests: str = None,
        postprocess: str = None,
        default_test_kwargs: dict = None,
        rundir: str = "results",
        report_hook: dict = None,
        **kwargs,
    ) -> None:
        # todo
        #  - instantiate from full experiment register (ignoring test/models
        #  config), or self-awareness functions?
        #  - instantiate as python objects (models/tests config)
        #  - check if model region matches experiment region for nonQuadTree?
        #    Or filter region?
        # Instantiate

        workdir = abspath(kwargs.get("path", os.getcwd()))
        if kwargs.get("timestamp", False):
            rundir = os.path.join(
                rundir, f"run_{datetime.datetime.utcnow().date().isoformat()}"
            )
        os.makedirs(os.path.join(workdir, rundir), exist_ok=True)

        self.name = name if name else "floatingExp"
        self.registry = ExperimentRegistry(workdir, rundir)
        self.results_repo = ResultsRepository(self.registry)
        self.catalog_repo = CatalogRepository(self.registry)

        self.config_file = kwargs.get("config_file", None)
        self.original_config = kwargs.get("original_config", None)
        self.original_run_dir = kwargs.get("original_rundir", None)
        self.run_dir = rundir
        self.seed = kwargs.get("seed", None)
        self.time_config = read_time_cfg(time_config, **kwargs)
        self.region_config = read_region_cfg(region_config, **kwargs)
        self.model_config = models if isinstance(models, str) else None
        self.test_config = tests if isinstance(tests, str) else None

        logger = kwargs.get("logging", False)
        if logger:
            filename = "experiment.log" if logger is True else logger
            self.registry.logger = os.path.join(workdir, rundir, filename)
            log.info(f"Logging at {self.registry.logger}")
            add_fhandler(self.registry.logger)

        log.debug("-------- BEGIN OF RUN --------")
        log.info(f"Setting up experiment {self.name}:")
        log.info(f"\tStart: {self.start_date}")
        log.info(f"\tEnd: {self.end_date}")
        log.info(f"\tTime windows: {len(self.timewindows)}")
        log.info(f"\tRegion: {self.region.name if self.region else None}")
        log.info(
            f"\tMagnitude range: [{numpy.min(self.magnitudes)},"
            f" {numpy.max(self.magnitudes)}]"
        )

        self.catalog = None
        self.models = []
        self.tests = []

        self.postprocess = postprocess if postprocess else {}
        self.default_test_kwargs = default_test_kwargs

        self.catalog_repo.set_main_catalog(catalog, self.time_config, self.region_config)

        self.models = self.set_models(
            models or kwargs.get("model_config"), kwargs.get("order", None)
        )
        self.tests = self.set_tests(tests or kwargs.get("test_config"))

        self.tasks = []
        self.task_graph = None

        self.report_hook = report_hook if report_hook else {}
        self.force_rerun = kwargs.get("force_rerun", False)

    def __getattr__(self, item: str) -> object:
        """
        Override built-in method to return the experiment attributes by also using the command
        ``experiment.{attr}``. Adds also to the experiment scope the keys of
        :attr:`region_config` or :attr:`time_config`. These are: ``start_date``, ``end_date``,
        ``timewindows``, ``horizon``, ``offset``, ``region``, ``magnitudes``, ``mag_min``,
        `mag_max``, ``mag_bin``, ``depth_min`` depth_max .
        """

        try:
            return self.__dict__[item]
        except KeyError:
            try:
                return self.time_config[item]
            except KeyError:
                try:
                    return self.region_config[item]
                except KeyError:
                    raise AttributeError(
                        f"Experiment '{self.name}'" f" has no attribute '{item}'"
                    ) from None

    def __dir__(self):
        """Adds the time and region configs keys the to instance scope."""

        _dir = (
            list(super().__dir__()) + list(self.time_config.keys()) + list(self.region_config)
        )
        return sorted(_dir)

    def set_models(self, model_config: Union[Dict, str, List], order: List = None) -> List:
        """
        Parse the models' configuration file/dict. Instantiates all the models as
        :class:`floatcsep.model.Model` and store them into :attr:`Experiment.models`.

        Args:
            model_config (dict, list, str): configuration file or dictionary
             containing the model initialization attributes, as defined in
             :meth:`~floatcsep.model.Model`
            order (list): desired order of models
        """

        models = []
        if isinstance(model_config, str):
            modelcfg_path = self.registry.abs(model_config)
            _dir = self.registry.abs_dir(model_config)
            with open(modelcfg_path, "r") as file_:
                config_dict = yaml.load(file_, NoAliasLoader)
        elif isinstance(model_config, (dict, list)):
            config_dict = model_config
            _dir = self.registry.workdir
        elif model_config is None:
            return models
        else:
            raise NotImplementedError(
                f"Load for model type" f" {model_config.__class__}" f"not implemented "
            )
        for element in config_dict:
            # Check if the model is unique or has multiple submodels

            if not any("flavours" in i for i in element.values()):
                name_ = next(iter(element))
                path_ = self.registry.rel(_dir, element[name_]["path"])
                model_i = {
                    name_: {
                        **element[name_],
                        "model_path": path_,
                        "workdir": self.registry.workdir,
                    }
                }
                model_i[name_].pop("path")
                models.append(Model.factory(model_i))

            else:
                model_flavours = list(element.values())[0]["flavours"].items()
                for flav, flav_path in model_flavours:
                    name_super = next(iter(element))
                    path_super = element[name_super].get("path", "")
                    path_sub = self.registry.rel(_dir, path_super, flav_path)
                    # updates name of submodel
                    name_flav = f"{name_super}@{flav}"
                    model_ = {
                        name_flav: {
                            **element[name_super],
                            "model_path": path_sub,
                            "workdir": self.registry.workdir,
                        }
                    }
                    model_[name_flav].pop("path")
                    model_[name_flav].pop("flavours")
                    models.append(Model.factory(model_))

        # Checks if there is any repeated model.
        names_ = [i.name for i in models]
        if len(names_) != len(set(names_)):
            reps = set([i for i in names_ if (sum([j == i for j in names_]) > 1)])
            one = not bool(len(reps) - 1)
            log.warning(
                f'Warning: Model{"s" * (not one)} {reps}'
                f' {"is" * one + "are" * (not one)} repeated'
            )
        log.info(f"\tModels: {[i.name for i in models]}")
        if order:
            models = [models[i] for i in order]

        return models

    def get_model(self, name: str) -> Model:
        """Returns a Model by its name string."""
        for model in self.models:
            if model.name == name:
                return model

    def get_test(self, name: str) -> Model:
        """Returns an Evaluation by its name string."""
        for test in self.tests:
            if test.name == name:
                return test

    def stage_models(self) -> None:
        """
        Stages all the experiment's models. See :meth:`floatcsep.model.Model.stage`
        """
        log.info("Staging models")
        for i in self.models:
            i.stage(self.timewindows)
            self.registry.add_forecast_registry(i)

    def set_tests(self, test_config: Union[str, Dict, List]) -> list:
        """
        Parse the tests' configuration file/dict. Instantiate them as
        :class:`floatcsep.evaluation.Evaluation` and store them into
        :attr:`Experiment.tests`.

        Args:
            test_config (dict, list, str): configuration file or dictionary
             containing the evaluation initialization attributes, as defined in
             :meth:`~floatcsep.evaluation.Evaluation`
        """
        tests = []

        if isinstance(test_config, str):

            with open(self.registry.abs(test_config), "r") as config:
                config_dict = yaml.load(config, NoAliasLoader)

            for eval_dict in config_dict:
                eval_i = Evaluation.from_dict(eval_dict)
                eval_i.results_repo = self.results_repo
                eval_i.catalog_repo = self.catalog_repo
                tests.append(eval_i)

        elif isinstance(test_config, (dict, list)):

            for eval_dict in test_config:
                eval_i = Evaluation.from_dict(eval_dict)
                eval_i.results_repo = self.results_repo
                eval_i.catalog_repo = self.catalog_repo
                tests.append(eval_i)

        log.info(f"\tEvaluations: {[i.name for i in tests]}")

        return tests

    def set_test_cat(self, tstring: str) -> None:
        """
        Filters the complete experiment catalog to a test sub-catalog bounded by the test
        time-window. Writes it to filepath defined in :attr:`Experiment.registry`

        Args:
            tstring (str): Time window string
        """

        self.catalog_repo.set_test_cat(tstring)

    def set_input_cat(self, tstring: str, model: Model) -> None:
        """
        Filters the complete experiment catalog to an input sub-catalog filtered to the
        beginning of the test time-window.

        Args:
            tstring (str): Time window string
            model (:class:`~floatcsep.model.Model`): Model to give the input
             catalog
        """

        self.catalog_repo.set_input_cat(tstring, model)

    def set_tasks(self) -> None:
        """
        Lazy definition of the experiment core tasks by wrapping instances,
        methods and arguments. Creates a graph with task nodes, while assigning
        task-parents to each node, depending on each Evaluation signature.
        The tasks can then be run sequentially as a list or asynchronous
        using the graph's node dependencies.
        For instance:

        * A forecast can only be made if catalog was filtered to its window
        * A consistency test can be run if the forecast exists in a window
        * A comparison test requires the forecast and ref forecast
        * A sequential test requires the forecasts exist for all windows
        * A batch test requires all forecast exist for a given window.

        """

        # Set the file path structure
        self.registry.build_tree(self.timewindows, self.models, self.tests)

        log.debug("Pre-run forecast summary")
        self.registry.log_forecast_trees(self.timewindows)
        log.debug("Pre-run result summary")
        self.registry.log_results_tree()

        log.info("Setting up experiment's tasks")

        # Get the time windows strings
        tw_strings = timewindow2str(self.timewindows)

        # Prepare the testing catalogs
        task_graph = TaskGraph()
        for time_i in tw_strings:
            # The method call Experiment.set_test_cat(time_i) is created lazily
            task_i = Task(instance=self, method="set_test_cat", tstring=time_i)
            # An is added to the task graph
            task_graph.add(task_i)
            # the task will be executed later with Experiment.run()
            # once all the tasks are defined

        # Set up the Forecasts creation
        for time_i in tw_strings:
            for model_j in self.models:
                if isinstance(model_j, TimeDependentModel):
                    task_tj = Task(
                        instance=self, method="set_input_cat", tstring=time_i, model=model_j
                    )

                    task_graph.add(task=task_tj)
                    # A catalog needs to have been filtered

                task_ij = Task(
                    instance=model_j,
                    method="create_forecast",
                    tstring=time_i,
                    force=self.force_rerun,
                )
                task_graph.add(task=task_ij)
                # A catalog needs to have been filtered
                if isinstance(model_j, TimeDependentModel):
                    task_graph.add_dependency(
                        task_ij, dep_inst=self, dep_meth="set_input_cat", dkw=(time_i, model_j)
                    )
                task_graph.add_dependency(
                    task_ij, dep_inst=self, dep_meth="set_test_cat", dkw=time_i
                )

        # Set up the Consistency Tests
        for test_k in self.tests:
            if test_k.type == "consistency":
                for time_i in tw_strings:
                    for model_j in self.models:
                        task_ijk = Task(
                            instance=test_k,
                            method="compute",
                            timewindow=time_i,
                            model=model_j,
                            region=self.region,
                        )
                        task_graph.add(task_ijk)
                        # the forecast needs to have been created
                        task_graph.add_dependency(
                            task_ijk, dep_inst=model_j, dep_meth="create_forecast", dkw=time_i
                        )
            # Set up the Comparative Tests
            elif test_k.type == "comparative":
                for time_i in tw_strings:
                    for model_j in self.models:
                        task_ik = Task(
                            instance=test_k,
                            method="compute",
                            timewindow=time_i,
                            model=model_j,
                            ref_model=self.get_model(test_k.ref_model),
                            region=self.region,
                        )
                        task_graph.add(task_ik)
                        task_graph.add_dependency(
                            task_ik, dep_inst=model_j, dep_meth="create_forecast", dkw=time_i
                        )
                        task_graph.add_dependency(
                            task_ik,
                            dep_inst=self.get_model(test_k.ref_model),
                            dep_meth="create_forecast",
                            dkw=time_i,
                        )
            # Set up the Sequential Scores
            elif test_k.type == "sequential":
                for model_j in self.models:
                    task_k = Task(
                        instance=test_k,
                        method="compute",
                        timewindow=tw_strings,
                        model=model_j,
                        region=self.region,
                    )
                    task_graph.add(task_k)
                    for tw_i in tw_strings:
                        task_graph.add_dependency(
                            task_k, dep_inst=model_j, dep_meth="create_forecast", dkw=tw_i
                        )
            # Set up the Sequential_Comparative Scores
            elif test_k.type == "sequential_comparative":
                tw_strs = timewindow2str(self.timewindows)
                for model_j in self.models:
                    task_k = Task(
                        instance=test_k,
                        method="compute",
                        timewindow=tw_strs,
                        model=model_j,
                        ref_model=self.get_model(test_k.ref_model),
                        region=self.region,
                    )
                    task_graph.add(task_k)
                    for tw_i in tw_strings:
                        task_graph.add_dependency(
                            task_k, dep_inst=model_j, dep_meth="create_forecast", dkw=tw_i
                        )
                        task_graph.add_dependency(
                            task_k,
                            dep_inst=self.get_model(test_k.ref_model),
                            dep_meth="create_forecast",
                            dkw=tw_i,
                        )
            # Set up the Batch comparative Scores
            elif test_k.type == "batch":
                time_str = timewindow2str(self.timewindows[-1])
                for model_j in self.models:
                    task_k = Task(
                        instance=test_k,
                        method="compute",
                        timewindow=time_str,
                        ref_model=self.models,
                        model=model_j,
                        region=self.region,
                    )
                    task_graph.add(task_k)
                    for m_j in self.models:
                        task_graph.add_dependency(
                            task_k, dep_inst=m_j, dep_meth="create_forecast", dkw=time_str
                        )

        self.task_graph = task_graph

    def run(self) -> None:
        """
        Run the task tree.

        todo:
         - Cleanup forecast (perhaps add a clean task in self.prepare_tasks,
            after all test had been run for a given forecast)
         - Memory monitor?
         - Queuer?
        """
        log.info(f"Running {self.task_graph.ntasks} tasks")

        if self.seed:
            numpy.random.seed(self.seed)

        self.task_graph.run()
        log.info("Calculation completed")
        log.debug("Post-run forecast registry")
        self.registry.log_forecast_trees(self.timewindows)
        log.debug("Post-run result summary")
        self.registry.log_results_tree()

    def read_results(self, test: Evaluation, window: str) -> List:
        """
        Reads an Evaluation result for a given time window and returns a list of the results
        for all tested models.
        """

        return test.read_results(window, self.models)

    def make_repr(self) -> None:
        """
        Creates a reproducibility configuration file, re-directing the forecasts/catalog paths,
        in order to reproduce the existing results and compare them with previous runs.

        """
        log.info("Creating reproducibility config file")
        repr_config = self.registry.get("repr_config")

        # Dropping region to results folder if it is a file
        region_path = self.region_config.get("path", False)
        if isinstance(region_path, str):
            if isfile(region_path) and region_path:
                new_path = join(self.registry.run_dir, self.region_config["path"])
                shutil.copy2(region_path, new_path)
                self.region_config.pop("path")
                self.region_config["region"] = self.registry.rel(new_path)

        # Dropping catalog to results folder
        target_cat = join(
            self.registry.workdir, self.registry.run_dir, split(self.catalog_repo.cat_path)[-1]
        )
        if not exists(target_cat):
            shutil.copy2(self.registry.abs(self.catalog_repo.cat_path), target_cat)

        relative_path = os.path.relpath(
            self.registry.workdir, os.path.join(self.registry.workdir, self.registry.run_dir)
        )
        self.registry.workdir = relative_path
        self.to_yml(repr_config, extended=True)

    def as_dict(self, extra: Sequence = (), extended=False) -> dict:
        """
        Converts an Experiment instance into a dictionary.

        Args:
            extra: additional instance attribute to include in the dictionary.
            extended: Include explicit parameters

        Returns:
            A dictionary with serialized instance's attributes, which are
            floatCSEP readable
        """

        dict_walk = {
            "name": self.name,
            "config_file": self.config_file,
            "path": self.registry.workdir,
            "run_dir": self.registry.run_dir,
            "time_config": {
                i: j
                for i, j in self.time_config.items()
                if (i not in ("timewindows",) or extended)
            },
            "region_config": {
                i: j
                for i, j in self.region_config.items()
                if (i not in ("magnitudes", "depths") or extended)
            },
            "catalog": self.catalog_repo.cat_path,
            "models": [i.as_dict() for i in self.models],
            "tests": [i.as_dict() for i in self.tests],
        }
        dict_walk.update(extra)

        return parse_nested_dicts(dict_walk)

    def to_yml(self, filename: str, **kwargs) -> None:
        """
        Serializes the :class:`~floatcsep.experiment.Experiment` instance into a .yml file.

        Note:
            This instance can then be reinstantiated using
            :meth:`~floatcsep.experiment.Experiment.from_yml`

        Args:
            filename: Name of the file onto which dump the instance
            **kwargs: Pass to :meth:`~floatcsep.experiment.Experiment.as_dict`

        Returns:
        """

        class NoAliasDumper(yaml.Dumper):
            def ignore_aliases(self, data):
                return True

        with open(filename, "w") as f_:
            yaml.dump(
                self.as_dict(**kwargs),
                f_,
                Dumper=NoAliasDumper,
                sort_keys=False,
                default_flow_style=False,
                indent=1,
                width=70,
            )

    @classmethod
    def from_yml(cls, config_yml: str, repr_dir=None, **kwargs):
        """
        Initializes an experiment from a .yml file. It must contain the.

        attributes described in the :class:`~floatcsep.experiment.Experiment`,
        :func:`~floatcsep.utils.read_time_config` and
        :func:`~floatcsep.utils.read_region_config` descriptions

        Args:
            config_yml (str): The path to the .yml file
            repr_dir (str): folder where to reproduce results

        Returns:
            An :class:`~floatcsep.experiment.Experiment` class instance
        """
        log.info("Initializing experiment from .yml file")
        with open(config_yml, "r") as yml:

            # experiment configuration file
            _dict = yaml.load(yml, NoAliasLoader)
            _dir_yml = dirname(config_yml)

            # Only ABSOLUTE PATH
            _dict["path"] = abspath(join(_dir_yml, _dict.get("path", "")))

            # replaces rundir case reproduce option is used
            if repr_dir:
                _dict["original_rundir"] = _dict.get("rundir", "results")
                _dict["rundir"] = relpath(join(_dir_yml, repr_dir), _dict["path"])
                _dict["original_config"] = abspath(join(_dict["path"], _dict["config_file"]))
            else:

                _dict["rundir"] = _dict.get("rundir", kwargs.pop("rundir", "results"))
            _dict["config_file"] = relpath(config_yml, _dir_yml)
            if "logging" in _dict:
                kwargs.pop("logging")

        return cls(**_dict, **kwargs)


class ExperimentComparison:

    def __init__(self, original, reproduced, **kwargs):
        """"""
        self.original = original
        self.reproduced = reproduced

        self.num_results = {}
        self.file_comp = {}

    @staticmethod
    def obs_diff(obs_orig, obs_repr):

        return numpy.abs(
            numpy.divide((numpy.array(obs_orig) - numpy.array(obs_repr)), numpy.array(obs_orig))
        )

    @staticmethod
    def test_stat(test_orig, test_repr):

        if isinstance(test_orig[0], str):
            if not isinstance(test_orig[1], str):
                stats = numpy.array(
                    [0, numpy.divide((test_repr[1] - test_orig[1]), test_orig[1]), 0, 0]
                )
            else:
                stats = None
        else:
            stats_orig = numpy.array(
                [numpy.mean(test_orig), numpy.std(test_orig), scipy.stats.skew(test_orig)]
            )
            stats_repr = numpy.array(
                [numpy.mean(test_repr), numpy.std(test_repr), scipy.stats.skew(test_repr)]
            )

            ks = scipy.stats.ks_2samp(test_orig, test_repr)
            stats = [*numpy.divide(numpy.abs(stats_repr - stats_orig), stats_orig), ks.pvalue]
        return stats

    def get_results(self):

        win_orig = timewindow2str(self.original.timewindows)

        tests_orig = self.original.tests

        models_orig = [i.name for i in self.original.models]

        results = dict.fromkeys([i.name for i in tests_orig])

        for test in tests_orig:
            if test.type in ["consistency", "comparative"]:
                results[test.name] = dict.fromkeys(win_orig)
                for tw in win_orig:
                    results_orig = self.original.read_results(test, tw)
                    results_repr = self.reproduced.read_results(test, tw)
                    results[test.name][tw] = {
                        models_orig[i]: {
                            "observed_statistic": self.obs_diff(
                                results_orig[i].observed_statistic,
                                results_repr[i].observed_statistic,
                            ),
                            "test_statistic": self.test_stat(
                                results_orig[i].test_distribution,
                                results_repr[i].test_distribution,
                            ),
                        }
                        for i in range(len(models_orig))
                    }

            else:
                results_orig = self.original.read_results(test, win_orig[-1])
                results_repr = self.reproduced.read_results(test, win_orig[-1])
                results[test.name] = {
                    models_orig[i]: {
                        "observed_statistic": self.obs_diff(
                            results_orig[i].observed_statistic,
                            results_repr[i].observed_statistic,
                        ),
                        "test_statistic": self.test_stat(
                            results_orig[i].test_distribution, results_repr[i].test_distribution
                        ),
                    }
                    for i in range(len(models_orig))
                }

        return results

    @staticmethod
    def get_hash(filename):

        with open(filename, "rb") as f:
            bytes_file = f.read()
            readable_hash = hashlib.sha256(bytes_file).hexdigest()
        return readable_hash

    def get_filecomp(self):

        win_orig = timewindow2str(self.original.timewindows)

        tests_orig = self.original.tests

        models_orig = [i.name for i in self.original.models]

        results = dict.fromkeys([i.name for i in tests_orig])

        for test in tests_orig:
            if test.type in ["consistency", "comparative"]:
                results[test.name] = dict.fromkeys(win_orig)
                for tw in win_orig:
                    results[test.name][tw] = dict.fromkeys(models_orig)
                    for model in models_orig:
                        orig_path = self.original.registry.get_result(tw, test, model)
                        repr_path = self.reproduced.registry.get_result(tw, test, model)

                        results[test.name][tw][model] = {
                            "hash": (self.get_hash(orig_path) == self.get_hash(repr_path)),
                            "byte2byte": filecmp.cmp(orig_path, repr_path),
                        }
            else:
                results[test.name] = dict.fromkeys(models_orig)
                for model in models_orig:
                    orig_path = self.original.registry.get_result(win_orig[-1], test, model)
                    repr_path = self.reproduced.registry.get_result(win_orig[-1], test, model)
                    results[test.name][model] = {
                        "hash": (self.get_hash(orig_path) == self.get_hash(repr_path)),
                        "byte2byte": filecmp.cmp(orig_path, repr_path),
                    }
        return results

    def compare_results(self):

        self.num_results = self.get_results()
        self.file_comp = self.get_filecomp()
