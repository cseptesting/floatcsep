import os
import os.path

import csep
import numpy
import yaml
import json

from typing import Union, List, Dict, Callable, Mapping, Sequence
from matplotlib import pyplot
from cartopy import crs as ccrs

from csep.models import EvaluationResult
from csep.core.catalogs import CSEPCatalog
from csep.utils.time_utils import decimal_year

from floatcsep import report
from floatcsep.registry import PathTree
from floatcsep.utils import NoAliasLoader, parse_csep_func, read_time_config, \
    read_region_config, Task, TaskGraph, timewindow2str, str2timewindow, \
    magnitude_vs_time
from floatcsep.model import Model
from floatcsep.evaluation import Evaluation
import warnings

numpy.seterr(all="ignore")
warnings.filterwarnings("ignore")


class Experiment:
    """

    Main class that handles an Experiment's context. Contains all the
    specifications, instructions and methods to carry out an experiment.

    Args:
        name (str): Experiment name
        path (str): Experiment working directory. All artifacts relative paths'
                    are defined from here (e.g. model files, source code,
                    catalog files, etc.)
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

            For further details, see :func:`~floatcsep.utils.timewindows_ti` and
            :func:`~floatcsep.utils.timewindows_td`

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
        postproc_config (dict): Contains the instruction for postprocessing
         (e.g. plot forecasts, catalogs)
        **kwargs: see Note

    Note:
        Instead of using `time_config` and `region_config`, an Experiment can
        be instantiated using these dicts as keywords. (e.g. ``Experiment(
        **time_config, **region_config)``, ``Experiment(start_date=start_date,
        intervals=1, region='csep-italy', ...)``

    """

    '''
    Data management
    
    Model:
        - FILE   - read from file, scale in runtime             
                 - drop to db, scale from function in runtime   
        - CODE  - run, read from file              
                - run, store in db, read from db   
    
    TEST:
        - use forecast from runtime
        - read forecast from file (TD)
          (does not make sense for TI (too much FS space) unless is already
           dropped to DB)
    '''

    def __init__(self,
                 name: str = None,
                 time_config: dict = None,
                 region_config: dict = None,
                 catalog: str = None,
                 models: str = None,
                 tests: str = None,
                 postproc_config: str = None,
                 default_test_kwargs: dict = None,
                 **kwargs) -> None:
        # todo
        #  - instantiate from full experiment register (ignoring test/models
        #  config), or self-awareness functions?
        #  - instantiate as python objects (models/tests config)
        #  - check if model region matches experiment region for nonQuadTree?
        #    Or filter region?
        # Instantiate
        self.name = name if name else 'floatingExp'
        self.path = kwargs.get('path') if kwargs.get('path',
                                                     None) else os.getcwd()

        self.time_config = read_time_config(time_config, **kwargs)
        self.region_config = read_region_config(region_config, **kwargs)
        self.model_config = models if isinstance(models, str) else None
        self.test_config = tests if isinstance(tests, str) else None

        self.catalog = None
        self.models = []
        self.tests = []

        self.postproc_config = postproc_config if postproc_config else {}
        self.default_test_kwargs = default_test_kwargs

        self.filetree = PathTree(self.path)

        self.catalog = catalog
        self.models = self.set_models(models or kwargs.get('model_config'))
        self.tests = self.set_tests(tests or kwargs.get('test_config'))

        self.tasks = []
        self.task_graph = None

        self.force_rerun = kwargs.get('force_rerun', False)

    def __getattr__(self, item: str) -> object:
        """
        Override built-in method to return attributes found within
        :attr:`region_config` or :attr:`time_config`
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
                        f"Experiment '{self.name}'"
                        f" has no attribute '{item}'") from None

    def __dir__(self):
        """
        Adds time and region configs keys to instance scope.
        """

        _dir = list(super().__dir__()) + list(self.time_config.keys()) + list(
            self.region_config)
        return sorted(_dir)

    def set_models(self, model_config: Union[Dict, str, List]) -> List:
        """

        Parse the models' configuration file/dict. Instantiates all the models
        as :class:`floatcsep.model.Model` and store them into
        :attr:`Experiment.models`.

        Args:
            model_config (dict, list, str): configuration file or dictionary
             containing the model initialization attributes, as defined in
             :meth:`~floatcsep.model.Model`


        """

        models = []
        if isinstance(model_config, str):
            modelcfg_path = self.filetree.abs(model_config)
            _dir = self.filetree.absdir(model_config)
            with open(modelcfg_path, 'r') as file_:
                config_dict = yaml.load(file_, NoAliasLoader)
        elif isinstance(model_config, (dict, list)):
            config_dict = model_config
            _path = [i['path'] for i in model_config[0].values()][0]
            _dir = self.filetree.absdir(_path)
        elif model_config is None:
            return models
        else:
            raise NotImplementedError(f'Load for model type'
                                      f' {model_config.__class__}'
                                      f'not implemented ')

        for element in config_dict:
            # Check if the model is unique or has multiple submodels
            if not any('flavours' in i for i in element.values()):

                name_ = next(iter(element))
                # updates path to absolute
                model_path = self.filetree.abs(_dir, element[name_]['path'])
                model_i = {name_: {**element[name_], 'path': model_path}}
                models.append(Model.from_dict(model_i))
            else:
                model_flavours = list(element.values())[0]['flavours'].items()
                for flav, flav_path in model_flavours:
                    name_super = next(iter(element))
                    # updates path to absolute
                    path_super = element[name_super].get('path', '')
                    path_sub = self.filetree.abs(_dir, path_super, flav_path)
                    # path_sub = self._abspath(_dir, path_super, flav_path)[1]
                    # updates name of submodel
                    name_flav = f'{name_super}@{flav}'
                    model_ = {name_flav: {**element[name_super],
                                          'path': path_sub}}
                    model_[name_flav].pop('flavours')
                    models.append(Model.from_dict(model_))

        # Checks if there is any repeated model.
        names_ = [i.name for i in models]
        if len(names_) != len(set(names_)):
            reps = set(
                [i for i in names_ if (sum([j == i for j in names_]) > 1)])
            one = not bool(len(reps) - 1)
            print(f'Warning: Model{"s" * (not one)} {reps}'
                  f' {"is" * one + "are" * (not one)} repeated')

        return models

    def get_model(self, name: str) -> Model:
        """ Returns a Model by its name string """
        for model in self.models:
            if model.name == name:
                return model

    def stage_models(self) -> None:
        """
        Stages all the experiment's models. See
        :meth:`floatcsep.model.Model.stage`
        """
        for i in self.models:
            i.stage(self.timewindows)

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
            with open(self.filetree.abs(test_config), 'r') as config:
                config_dict = yaml.load(config, NoAliasLoader)
            for eval_dict in config_dict:
                tests.append(Evaluation.from_dict(eval_dict))
        elif isinstance(test_config, (dict, list)):
            for eval_dict in test_config:
                tests.append(Evaluation.from_dict(eval_dict))

        return tests

    @property
    def catalog(self) -> CSEPCatalog:
        """
        Returns a CSEP catalog loaded from the given query function or
         a stored file if it exists.
        """
        if callable(self._catalog):
            if os.path.isfile(self._catpath):
                return CSEPCatalog.load_json(self._catpath)
            bounds = {'start_time': min([item for sublist in self.timewindows
                                         for item in sublist]),
                      'end_time': max([item for sublist in self.timewindows
                                       for item in sublist]),
                      'min_magnitude': self.magnitudes.min(),
                      'max_depth': self.depths.max()}
            if self.region:
                bounds.update({i: j for i, j in
                               zip(['min_longitude', 'max_longitude',
                                    'min_latitude', 'max_latitude'],
                                   self.region.get_bbox())})

            catalog = self._catalog(
                catalog_id='cat',  # todo name as run
                **bounds
            )

            if self.region:
                catalog.filter_spatial(region=self.region)
                catalog.region = None
            catalog.write_json(self._catpath)

            return catalog

        elif os.path.isfile(self._catalog):
            try:
                return CSEPCatalog.load_json(self._catpath)
            except json.JSONDecodeError:
                return csep.load_catalog(self._catpath)

    @catalog.setter
    def catalog(self, cat: Union[Callable, CSEPCatalog, str]) -> None:

        if cat is None:
            self._catalog = None
            self._catpath = None

        elif os.path.isfile(self.filetree.abs(cat)):
            print(f"Using catalog from {cat}")
            self._catalog = self.filetree.abs(cat)
            self._catpath = self.filetree.abs(cat)

        else:
            # catalog can be a function
            self._catalog = parse_csep_func(cat)
            self._catpath = self.filetree.abs('catalog.json')
            if os.path.isfile(self._catpath):
                print(f"Using stored catalog "
                      f"'{os.path.relpath(self._catpath, self.path)}', "
                      f"obtained from function '{cat}'")
            else:
                print(f"Downloading catalog from function {cat}")

    def set_test_cat(self, tstring: str) -> None:
        """

        Filters the complete experiment catalog to a test sub-catalog bounded
        by the test time-window. Writes it to filepath defined in
        :attr:`Experiment.filetree`

        Args:
            tstring (str): Time window string

        """
        start, end = str2timewindow(tstring)
        sub_cat = self.catalog.filter(
            [f'origin_time < {end.timestamp() * 1000}',
             f'origin_time >= {start.timestamp() * 1000}',
             f'magnitude >= {self.mag_min}',
             f'magnitude < {self.mag_max}'], in_place=False)
        if self.region:
            sub_cat.filter_spatial(region=self.region)
        sub_cat.write_json(filename=self.filetree(tstring, 'catalog'))

    def set_input_cat(self, tstring: str, model: Model) -> None:
        """

        Filters the complete experiment catalog to a input sub-catalog filtered
        to the beginning of thetest time-window. Writes it to filepath defined
        in :attr:`Model.tree.catalog`

        Args:
            tstring (str): Time window string
            model (:class:`~floatcsep.model.Model`): Model to give the input
             catalog

        """
        start, end = str2timewindow(tstring)
        sub_cat = self.catalog.filter(
            [f'origin_time < {start.timestamp() * 1000}'])
        sub_cat.write_ascii(filename=model.tree('cat'))

    def set_tasks(self, results_path=None, run_name=None):
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

        Args:
            results_path (str, None): Path where the results will be stored.
             Defaults to `results`.
            run_name (str, None): Sub-folder where this particular run will be
             stored. Default is ''

        Returns:

        """

        # Set the file path structure
        self.filetree.set_pathtree(self.timewindows,
                                   self.models,
                                   self.tests,
                                   results_path=results_path,
                                   run_name=run_name)

        # Get the time windows strings
        tw_strings = timewindow2str(self.timewindows)

        # Prepare the testing catalogs
        task_graph = TaskGraph()
        for time_i in tw_strings:
            # The method call Experiment.set_test_cat(time_i) is created lazily
            task_i = Task(instance=self,
                          method='set_test_cat',
                          tstring=time_i)
            # An is added to the task graph
            task_graph.add(task_i)
            # the task will be executed later with Experiment.run()
            # once all the tasks are defined

        # Set up the Forecasts creation
        for time_i in tw_strings:
            for model_j in self.models:
                if model_j.model_class == 'td':
                    task_tj = Task(instance=self,
                                   method='set_input_cat',
                                   tstring=time_i,
                                   model=model_j)

                    task_graph.add(task=task_tj)
                    # A catalog needs to have been filtered

                task_ij = Task(instance=model_j,
                               method='create_forecast',
                               tstring=time_i,
                               force=self.force_rerun)
                task_graph.add(task=task_ij)
                # A catalog needs to have been filtered
                if model_j.model_class == 'td':
                    task_graph.add_dependency(task_ij,
                                              dinst=self,
                                              dmeth='set_input_cat',
                                              dkw=(time_i, model_j))
                task_graph.add_dependency(task_ij,
                                          dinst=self,
                                          dmeth='set_test_cat',
                                          dkw=time_i)

        # Set up the Consistency Tests
        for test_k in self.tests:
            if test_k.type == 'consistency':
                for time_i in tw_strings:
                    for model_j in self.models:
                        task_ijk = Task(
                            instance=test_k,
                            method='compute',
                            timewindow=time_i,
                            catalog=self.filetree(time_i, 'catalog'),
                            model=model_j,
                            region=self.region,
                            path=self.filetree(time_i, 'evaluations',
                                               test_k, model_j))
                        task_graph.add(task_ijk)
                        # the forecast needs to have been created
                        task_graph.add_dependency(task_ijk,
                                                  dinst=model_j,
                                                  dmeth='create_forecast',
                                                  dkw=time_i)
            # Set up the Comparative Tests
            elif test_k.type == 'comparative':
                for time_i in tw_strings:
                    for model_j in self.models:
                        task_ik = Task(
                            instance=test_k,
                            method='compute',
                            timewindow=time_i,
                            catalog=self.filetree(time_i, 'catalog'),
                            model=model_j,
                            ref_model=self.get_model(test_k.ref_model),
                            region=self.region,
                            path=self.filetree(time_i, 'evaluations', test_k,
                                               model_j)
                        )
                        task_graph.add(task_ik)
                        task_graph.add_dependency(task_ik,
                                                  dinst=model_j,
                                                  dmeth='create_forecast',
                                                  dkw=time_i)
                        task_graph.add_dependency(task_ik,
                                                  dinst=self.get_model(
                                                      test_k.ref_model),
                                                  dmeth='create_forecast',
                                                  dkw=time_i)
            # Set up the Sequential Scores
            elif test_k.type == 'sequential':
                for model_j in self.models:
                    task_k = Task(
                        instance=test_k,
                        method='compute',
                        timewindow=tw_strings,
                        catalog=[self.filetree(i, 'catalog')
                                 for i in tw_strings],
                        model=model_j,
                        region=self.region,
                        path=self.filetree(tw_strings[-1], 'evaluations',
                                           test_k, model_j)
                    )
                    task_graph.add(task_k)
                    for tw_i in tw_strings:
                        task_graph.add_dependency(task_k,
                                                  dinst=model_j,
                                                  dmeth='create_forecast',
                                                  dkw=tw_i)
            # Set up the Sequential_Comparative Scores
            elif test_k.type == 'sequential_comparative':
                tw_strs = timewindow2str(self.timewindows)
                for model_j in self.models:
                    task_k = Task(
                        instance=test_k,
                        method='compute',
                        timewindow=tw_strs,
                        catalog=[self.filetree(i, 'catalog') for i in tw_strs],
                        model=model_j,
                        ref_model=self.get_model(test_k.ref_model),
                        region=self.region,
                        path=self.filetree(tw_strs[-1], 'evaluations', test_k,
                                           model_j)
                    )
                    task_graph.add(task_k)
                    for tw_i in tw_strings:
                        task_graph.add_dependency(task_k,
                                                  dinst=model_j,
                                                  dmeth='create_forecast',
                                                  dkw=tw_i)
                        task_graph.add_dependency(task_k,
                                                  dinst=self.get_model(
                                                      test_k.ref_model),
                                                  dmeth='create_forecast',
                                                  dkw=tw_i)
            # Set up the Batch comparative Scores
            elif test_k.type == 'batch':
                time_str = timewindow2str(self.timewindows[-1])
                for model_j in self.models:
                    task_k = Task(
                        instance=test_k,
                        method='compute',
                        timewindow=time_str,
                        catalog=self.filetree(time_str, 'catalog'),
                        ref_model=self.models,
                        model=model_j,
                        region=self.region,
                        path=self.filetree(time_str, 'evaluations', test_k,
                                           model_j)
                    )
                    task_graph.add(task_k)
                    for m_j in self.models:
                        task_graph.add_dependency(task_k,
                                                  dinst=m_j,
                                                  dmeth='create_forecast',
                                                  dkw=time_str)

        self.task_graph = task_graph

    def run(self) -> None:
        """
        Run the task tree

        todo:
         - Cleanup forecast (perhaps add a clean task in self.prepare_tasks,
            after all test had been run for a given forecast)
         - Memory monitor?
         - Queuer?

        """
        self.task_graph.run()
        self.to_yml(self.filetree('config'), extended=True)

    def read_results(self, test: Evaluation, window: str) -> List:
        """
        Reads an Evaluation result for a given time window and returns a list
        of the results for all tested models.
        """
        test_results = []
        if not isinstance(window, str):
            wstr_ = timewindow2str(window)
        else:
            wstr_ = window

        if 'T' in test.name:  # todo cleaner
            models = [i for i in self.models if i.name != test.ref_model]
        else:
            models = self.models
        for i in models:
            eval_path = self.filetree(wstr_, 'evaluations', test, i.name)
            with open(eval_path, 'r') as file_:
                model_eval = EvaluationResult.from_dict(json.load(file_))
            test_results.append(model_eval)
        return test_results

    def plot_results(self, dpi: int = 300, show: bool = False) -> None:
        """
        
        Plots all evaluation results
 
        Args:
            dpi: Figure resolution with which to save
            show: show in runtime

        """

        for time in self.timewindows:
            time_str = timewindow2str(time)

            # consistency and comparative
            for test in self.tests:
                if test.type in ['consistency', 'comparative']:
                    fig_path = self.filetree(time_str, 'figures', test.name)
                    results = self.read_results(test, time)
                    ax = test.plot_func(results, plot_args=test.plot_args,
                                        **test.plot_kwargs)
                    if 'code' in test.plot_args:
                        exec(test.plot_args['code'])
                        fig_path = self.filetree(time_str, 'figures',
                                                 test.name)
                    pyplot.savefig(fig_path, dpi=dpi)
                    if show:
                        pyplot.show()

        for test in self.tests:
            # todo improve the logic of this plots
            time_str = timewindow2str(self.timewindows[-1])

            results = self.read_results(test, time_str)
            if test.type == 'seqcomp':
                results_ = []
                for i in results:
                    if i.sim_name != test.ref_model:
                        results_.append(i)
                results = results_
            if 'Sequential' in test.type:
                test.plot_args['timestrs'] = timewindow2str(self.timewindows)
            import matplotlib
            ax = test.plot_func(results, plot_args=test.plot_args,
                                **test.plot_kwargs)
            if 'code' in test.plot_args:
                exec(test.plot_args['code'])
            fig_path = self.filetree(time_str, 'figures', test.name)
            pyplot.savefig(fig_path, dpi=dpi)
            if show:
                pyplot.show()

    def plot_catalog(self, dpi: int = 300, show: bool = False) -> None:
        """

        Plots the evaluation catalogs

        Args:
            dpi: Figure resolution with which to save
            show: show in runtime

        """
        plot_args = {'basemap': 'ESRI_terrain',
                     'figsize': (12, 8),
                     'markersize': 8,
                     'markercolor': 'black',
                     'grid_fontsize': 16,
                     'title': '',
                     'legend': True,
                     }
        plot_args.update(self.postproc_config.get('plot_catalog', {}))

        if self.postproc_config.get('all_time_windows'):
            timewindow = self.timewindows
        else:
            timewindow = [self.timewindows[-1]]

        for tw in timewindow:
            catpath = self.filetree(tw, 'catalog')
            catalog = CSEPCatalog.load_json(catpath)

            ax = catalog.plot(plot_args=plot_args, show=show)
            ax.get_figure().tight_layout()
            ax.get_figure().savefig(self.filetree(tw, 'figures', 'catalog'),
                                    dpi=dpi)

            ax2 = magnitude_vs_time(catalog)
            ax2.get_figure().tight_layout()
            ax2.get_figure().savefig(self.filetree(tw, 'figures',
                                                   'magnitude_time'), dpi=dpi)

    def plot_forecasts(self) -> None:
        """

        Plots and saves all the generated forecasts

        """

        plot_fc_config = self.postproc_config.get('plot_forecasts')
        if plot_fc_config:
            if plot_fc_config is True:
                plot_fc_config = {}
            try:
                proj_ = plot_fc_config.get('projection')
                if isinstance(proj_, dict):
                    proj_name = list(proj_.keys())[0]
                    proj_args = list(proj_.values())[0]
                else:
                    proj_name = proj_
                    proj_args = {}
                plot_fc_config['projection'] = getattr(ccrs, proj_name)(
                    **proj_args)
            except (IndexError, KeyError, TypeError):
                plot_fc_config['projection'] = ccrs.PlateCarree(
                    central_longitude=0.0)

            cat = plot_fc_config.get('catalog')
            cat_args = {}
            if cat:
                cat_args = {'markersize': 7, 'markercolor': 'black',
                            'title': None,
                            'legend': False, 'basemap': None,
                            'region_border': False}
                if self.region:
                    self.catalog.filter_spatial(self.region, in_place=True)
                if isinstance(cat, dict):
                    cat_args.update(cat)

            window = self.timewindows[-1]
            winstr = timewindow2str(window)

            for model in self.models:
                fig_path = self.filetree(winstr, 'forecasts', model.name)
                start = decimal_year(window[0])
                end = decimal_year(window[1])
                time = f'{round(end - start, 3)} years'
                plot_args = {'region_border': False,
                             'cmap': 'magma',
                             'clabel': r'$\log_{10} N\left(M_w \in [{%.2f},'
                                       r'\,{%.2f}]\right)$ per '
                                       r'$0.1^\circ\times 0.1^\circ $ per %s' %
                                       (min(self.magnitudes),
                                        max(self.magnitudes), time)}
                if not self.region or self.region.name == 'global':
                    set_global = True
                else:
                    set_global = False
                plot_args.update(plot_fc_config)
                ax = model.get_forecast(winstr, self.region).plot(
                    set_global=set_global, plot_args=plot_args)

                if self.region:
                    bbox = self.region.get_bbox()
                    dh = self.region.dh
                    extent = [bbox[0] - 3 * dh, bbox[1] + 3 * dh,
                              bbox[2] - 3 * dh, bbox[3] + 3 * dh]
                else:
                    extent = None
                if cat:
                    self.catalog.plot(ax=ax, set_global=set_global,
                                      extent=extent, plot_args=cat_args)

                pyplot.savefig(fig_path, dpi=300, facecolor=(0, 0, 0, 0))

    def generate_report(self) -> None:
        """

        Creates a report summarizing the Experiment's results

        """

        report.generate_report(self)

    def check_reproducibility(self) -> None:

        pass

    def to_dict(self, exclude: Sequence = ('magnitudes', 'depths',
                                           'timewindows', 'filetree',
                                           'task_graph', 'tasks',
                                           'models', 'tests'),
                extended: bool = False) -> dict:
        """
        Converts an Experiment instance into a dictionary.

        Args:
            exclude (tuple, list): Attributes, or attribute keys, to ignore
            extended (bool): Verbose representation of pycsep objects

        Returns:
            A dictionary with serialized instance's attributes, which are
            floatCSEP readable
        """

        def _get_value(x):
            # For each element type, transforms to desired string/output
            if hasattr(x, 'to_dict') and extended:
                # e.g. csep region, model, test, etc.
                o = x.to_dict()
            else:
                try:
                    try:
                        o = getattr(x, '__name__')
                    except AttributeError:
                        o = getattr(x, 'name')
                except AttributeError:
                    if isinstance(x, numpy.ndarray):
                        o = x.tolist()
                    else:
                        o = x
            return o

        def iter_attr(val):
            # recursive iter through nested dicts/lists
            if isinstance(val, Mapping):
                return {item: iter_attr(val_) for item, val_ in val.items()
                        if ((item not in exclude) and val_) or extended}
            elif isinstance(val, Sequence) and not isinstance(val, str):
                return [iter_attr(i) for i in val]
            else:
                return _get_value(val)

        listwalk = [(i, j) for i, j in self.__dict__.items() if
                    not i.startswith('_')]
        listwalk.insert(3, ('catalog', self._catpath))

        dictwalk = {i: j for i, j in listwalk}
        # if self.model_config is None:
        #     dictwalk['models'] = iter_attr(self.models)
        # if self.test_config is None:
        #     dictwalk['tests'] = iter_attr(self.tests)

        return iter_attr(dictwalk)

    def to_yml(self, filename: str, **kwargs) -> None:
        """

        Serializes the :class:`~floatcsep.experiment.Experiment` instance into a
        .yml file.

        Note:
            This instance can then be reinstantiated using
            :meth:`~floatcsep.experiment.Experiment.from_yml`

        Args:
            filename: Name of the file onto which dump the instance
            **kwargs: Passed to :meth:`~floatcsep.experiment.Experiment.to_dict`

        Returns:

        """

        class NoAliasDumper(yaml.Dumper):
            def ignore_aliases(self, data):
                return True

        with open(filename, 'w') as f_:
            yaml.dump(
                self.to_dict(**kwargs), f_,
                Dumper=NoAliasDumper,
                sort_keys=False,
                default_flow_style=False,
                indent=1,
                width=70
            )

    @classmethod
    def from_yml(cls, config_yml: str):
        """

        Initializes an experiment from a .yml file. It must contain the
        attributes described in the :class:`~floatcsep.experiment.Experiment`,
        :func:`~floatcsep.utils.read_time_config` and
        :func:`~floatcsep.utils.read_region_config` descriptions

        Args:
            config_yml (str): The path to the .yml file

        Returns:
            An :class:`~floatcsep.experiment.Experiment` class instance

        """
        with open(config_yml, 'r') as yml:
            config_dict = yaml.safe_load(yml)
            if 'path' not in config_dict:
                config_dict['path'] = os.path.abspath(
                    os.path.dirname(config_yml))
        return cls(**config_dict)
