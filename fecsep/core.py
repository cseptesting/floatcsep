import copy
import json

import csep.core.forecasts
import git
import numpy
import cartopy.crs as ccrs
from collections.abc import Mapping, Sequence
import h5py
import yaml
from matplotlib import pyplot
import os
import os.path

from csep.models import EvaluationResult
from csep.core.catalogs import CSEPCatalog
from csep.core.forecasts import GriddedForecast
from csep.utils.time_utils import decimal_year
from csep.core.regions import QuadtreeGrid2D, CartesianGrid2D
from csep.models import Polygon

import fecsep.utils
import fecsep.accessors
import fecsep.evaluations
from fecsep.utils import NoAliasLoader, _set_dockerfile, \
    parse_csep_func, read_time_config, read_region_config, Task, timewindow_str
from fecsep.accessors import from_zenodo, from_git
from fecsep.dbparser import load_from_hdf5
import docker
import docker.errors

_client = docker.from_env()


class Model:
    def __init__(self, name, path,
                 forecast_unit=1,
                 authors=None, doi=None,
                 func=None, func_args=None,
                 zenodo_id=None, giturl=None, repo_hash=None):
        """

        Args:
            name (str):
            path (str):
            format (str):
            db_type (str):
            forecast_unit (str):
            authors (str):
            doi (str):
            markdown:
            func:
            func_args:
            zenodo_id:
            giturl:
            repo_hash:
        """

        '''
        Model typologies:
        
            Updating
            - Time-Independent
            - Time-Dependent
            Origin
            - File
            - Code
            Source 
            - Local
            - Zenodo
            - Git
    
        To implement in beta version
            (ti - file - local)
            (ti - file - zenodo)
            (ti - file - git)
            
            (td - code - local)
            (td - code - git)
        '''

        # todo list
        #  - Check format
        #  - Instantiate from source code
        #  - Check contents when instantiating from_git

        self.name = name
        self.path = path
        self._dir = None
        self.dbpath = None

        self.zenodo_id = zenodo_id
        self.giturl = giturl
        self.repo_hash = repo_hash

        self.format = None
        self.get_source(zenodo_id, giturl)

        if self.format != 'src':
            self.dbserializer = parse_csep_func(self.format)
            self.make_db()

        self.authors = authors
        self.doi = doi

        self.forecast_unit = forecast_unit

        self.forecasts = {}
        self.func = func
        self.func_args = func_args

    def get_source(self, zenodo_id=None, giturl=None, **kwargs):
        """

        Search(and download/clone) the model source in the filesystem, zenodo
        and git. Identifies if the instance path points to a file or to its
        parent directory

        Args:
            zenodo_id: Zenodo identifier of the repository. Usually as
             `https://zenodo.org/record/{zenodo_id}`
            giturl: git remote repository URL from which to clone the source
            **kwargs: see :func:`~fecsep.utils.from_zenodo` and
             :func:`~fecsep.utils.from_git`

        Returns:

        """

        # Check if the provided path is a file or dir.
        ext = os.path.splitext(self.path)[-1]

        if bool(ext):
            self._dir = os.path.dirname(self.path)
            self.format = ext.split('.')[-1]
        else:
            self._dir = self.path
            self.format = 'src'

        if not os.path.exists(self.path):
            # It does not exist, get from zenodo or git
            if zenodo_id is None and giturl is None:
                raise FileNotFoundError(
                    f"Model file or directory '{self.path}' not found")
            # Model needs to be downloaded from zenodo/git
            os.makedirs(self._dir, exist_ok=True)
            try:
                # Zenodo is the first source of retrieval
                from_zenodo(zenodo_id, self._dir, **kwargs)
            except KeyError or TypeError as zerror_:
                try:
                    from_git(giturl, self._dir, **kwargs)
                except (git.NoSuchPathError, git.CommandError) as giterror_:
                    if giturl is None:
                        raise KeyError('Zenodo identifier is not valid')
                    else:
                        raise git.NoSuchPathError('git url was not found')

            # Check if file or directory exists after downloading
            if bool(ext):
                path_exists = os.path.isfile(self.path)
            else:
                path_exists = os.path.isdir(self.path)

            assert path_exists

    @staticmethod
    def build_docker(model_name, folder):
        dockerfile = os.path.join(folder, 'Dockerfile')
        if os.path.isfile(dockerfile):
            print('Dockerfile exists')
        else:
            with open(dockerfile, 'w') as file_:
                file_.write(_set_dockerfile(model_name))
        client = docker.from_env()
        img = client.images.build(path=folder,
                                  quiet=False,
                                  tag=model_name,
                                  forcerm=True,
                                  buildargs={'USERNAME': os.environ['USER'],
                                             'USER_UID': str(os.getuid()),
                                             'USER_GID': str(os.getgid())},
                                  dockerfile='Dockerfile')
        for stream in img[1]:
            print(stream.get('stream', '').split('\n')[0])

        return img

    def make_db(self):
        """

        Returns:

        """

        self.dbpath = os.path.splitext(self.path)[0] + '.hdf5'

        if not os.path.isfile(self.dbpath):
            self.dbserializer(self.path, self.dbpath)

    def rm_db(self):

        if os.path.isfile(self.dbpath):
            os.remove(self.dbpath)
            return True
        else:
            print("The HDF5 file does not exist")
            return False

    def forecast_path(self, win):
        return 'a'

    def create_forecast(self, start_date, end_date, **kwargs):
        """
        Creates a forecast from a model and a time window
        :param start_date: A model configuration dict
        :param test_date: A test date to calculate the horizon
        :return: A pycsep.core.forecasts.GriddedForecast object
        """

        if self.path == self._dir:
            # Forecasts are created from source code
            self.make_forecast_td(start_date, end_date, **kwargs)
        else:
            # Forecasts are created from file
            self.make_forecast_ti(start_date, end_date, **kwargs)

    def make_forecast_td(self, start_date, end_date, **kwargs):
        pass

    def make_forecast_ti(self, start_date, end_date, **kwargs):

        time_horizon = decimal_year(end_date) - decimal_year(start_date)
        tstring = timewindow_str([start_date, end_date])

        # todo implement these functions in dbparser

        rates, region, magnitudes = load_from_hdf5(self.dbpath)

        forecast = GriddedForecast(
            name=f'{self.name}_{tstring}',
            data=rates,
            region=region,
            magnitudes=magnitudes,
            start_time=start_date,
            end_time=end_date
        )

        forecast = forecast.scale(time_horizon / self.forecast_unit)
        print(
            f"Forecast expected count: {forecast.event_count:.2f}"
            f" with scaling parameter: {time_horizon:.1f}")
        self.forecasts[tstring] = forecast

    def to_dict(self):
        # todo: modify this function to include more state from the class
        out = {}
        included = ['name', 'path']
        for k, v in self.__dict__.items():
            if k in included:
                out[k] = v
        return out

    @classmethod
    def from_dict(cls, record):
        if len(record) != 1:
            raise IndexError('A single model has not been passed')
        name = next(iter(record))
        return cls(name=name, **record[name])


class Test:
    """

    """

    def __init__(self, name, func, markdown='', func_args=None,
                 func_kwargs=None, plot_func=None,
                 plot_args=None, plot_kwargs=None, model=None, ref_model=None,
                 path=None):
        """

        :param name:
        :param func:
        :param func_args:
        :param plot_func:
        :param plot_args:
        :param ref_model:
        :param kwargs:
        """
        self.name = name
        self.func = parse_csep_func(func)
        self.func_kwargs = func_kwargs  # todo set default args from exp?
        self.func_args = func_args
        self.plot_func = parse_csep_func(plot_func)
        self.plot_args = plot_args or {}  # todo default args?
        self.plot_kwargs = plot_kwargs or {}
        self.ref_model = ref_model
        self.model = model
        self.path = path
        self.markdown = markdown

    def compute(self, timewindow,
                catpath,
                model,
                path,
                region=None):

        forecast = model.forecasts[timewindow]
        catalog = CSEPCatalog.load_json(catpath)

        if region:
            catalog.filter_spatial(in_place=True)
        # if self.ref_model is not None:
        #     ref_model = self.get_model(test.ref_model)
        #     test_args = (forecast, self.get_forecast(ref_model), catalog)
        # elif test.func == fecsep.evaluations.vector_poisson_t_w_test:
        #     forecast_batch = [self.get_forecast(model_i) for model_i in
        #                       self.models]
        #     test_args = (forecast, forecast_batch, catalog)
        # else:
        test_args = (forecast, catalog)
        result = self.func(*test_args, **self.func_kwargs)

        with open(path, 'w') as _file:
            json.dump(result.to_dict(), _file, indent=4)
        # return test_args

    def to_dict(self):
        out = {}
        included = ['name', 'model', 'ref_model', 'path', 'func_kwargs']
        for k, v in self.__dict__.items():
            if k in included and v is not None:
                out[k] = v
        return out

    def __str__(self):
        return (
            f"name: {self.name}\n"
            f"reference model: {self.ref_model}\n"
            f"kwargs: {self.func_kwargs}\n"
            f"path: {self.path}"
        )

    @classmethod
    def from_dict(cls, record):
        if len(record) != 1:
            raise IndexError('A single test has not been passed')
        name = next(iter(record))
        return cls(name=name, **record[name])


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

            For further details, see :func:`~fecsep.utils.time_windows_ti` and
            :func:`~fecsep.utils.time_windows_td`

        region_config (dict): Contains all the spatial and magnitude
            specifications. It must contain the following keys:

            - region (:py:class:`str`,
              :class:`csep.core.regions.CartesianGrid2D`): The geographical
              region, specified as:
              (i) the name of a :mod:`csep`/:mod:`fecsep` default region
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

    def __init__(self,
                 name=None,
                 path=None,
                 time_config=None,
                 region_config=None,
                 catalog=None,
                 model_config=None,
                 test_config=None,
                 postproc_config=None,
                 default_test_kwargs=None,
                 **kwargs):

        # todo
        #  - instantiate from full experiment register (ignoring
        #  test/models config), or self-awareness functions?
        #  - instantiate as python objects (rethink models/tests config)

        # Instantiate
        self.name = name
        self.path = path if path else os.getcwd()

        self.time_config = read_time_config(time_config, **kwargs)
        self.region_config = read_region_config(region_config, **kwargs)

        # todo: make it simple and also load catalog as file if given:
        self.catalog = catalog

        self.model_config = model_config
        self.test_config = test_config
        self.postproc_config = postproc_config if postproc_config else {}
        self.default_test_kwargs = default_test_kwargs

        # Initialize class attributes
        self.models = []
        self.tests = []

        self.tasks = []
        self.run_results = {}
        # self.catalog = None
        self.run_folder: str = ''
        self.target_paths: dict = {}
        self.exists: dict = {}

        # Update if attributes were passed explicitly
        # todo check reinstantiation
        # self.__dict__.update(**kwargs)

    def __getattr__(self, item):
        # Gets time_config and region_config keys as Experiment's attribute
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
        # Adds time and region configs keys to instance scope
        _dir = list(super().__dir__()) + list(self.time_config.keys()) + list(
            self.region_config)
        return sorted(_dir)

    def _abspath(self, *paths):
        # Gets the absolute path of a file, when it was defined relative to the
        # experiment working dir.
        # todo check redundancy when passing an actual absolute path (e.g.
        #  when reinstantiating)
        _path = os.path.normpath(
            os.path.abspath(os.path.join(self.path, *paths)))
        _dir = os.path.dirname(_path)
        return _dir, _path

    def set_models(self):
        """

        Parse the models' configuration file/dict. Instantiates all the models
        as :class:`fecsep.core.Model` and store them into :attr:`self.models`.

        """
        # todo: handle when model cfg is a dict instead of a file.

        _dir, modelcfg_path = self._abspath(self.model_config)

        with open(modelcfg_path, 'r') as file_:
            config_dict = yaml.load(file_, NoAliasLoader)

        for element in config_dict:
            # Check if the model is unique or has multiple submodels
            if not any('flavours' in i for i in element.values()):

                name_ = next(iter(element))
                # updates path to absolute
                model_abspath = self._abspath(_dir, element[name_]['path'])[1]
                model_i = {name_: {**element[name_], 'path': model_abspath}}

                self.models.append(Model.from_dict(model_i))
            else:
                model_flavours = list(element.values())[0]['flavours'].items()
                for flav, flav_path in model_flavours:
                    name_super = next(iter(element))
                    # updates path to absolute
                    path_super = element[name_super].get('path', '')
                    path_sub = self._abspath(_dir, path_super, flav_path)[1]
                    # updates name of submodel
                    name_flav = f'{name_super}@{flav}'
                    model_ = {name_flav: {**element[name_super],
                                          'path': path_sub}}
                    model_[name_flav].pop('flavours')
                    self.models.append(Model.from_dict(model_))

        # Checks if there is any repeated model.
        names_ = [i.name for i in self.models]
        if len(names_) != len(set(names_)):
            reps = set(
                [i for i in names_ if (sum([j == i for j in names_]) > 1)])
            one = not bool(len(reps) - 1)
            print(f'Warning: Model{"s" * (not one)} {reps}'
                  f' {"is" * one + "are" * (not one)} repeated')

    def set_tests(self):
        """
        Parse the tests' configuration file/dict. Instantiate them as
        :class:`fecsep.core.Test` and store them into :attr:`self.tests`.

        """

        with open(self.test_config, 'r') as config:
            config_dict = yaml.load(config, NoAliasLoader)
        self.tests = [Test.from_dict(tdict) for tdict in config_dict]

    def prepare_paths(self, results_path=None, run_name=None):
        """

        Creates the run directory, and reads the file structure inside


        Args:
            results_path:
            run_name:

        Returns:
            run_folder: Path to the run
            exists: flag if forecasts, catalogs and test_results if they exist
             already
            target_paths: flag to each element of the gefe
                (catalog and evaluation results)

        """

        # grab names for creating directories
        windows = timewindow_str(self.time_windows)
        models = [i.name for i in self.models]
        tests = [i.name for i in self.tests]

        # todo create datetime parser for filenames
        if run_name is None:
            run_name = 'run'
            # todo find better way to name paths
            # run_name = f'run_{datetime.now().date().isoformat()}'

        # Determine required directory structure for run
        # results > test_date > time_window > cats / evals / figures

        self.run_folder = self._abspath(results_path or 'results', run_name)[1]
        subfolders = ['catalog', 'evaluations', 'figures', 'forecasts']

        dirtree = {
            win: {folder: self._abspath(self.run_folder, win, folder)[1] for
                  folder
                  in subfolders} for win in windows}

        # create directories if they don't exist
        for tw, tw_folder in dirtree.items():
            for _, folder_ in tw_folder.items():
                os.makedirs(folder_, exist_ok=True)

        # Check existing files
        files = {win: {name: list(os.listdir(path)) for name, path in
                       windir.items()} for win, windir in dirtree.items()}

        exists = {win: {
            'forecasts': False,
            # todo Modify for time-dependent, and/or forecast storage
            'catalog': any(file for file in files[win]['catalog']),
            'evaluations': {
                test: {
                    model: any(f'{test}_{model}.json' in file for file in
                               files[win]['evaluations'])
                    for model in models
                }
                for test in tests
            }
        } for win in windows}

        target_paths = {win: {
            'models': {
                'forecasts': {
                    model_name: self.get_model(model_name).forecast_path(win)
                    for model_name in models},
                # todo: important in time-dependent, and/or forecast storage
                'figures': {model: os.path.join(dirtree[win]['figures'],
                                                f'{model}')
                            for model in models}},
            'catalog': os.path.join(dirtree[win]['catalog'], 'catalog.json'),
            'evaluations': {
                test: {
                    model: os.path.join(dirtree[win]['evaluations'],
                                        f'{test}_{model}.json')
                    for model in models
                }
                for test in tests
            },
            'figures': {test: os.path.join(dirtree[win]['figures'], f'{test}')
                        for test in tests}
        } for win in windows}

        self.run_folder = self.run_folder
        self.target_paths = target_paths
        self.exists = exists

    def prepare_tasks(self):

        tasks = []

        # todo: Depth? Magnitude?
        filter_spatial = Task(instance=self.catalog, method='filter_spatial',
                              region=self.region, in_place=True)

        tasks.append(filter_spatial)
        for time_i in self.time_windows:

            time_str = timewindow_str(time_i)
            filter_catalog = Task(
                instance=self.catalog,
                method='filter',
                statements=[f'origin_time >= {time_i[0].timestamp() * 1000}',
                            f'origin_time < {time_i[1].timestamp() * 1000}']
            )

            write_catalog = Task(
                instance=filter_catalog,
                method='write_json',
                filename=self.target_paths[time_str]['catalog']
            )
            tasks_i = [filter_catalog, write_catalog]
            tasks.extend(tasks_i)

            for model_j in self.models:

                task_ij = Task(
                    instance=model_j,
                    method='create_forecast',
                    start_date=time_i[0],
                    end_date=time_i[1]
                )
                tasks.append(task_ij)

                for test_k in self.tests:
                    task_ijk = Task(
                        instance=test_k,
                        method='compute',
                        timewindow=time_str,
                        catpath=self.target_paths[time_str][
                            'catalog'],
                        model=model_j,
                        path=self.target_paths[time_str][
                            'evaluations'][test_k.name][model_j.name]
                    )
                    tasks.append(task_ijk)

        self.tasks = tasks

    def get_model(self, name):
        for model in self.models:
            if model.name == name:
                return model

    def get_forecast(self, model, datetimes):
        # if already bound to model class, simply return
        try:
            forecast = model.forecasts[timewindow_str(datetimes)]
        except KeyError:
            # this call binds to model class
            forecast = model.create_forecast(datetimes[0], datetimes[1],
                                             name=model.name)
        return forecast

    @property
    def catalog(self):

        if callable(self._catalog):
            if os.path.isfile(self._catpath):
                return CSEPCatalog.load_json(self._catpath)

            min_mag = self.magnitudes.min()
            max_depth = self.depths.max()
            if self.region is not None:
                spatial_bounds = {i: j for i, j in
                                  zip(['min_longitude', 'max_longitude',
                                       'min_latitude', 'max_latitude'],
                                      self.region.get_bbox())}
            else:
                spatial_bounds = {}
            time_bounds = [min([item for sublist in self.time_windows
                                for item in sublist]),
                           max([item for sublist in self.time_windows
                                for item in sublist])]

            catalog = self._catalog(
                catalog_id='cat',  # todo name as run
                start_time=time_bounds[0],
                end_time=time_bounds[1],
                min_magnitude=min_mag,
                max_depth=max_depth,
                verbose=True,
                **spatial_bounds
            )

            catalog.write_json(self._catpath)
            return catalog

        elif os.path.isfile(self._catalog):
            return CSEPCatalog.load_json(self._catpath)

    @catalog.setter
    def catalog(self, cat):

        if os.path.isfile(self._abspath(cat)[1]):
            print(f"Using catalog from {cat}")
            self._catalog = cat
            self._catpath = cat

        else:
            # catalog can be a function
            self._catalog = parse_csep_func(cat)
            self._catpath = self._abspath('catalog.json')[1]
            if os.path.isfile(self._catpath):
                print(f"Load stored catalog "
                      f"'{os.path.relpath(self._catpath, self.path)}', "
                      f"obtained from function '{cat}'")
            else:
                print(f"Downloading catalog from function {cat}")

    def run(self):

        for task in self.tasks:
            task.run()

    def read_evaluation_result(self, test, models, target_paths):
        test_results = []
        if 'T' in test.name:  # todo cleaner
            models = [i for i in models if i.name != test.ref_model]

        for model_i in models:
            eval_path = target_paths['evaluations'][test.name][model_i.name]
            with open(eval_path, 'r') as file_:
                model_eval = EvaluationResult.from_dict(json.load(file_))
            test_results.append(model_eval)
        return test_results

    def plot_results(self, run_results, file_paths=None, dpi=300, show=False):
        """ plots test results
        :param run_results: defaultdict(list) where keys are the test name
        :param file_paths: figure path for each test result
        :param dpi: resolution for output image
        """
        if file_paths is None:
            file_paths = self.target_paths
        for test in self.tests:
            test_result = run_results[test.name]
            ax = test.plot_func(test_result, plot_args=test.plot_args,
                                **test.plot_kwargs)
            if 'code' in test.plot_args:
                exec(test.plot_args['code'])
            pyplot.savefig(file_paths['figures'][test.name], dpi=dpi)
            if show:
                pyplot.show()

        # todo create different method for forecast plotting
        plot_fc_config = self.postproc_config.get('plot_forecasts')
        if plot_fc_config:
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
            except:
                plot_fc_config['projection'] = ccrs.PlateCarree(
                    central_longitude=0.0)

            cat = plot_fc_config.get('catalog')
            if cat:
                cat_args = {'markersize': 7, 'markercolor': 'black',
                            'title': None,
                            'legend': False, 'basemap': None,
                            'region_border': False}
                if self.region:
                    self.catalog.filter_spatial(self.region, in_place=True)
                if isinstance(cat, dict):
                    cat_args.update(cat)

            for model in self.models:
                fig_path = self.target_paths['models']['figures'][model.name]
                start = decimal_year(self.start_date)
                end = decimal_year(self.test_date)
                time = f'{round(end - start, 3)} years'
                plot_args = {'region_border': False,
                             'cmap': 'magma',
                             'clabel': r'$\log_{10} N\left(M_w \in [{%.2f},\,{%.2f}]\right)$ per '
                                       r'$0.1^\circ\times 0.1^\circ $ per %s' %
                                       (min(self.magnitude_range),
                                        max(self.magnitude_range), time)}
                if not self.region:
                    set_global = True
                else:
                    set_global = False
                plot_args.update(plot_fc_config)
                ax = model.forecasts[self.test_date].plot(
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

    def to_dict(self, exclude=('magnitudes', 'depths', 'time_windows'),
                extended=False):
        """
        Converts an Experiment instance into a dictionary.

        Args:
            exclude (tuple, list): Attributes, or attribute keys, to ignore
            extended (bool): Verbose representation of pycsep objects
            (e.g. region)

        Returns:
            A dictionary with serialized instance's attributes, which are
            feCSEP readable
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
                    # elif isinstance(x, datetime):
                    #     o = x.isoformat(' ')
                    else:
                        o = x
            return o

        def iter_attr(val):
            # recursive iter through nested dicts/lists
            if isinstance(val, Mapping):
                return {item: iter_attr(val_) for item, val_ in val.items()
                        if (item not in exclude) or extended}
            elif isinstance(val, Sequence) and not isinstance(val, str):
                return [iter_attr(i) for i in val]
            else:
                return _get_value(val)

        return iter_attr(self.__dict__)

    def to_yml(self, filename, **kwargs):
        """

        Serializes the :class:`~fecsep.core.Experiment` instance into a .yml
        file.

        Note:
            This instance can then be reinstantiated using
            :meth:`~fecsep.core.Experiment.from_yml`

        Args:
            filename: Name of the file onto which dump the instance
            **kwargs: Passed to :meth:`~fecsep.core.Experiment.to_dict`

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
    def from_yml(cls, config_yml):
        """

        Initializes an experiment from a .yml file. It must contain the
        attributes described in the :class:`~fecsep.core.Experiment`,
        :func:`~fecsep.utils.read_time_config` and
        :func:`~fecsep.utils.read_region_config` descriptions

        Args:
            config_yml (str): The path to the .yml file

        Returns:
            An :class:`~fecsep.core.Experiment` class instance

        """
        with open(config_yml, 'r') as yml:
            config_dict = yaml.safe_load(yml)
            if 'path' not in config_dict:
                config_dict['path'] = os.path.abspath(
                    os.path.dirname(config_yml))
        return cls(**config_dict)

    #
    # def prepare_catalogs(self, datetimes, savepath):
    #
    #     cat = self.catalog.filter_spatial(self.region)
    #     cat.filter([f'magnitude >= {self.magnitudes.min()}',
    #                 f'magnitude <= {self.magnitudes.max()}',
    #                 f'depth >= {self.depths.min()}',
    #                 f'depth <= {self.depths.min()}',
    #                 f'origin_time >= {datetimes[0].timestamp() * 1000}',
    #                 f'origin_time < {datetimes[1].timestamp() * 1000}'],
    #                in_place=True)
    #     cat.write_json(savepath)
    #
    # def stage_models(self, force=False):
    #     for model in self.models:
    #         model.get_source(force)
    #         model.stage_db(force)
    #
    # @staticmethod
    # def run_test(test, write=True):
    #     # requires that test be fully configured, probably by calling
    #     # enumerate_tests() first
    #     result = test.compute()
    #     if write:
    #         with open(test.path, 'w') as _file:
    #             json.dump(result.to_dict(), _file, indent=4)
    #     return result
    #
    # def prepare_all_tests(self):
    #     """ Prepare test to be run for the gefe by including runtime arguments like forecasts and catalogs
    #
    #     :return tests: Complete list of evaluations to run for gefe
    #     """
    #     # prepares arguments for test
    #     test_list = []
    #     for model in self.models:
    #         for test in self.tests:
    #             # skip t-test if model is the same as ref_model
    #             if test.ref_model == model.name:
    #                 continue
    #             # prepare args so test is callable
    #             t = Test(
    #                 name=test.name,
    #                 func=test.func,
    #                 func_args=self._prepare_test_func_args(test, model),
    #                 func_kwargs=test.func_kwargs,
    #                 plot_func=test.plot_func,
    #                 plot_args=test.plot_args,
    #                 model=model,
    #                 path=self.target_paths['evaluations'][test.name][
    #                     model.name],
    #                 ref_model=test.ref_model
    #             )
    #             test_list.append(t)
    #             print("Prepared...\n", t)
    #     return test_list
    #
    # def _prepare_test_func_args(self, test, model):
    #     forecast = self.get_forecast(model)
    #     catalog = copy.deepcopy(self.get_catalog())
    #     catalog.region = forecast.region
    #     if self.region:
    #         catalog.filter_spatial(in_place=True)
    #     if test.ref_model is not None:
    #         ref_model = self.get_model(test.ref_model)
    #         test_args = (forecast, self.get_forecast(ref_model), catalog)
    #     elif test.func == fecsep.evaluations.vector_poisson_t_w_test:
    #         forecast_batch = [self.get_forecast(model_i) for model_i in
    #                           self.models]
    #         test_args = (forecast, forecast_batch, catalog)
    #     else:
    #         test_args = (forecast, catalog)
    #     return test_args
