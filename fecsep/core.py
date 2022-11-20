import copy
import json

import git
import numpy
import cartopy.crs as ccrs
from collections.abc import Mapping, Sequence
import h5py
import yaml
from matplotlib import pyplot
from datetime import datetime
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
    parse_csep_func, read_time_config, read_region_config
from fecsep.accessors import from_zenodo, from_git
import docker
import docker.errors

_client = docker.from_env()


class Model:
    def __init__(self, name, path, format='quadtree',
                 db_type=None, forecast_unit=1,
                 authors=None, doi=None, markdown=None,
                 func=None, func_args=None,
                 zenodo_id=None, giturl=None, repo_hash=None):
        """

        Model constructor

        :param name: Name of the model
        :param func: Function that creates a forecast from the model
        :param func_args: Function arguments
        :param authors:
        :param doi:
        :param markdown: Template for the markdown
        :param kwargs:
        :return:
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

        self.name = name
        self.path = path
        self._dir = None

        self.zenodo_id = zenodo_id
        self.giturl = giturl
        self.repo_hash = repo_hash

        self.get_source(zenodo_id, giturl)

        self.format = format

        self.authors = authors
        self.doi = doi
        self.db_type = db_type if db_type else self.format
        self.markdown = markdown
        self.forecast_unit = forecast_unit
        self.forecasts = {}

        self.image = None
        self.bind = None
        self.func = func
        self.func_args = func_args

    def get_source(self, zenodo_id=None, giturl=None, **kwargs):
        """



        Args:
            zenodo_id:
            giturl:
            **kwargs:

        Returns:

        """

        is_file = bool(os.path.splitext(self.path)[-1])
        if is_file:
            self._dir = os.path.dirname(self.path)
        else:
            self._dir = self.path

        if not os.path.exists(self.path):
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

        # Check paths
        if is_file:
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

    def stage_db(self, force=False):
        """
        Stage model deployment.
        Checks download and builds image container.
        Makes a model forecast with desired params
        Transform to desired db format if asked

        Returns:

        """
        # Creates one docker per repo
        img_name = os.path.basename(self.path).lower()
        if force:
            self.image = self.build_docker(img_name, self.path)[0]
        else:
            try:
                self.image = _client.images.get(img_name)
            except docker.errors.ImageNotFound:
                self.image = self.build_docker(img_name, self.path)[0]
        self.bind = self.image.attrs['Config']['WorkingDir']

        if self.db_type in ['hdf5']:
            fn_h5 = os.path.splitext(self.filename)[0] + '.hdf5'
            path_h5 = os.path.join(self.path, fn_h5)
            if os.path.isfile(path_h5):
                self.filename = fn_h5
            else:
                fecsep_bind = f'/usr/src/fecsep'
                cmd = f'python {fecsep_bind}/dbparser.py --format {self.format} --filename {self.filename}'
                a = _client.containers.run(self.image, remove=True,
                                           volumes={
                                               os.path.abspath(self.path): {
                                                   'bind': self.bind,
                                                   'mode': 'rw'},
                                               os.path.abspath(
                                                   fecsep.__path__[0]): {
                                                   'bind': fecsep_bind,
                                                   'mode': 'ro'}},
                                           command=cmd)
                self.filename = fn_h5

    def create_forecast(self, start_date, test_date, name=None):
        """
        Creates a forecast from a model and a time window
        :param model: A model configuration dict
        :param test_date: A test date to calculate the horizon
        :return: A pycsep.core.forecasts.GriddedForecast object
        """

        time_horizon = decimal_year(test_date) - decimal_year(start_date)
        print(f"Loading model from {self.path}...")
        fn = os.path.join(self.path, self.filename)

        # todo implement these functions in dbparser
        if self.db_type == 'hdf5':
            with h5py.File(fn, 'r') as db:
                rates = db['rates'][
                        :]  # todo check memory efficiency. Is it better to leave db open for multiple time intervals?
                magnitudes = db['magnitudes'][:]
                if self.format == 'quadtree':
                    region = QuadtreeGrid2D.from_quadkeys(
                        db['quadkeys'][:].astype(str), magnitudes=magnitudes)
                    region.get_cell_area()
                elif self.format in ['dat', 'csep', 'xml']:
                    dh = db['dh'][:]
                    bboxes = db['bboxes'][:]
                    poly_mask = db['poly_mask'][:]
                    region = CartesianGrid2D(
                        [Polygon(bbox) for bbox in bboxes], dh, mask=poly_mask)
        forecast = GriddedForecast(
            name=name,
            data=rates,
            region=region,
            magnitudes=magnitudes,
            start_time=start_date,
            end_time=test_date
        )
        forecast = forecast.scale(time_horizon / self.forecast_unit)
        print(
            f"Expected forecast count after scaling: {forecast.event_count} with parameter {time_horizon}.")
        self.forecasts[test_date] = forecast
        return forecast

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

    def compute(self):
        print(f"Computing {self.name} for model {self.model.name}...")
        return self.func(*self.func_args, **self.func_kwargs)

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
            f"model: {self.model.name}\n"
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
                 catalog_reader=None,
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
        self.catalog_reader = parse_csep_func(catalog_reader)

        self.model_config = model_config
        self.test_config = test_config
        self.postproc_config = postproc_config if postproc_config else {}
        self.default_test_kwargs = default_test_kwargs

        # Initialize class attributes
        self.models = []
        self.tests = []
        self.run_results = {}
        self.catalog = None
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

        # todo:  extrapolate to multiple test_dates
        if self.test_date is None:
            raise RuntimeError(
                "Test date must be set before running experiment.")

        # grab names for creating directories
        tests = [i.name for i in self.tests]
        models = [i.name for i in self.models]

        # use the test date by default
        # todo create datetime parser for filenames
        if run_name is None:
            run_name = self.test_date.isoformat().replace('-', '').replace(':',
                                                                           '')

        # determine required directory structure for run
        run_folder = os.path.join(os.getcwd(), results_path or 'results',
                                  run_name)

        # results > test_date > cats / evals / figures
        folders = ['catalog', 'evaluations', 'figures']
        folder_paths = {folder: os.path.join(run_folder, folder) for folder in
                        folders}

        # create directories if they don't exist
        for key, val in folder_paths.items():
            os.makedirs(val, exist_ok=True)

        files = {name: list(os.listdir(path)) for name, path in
                 folder_paths.items()}
        exists = {
            'models': False,  # Modify for time-dependent
            'catalog': any(file for file in files['catalog']),
            'evaluations': {
                test: {
                    model: any(f'{test}_{model}.json' in file for file in
                               files['evaluations'])
                    for model in models
                }
                for test in tests
            }
        }

        target_paths = {
            'models': {
                'forecasts': {model: os.path.join(model.path, model.filename)
                              for model in models},
                'figures': {model: os.path.join(folder_paths['figures'],
                                                f'{model}') for model in
                            models}},
            'catalog': os.path.join(folder_paths['catalog'], 'catalog.json'),
            'evaluations': {
                test: {
                    model: os.path.join(folder_paths['evaluations'],
                                        f'{test}_{model}.json')
                    for model in models
                }
                for test in tests
            },
            'figures': {test: os.path.join(folder_paths['figures'], f'{test}')
                        for test in tests}
        }

        self.run_folder = run_folder
        self.target_paths = target_paths
        self.exists = exists

    def get_model(self, name):
        for model in self.models:
            if model.name == name:
                return model

    def get_forecast(self, model):
        # if already bound to model class, simply return
        try:
            forecast = model.forecasts[self.test_date]
        except KeyError:
            # this call binds to model class
            forecast = model.create_forecast(self.start_date, self.end_date,
                                             name=model.name)
        return forecast

    def set_catalog_reader(self, loader):
        self.catalog_reader = loader

    def get_catalog(self):
        """ Returns filtered catalog either from a previous run or for a new
        run downloads from ISC gCMT catalogue.

        This function is passively optimized for the global gefe. Meaning that
         no filtering needs to occur aside from magnitudes.

        :return:
        """
        if hasattr(self, 'catalog'):
            catalog = self.catalog
        elif os.path.exists(self.target_paths['catalog']):
            print(
                f"Catalog found at {self.target_paths['catalog']}."
                f" Using existing filtered catalog...")
            catalog = CSEPCatalog.load_json(self.target_paths['catalog'])
            self.set_catalog(catalog)
        else:
            print("Downloading catalog")
            min_mag = self.magnitude_range.min()
            max_depth = self.depth_range.max()
            if self.region is not None:
                bounds = {i: j for i, j in
                          zip(['min_longitude', 'max_longitude',
                               'min_latitude', 'max_latitude'],
                              self.region.get_bbox())}
            else:
                bounds = {}
            catalog = self.catalog_reader(
                catalog_id=self.test_date,
                start_time=self.start_date,
                end_time=self.test_date,
                min_magnitude=min_mag,
                max_depth=max_depth,
                verbose=True,
                **bounds
            )

            self.set_catalog(catalog)
        if not os.path.exists(self.target_paths['catalog']):
            catalog.write_json(self.target_paths['catalog'])

        return catalog

    def set_catalog(self, catalog):
        self.catalog = catalog

    def stage_models(self, force=False):
        for model in self.models:
            model.get_source(force)
            model.stage_db(force)

    @staticmethod
    def run_test(test, write=True):
        # requires that test be fully configured, probably by calling
        # enumerate_tests() first
        result = test.compute()
        if write:
            with open(test.path, 'w') as _file:
                json.dump(result.to_dict(), _file, indent=4)
        return result

    def prepare_all_tests(self):
        """ Prepare test to be run for the gefe by including runtime arguments like forecasts and catalogs

        :return tests: Complete list of evaluations to run for gefe
        """
        # prepares arguments for test
        test_list = []
        for model in self.models:
            for test in self.tests:
                # skip t-test if model is the same as ref_model
                if test.ref_model == model.name:
                    continue
                # prepare args so test is callable
                t = Test(
                    name=test.name,
                    func=test.func,
                    func_args=self._prepare_test_func_args(test, model),
                    func_kwargs=test.func_kwargs,
                    plot_func=test.plot_func,
                    plot_args=test.plot_args,
                    model=model,
                    path=self.target_paths['evaluations'][test.name][
                        model.name],
                    ref_model=test.ref_model
                )
                test_list.append(t)
                print("Prepared...\n", t)
        return test_list

    def _prepare_test_func_args(self, test, model):
        forecast = self.get_forecast(model)
        catalog = copy.deepcopy(self.get_catalog())
        catalog.region = forecast.region
        if self.region:
            catalog.filter_spatial(in_place=True)
        if test.ref_model is not None:
            ref_model = self.get_model(test.ref_model)
            test_args = (forecast, self.get_forecast(ref_model), catalog)
        elif test.func == fecsep.evaluations.vector_poisson_t_w_test:
            forecast_batch = [self.get_forecast(model_i) for model_i in
                              self.models]
            test_args = (forecast, forecast_batch, catalog)
        else:
            test_args = (forecast, catalog)
        return test_args

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
