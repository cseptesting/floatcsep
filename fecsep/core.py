import copy
import os
import json

import cartopy.crs as ccrs
import six
from collections.abc import Iterable
import h5py
import yaml
from matplotlib import pyplot

from csep import GriddedForecast
from csep.models import EvaluationResult
from csep.utils.calc import cleaner_range
from csep.core.catalogs import CSEPCatalog
from csep.utils.time_utils import decimal_year
from csep.core.regions import QuadtreeGrid2D, CartesianGrid2D
from csep.models import Polygon
import fecsep
import fecsep.utils
import fecsep.accessors
import fecsep.evaluations
from fecsep.utils import MarkdownReport, NoAliasLoader, _set_dockerfile, parse_csep_func
from fecsep.accessors import from_zenodo, from_git
import docker
import docker.errors


client = docker.from_env()


class Model:
    def __init__(self, name, path, filename=None, format='quadtree', db_type=None, forecast_unit=1,
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
        self.name = name
        self.path = path
        self.filename = filename
        self.authors = authors
        self.doi = doi
        self.format = format
        self.db_type = db_type if db_type else self.format
        self.markdown = markdown
        self.forecast_unit = forecast_unit         # Model is defined as rates per 1 year
        self.forecasts = {}
        self.zenodo_id = zenodo_id
        self.giturl = giturl
        self.repo_hash = hash
        self.image = None
        self.bind = None
        self.func = func
        self.func_args = func_args
        os.makedirs(path, exist_ok=True)

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

    def get_source(self, force=False):

        if not os.path.isfile(os.path.join(self.path, self.filename if self.filename else '')):
            force = True
        if force and (self.zenodo_id or self.giturl):
            try:
                print('Retrieving from Zenodo')
                from_zenodo(self.zenodo_id, self.path)
            except KeyError or TypeError:
                print('Retrieving from git')
                from_git(self.giturl, self.path)
        else:
            print('Retrieving from local')

    def stage_db(self, force=False):
        """
        Stage model deployment.
        Checks download and builds image container.
        Makes a model forecast with desired params
        Transform to desired db format if asked

        Returns:

        """

        ## Creates one docker per repo
        img_name = os.path.basename(self.path).lower()
        if force:
            self.image = self.build_docker(img_name, self.path)[0]
        else:
            try:
                self.image = client.images.get(img_name)
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
                a = client.containers.run(self.image, remove=True,
                                      volumes={os.path.abspath(self.path): {'bind': self.bind, 'mode': 'rw'},
                                               os.path.abspath(fecsep.__path__[0]): {'bind': fecsep_bind, 'mode': 'ro'}},
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

        #todo implement these functions in dbparser
        if self.db_type == 'hdf5':
            with h5py.File(fn, 'r') as db:
                rates = db['rates'][:]  #todo check memory efficiency. Is it better to leave db open for multiple time intervals?
                magnitudes = db['magnitudes'][:]
                if self.format == 'quadtree':
                    region = QuadtreeGrid2D.from_quadkeys(db['quadkeys'][:].astype(str), magnitudes=magnitudes)
                    region.get_cell_area()
                elif self.format in ['dat', 'csep', 'xml']:
                    dh = db['dh'][:]
                    bboxes = db['bboxes'][:]
                    poly_mask = db['poly_mask'][:]
                    region = CartesianGrid2D([Polygon(bbox) for bbox in bboxes], dh, mask=poly_mask)
        forecast = GriddedForecast(
            name=name,
            data=rates,
            region=region,
            magnitudes=magnitudes,
            start_time=start_date,
            end_time=test_date
        )
        forecast = forecast.scale(time_horizon/self.forecast_unit)
        print(f"Expected forecast count after scaling: {forecast.event_count} with parameter {time_horizon}.")
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
    def __init__(self, name, func, markdown='', func_args=None, func_kwargs=None, plot_func=None,
                 plot_args=None, plot_kwargs=None, model=None, ref_model=None, path=None):
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
        self.func_kwargs = func_kwargs      # todo set default args from exp?
        self.func_args = func_args
        self.plot_func = parse_csep_func(plot_func)
        self.plot_args = plot_args or {}        # todo default args?
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

    def __init__(self, start_date, end_date, test_date=None, intervals=1,
                 name=None,
                 catalog_reader=None,
                 mag_min=None, mag_max=None, mag_bin=None,
                 depth_min=None, depth_max=None, region=None,
                 models_config=None, tests_config=None, postproc_config=None,
                 default_test_kwargs=None, **kwargs):

        self.name = name
        self.start_date = start_date
        self.end_date = end_date
        self.intervals = intervals
        self.test_date = test_date
        self.catalog_reader = parse_csep_func(catalog_reader)
        self.models_config = models_config
        self.tests_config = tests_config
        self.postproc_config = postproc_config if postproc_config else {}
        self.default_test_kwargs = default_test_kwargs
        self.models = []
        self.tests = []
        self.run_results = {}
        self.set_magnitude_range(mag_min, mag_max, mag_bin)
        self.set_depth_range(depth_min, depth_max)
        self.region = parse_csep_func(region)() if region else None
        self.__dict__.update(kwargs)

    def get_run_struct(self, run_name=None):
        """
        Creates the run directory, and reads the file structure inside

        :param args: Dictionary containing the Experiment object and the Run arguments
        :return: run_folder: Path to the run
                 exists: flag if forecasts, catalogs and test_results if they exist already
                 target_paths: flag to each element of the gefe (catalog and evaluation results)
        """

        if self.test_date is None:
            raise RuntimeError("Test date must be set before running gefe.")

        # grab names for creating directories
        tests = [i.name for i in self.tests]
        models = [i.name for i in self.models]

        # use the test date by default
        if run_name is None:
            run_name = self.test_date.isoformat().replace('-', '').replace(':', '')

        # determine required directory structure for run
        parent_dir = '.'
        results_dir = 'results'
        run_folder = os.path.join(parent_dir, results_dir, run_name)
        folders = ['catalog', 'evaluations', 'figures']
        folder_paths = {folder: os.path.join(run_folder, folder) for folder in folders}

        # create directories if they don't exist
        for key, val in folder_paths.items():
            os.makedirs(val, exist_ok=True)

        files = {name: list(os.listdir(path)) for name, path in folder_paths.items()}
        exists = {
            'models': False,  # Modify for time-dependent
            'catalog': any(file for file in files['catalog']),
            'evaluations': {
                test: {
                    model: any(f'{test}_{model}.json' in file for file in files['evaluations'])
                    for model in models
                }
                for test in tests
            }
        }

        target_paths = {
            'models': {'figures': {model: os.path.join(folder_paths['figures'],
                                                       f'{model}') for model in models}},
            'catalog': os.path.join(folder_paths['catalog'], 'catalog.json'),
            'evaluations': {
                test: {
                    model: os.path.join(folder_paths['evaluations'], f'{test}_{model}.json')
                    for model in models
                }
                for test in tests
            },
            'figures': {test: os.path.join(folder_paths['figures'], f'{test}') for test in tests}
        }

        # store in gefe configuration
        self.run_folder = run_folder
        self.target_paths = target_paths
        self.exists = exists

        return run_folder, exists, target_paths

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
            forecast = model.create_forecast(self.start_date, self.end_date, name=model.name)
        return forecast

    def set_models(self):
        """
        Loads a model and its configurations to the gefe

        :param models: list of Model objects
        :return:
        """

        # todo checks:  Repeated model? Does model file exists?
        with open(self.models_config, 'r') as config:
            config_dict = yaml.load(config, NoAliasLoader)
            for element in config_dict:
                # Check if the model has multiple submodels from its repository
                if any('flavours' in i for i in element.values()):
                    for flav, flav_path in list(element.values())[0]['flavours'].items():
                        name_root = next(iter(element))
                        name_flav = f'{name_root}_{flav}'
                        model_ = {name_flav: {**element[name_root],
                                              'filename': flav_path}}
                        model_[name_flav].pop('flavours')
                        self.models.append(Model.from_dict(model_))
                else:
                    self.models.append(Model.from_dict(element))

    def set_tests(self):
        """
        Loads a test configuration file to the gefe

        :param tests
        :return:
        """

        with open(self.tests_config, 'r') as config:
            config_dict = yaml.load(config, NoAliasLoader)
            self.tests = [Test.from_dict(tdict) for tdict in config_dict]

    def set_catalog_reader(self, loader):
        self.catalog_reader = loader

    def set_test_date(self, date):
        self.test_date = date

    def set_magnitude_range(self, mw_min, mw_max, mw_inc):
        self.magnitude_range = cleaner_range(mw_min, mw_max, mw_inc)

    def set_depth_range(self, min_depth, max_depth):
        self.depth_range = cleaner_range(min_depth, max_depth, max_depth - min_depth)

    def get_catalog(self):
        """ Returns filtered catalog either from a previous run or for a new run downloads from ISC gCMT catalogue.

        This function is passively optimized for the global gefe. Meaning that no filtering needs to
        occur aside from magnitudes.

        :param region:
        :param path:
        :return:
        """
        if hasattr(self, 'catalog'):
            catalog = self.catalog
        elif os.path.exists(self.target_paths['catalog']):
            print(f"Catalog found at {self.target_paths['catalog']}. Using existing filtered catalog...")
            catalog = CSEPCatalog.load_json(self.target_paths['catalog'])
            self.set_catalog(catalog)
        else:
            print("Downloading catalog from ISC gCMT service...")
            min_mag = self.magnitude_range.min()
            max_depth = self.depth_range.max()
            if self.region is not None:
                bounds = {i: j for i,j in zip(['min_longitude', 'max_longitude',
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

    def run_test(self, test, write=True):
        # requires that test be fully configured, probably by calling enumerate_tests() first
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
                    name = test.name,
                    func = test.func,
                    func_args = self._prepare_test_func_args(test, model),
                    func_kwargs = test.func_kwargs,
                    plot_func = test.plot_func,
                    plot_args = test.plot_args,
                    model = model,
                    path = self.target_paths['evaluations'][test.name][model.name],
                    ref_model = test.ref_model
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
            forecast_batch = [self.get_forecast(model_i) for model_i in self.models]
            test_args = (forecast, forecast_batch, catalog)
        else:
            test_args = (forecast, catalog)
        return test_args

    def read_evaluation_result(self, test, models, target_paths):
        test_results = []
        if 'T' in test.name:    #todo cleaner
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
            ax = test.plot_func(test_result, plot_args=test.plot_args, **test.plot_kwargs)
            if 'code' in test.plot_args:
                exec(test.plot_args['code'])
            pyplot.savefig(file_paths['figures'][test.name], dpi=dpi)
            if show:
                pyplot.show()

        #todo create different method for forecast plotting
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
                plot_fc_config['projection'] = getattr(ccrs, proj_name)(**proj_args)
            except:
                plot_fc_config['projection'] = ccrs.PlateCarree(central_longitude=0.0)

            cat = plot_fc_config.get('catalog')
            if cat:
                cat_args = {'markersize': 7, 'markercolor': 'black', 'title': None,
                            'legend': False, 'basemap': None, 'region_border': False}
                if self.region:
                    self.catalog.filter_spatial(self.region, in_place=True)
                if isinstance(cat, dict):
                    cat_args.update(cat)

            for model in self.models:
                fig_path = self.target_paths['models']['figures'][model.name]
                start = decimal_year(self.start_date)
                end = decimal_year(self.test_date)
                time = f'{round(end-start,3)} years'
                plot_args = {'region_border': False,
                             'cmap': 'magma',
                             'clabel': r'$\log_{10} N\left(M_w \in [{%.2f},\,{%.2f}]\right)$ per '
                                       r'$0.1^\circ\times 0.1^\circ $ per %s' %
                                       (min(self.magnitude_range), max(self.magnitude_range), time)}
                if not self.region:
                    set_global = True
                else:
                    set_global = False
                plot_args.update(plot_fc_config)
                ax = model.forecasts[self.test_date].plot(set_global=set_global, plot_args=plot_args)

                if self.region:
                    bbox = self.region.get_bbox()
                    dh = self.region.dh
                    extent = [bbox[0] - 3*dh , bbox[1] + 3*dh,
                              bbox[2] - 3*dh, bbox[3] + 3*dh]
                else:
                    extent = None
                if cat:
                    self.catalog.plot(ax=ax, set_global=set_global, extent=extent, plot_args=cat_args)

                pyplot.savefig(fig_path, dpi=300, facecolor=(0, 0, 0, 0))

    def generate_report(self):
        report = MarkdownReport()
        report.add_title(
            "Global Earthquake Forecasting Experiment -- Quadtree",
            "The RISE (Real-time earthquake rIsk reduction for a reSilient Europe, "
            "[http://www.rise-eu.org/](http://www.rise-eu.org/) research group in collaboration "
            "with CSEP (Collaboratory for the Study of Earthquake Predictability, "
            "[https://cseptesting.org/](https://cseptesting.org/) is conducting a global "
            "earthquake forecast experiments using multi-resolution grids implemented as a quadtree."
        )
        report.add_heading("Objectives", level=2)
        objs = [
            "Describe the predictive skills of posited hypothesis about seismogenesis with earthquakes of "
            "M5.95+ independent observations around the globe.",
            "Identify the methods and geophysical datasets that lead to the highest information gains in "
            "global earthquake forecasting.",
            "Test earthquake forecast models on different grid settings.",
            "Use Quadtree based grid to represent and evaluate earthquake forecasts."
        ]
        report.add_list(objs)
        # Generate plot of the catalog
        if self.catalog is not None:
            figure_path = os.path.splitext(self.target_paths['catalog'])[0]
            # relative to top-level directory
            if self.region:
                self.catalog.filter_spatial(self.region, in_place=True)
            ax =self.catalog.plot(plot_args={
                'figsize': (12, 8),
                'markersize': 8,
                'markercolor': 'black',
                'grid_fontsize': 16,
                'title': '',
                'legend': False
            })
            ax.get_figure().tight_layout()
            ax.get_figure().savefig(f"{figure_path}.png")
            report.add_figure(
                f"ISC gCMT Authoritative Catalog",
                figure_path.replace(self.run_folder, '.'),
                level=2,
                caption="The authoritative evaluation data is the full Global CMT catalog (Ekström et al. 2012). "
                        "We confine the hypocentral depths of earthquakes in training and testing datasets to a "
                       f"maximum of 70km. The plot shows the catalog for the testing period which ranges from "
                       f"{self.start_date} until {self.test_date}. "
                       f"Earthquakes are filtered above Mw {self.magnitude_range.min()}. "
                        "Black circles depict individual earthquakes with its radius proportional to the magnitude.",
                add_ext = True
            )
        report.add_heading(
            "Results",
            level=2,
            text="We apply the following tests to each of the forecasts considered in this gefe. "
                 "More information regarding the tests can be found [here](https://docs.cseptesting.org/getting_started/theory.html)."
            )
        test_names = [test.name for test in self.tests]
        report.add_list(test_names)

        # Include results from Experiment
        for test in self.tests:
            fig_path = self.target_paths['figures'][test.name]
            report.add_figure(
                f"{test.name}",
                fig_path.replace(self.run_folder, '.'),
                level=3,
                caption=test.markdown,
                add_ext=True
            )

        report.table_of_contents()
        report.save(self.run_folder)

    def to_dict(self):
        out = {}
        excluded = ['run_results', 'magnitude_range', 'depth_range', 'region']

        def _get_value(x):
            if hasattr(x, 'to_dict'):
                o = x.to_dict()
            else:
                try:
                    o = x.__name__
                except AttributeError:
                    o = x
            return o

        for k, v in self.__dict__.items():
            if k not in excluded:
                if isinstance(v, Iterable) and not isinstance(v, six.string_types):
                    out[k] = []
                    for item in v:
                        out[k].append(_get_value(item))
                else:
                    out[k] = _get_value(v)
        return out

    def to_yaml(self):
        class NoAliasDumper(yaml.Dumper):
            def ignore_aliases(self, data):
                return True

        return yaml.dump(
            self.to_dict(),
            Dumper=NoAliasDumper,
            sort_keys=False,
            default_flow_style=False,
            indent=1
        )

    @classmethod
    def from_yaml(cls, config_yml):

        with open(config_yml, 'r') as yml:
            config_dict = yaml.safe_load(yml)
        return cls(**config_dict)

