import copy
import os
import datetime
import json
import six
from collections.abc import Iterable

import yaml
from matplotlib import pyplot

from csep import GriddedForecast
from csep.models import EvaluationResult
from csep.utils.calc import cleaner_range
from csep.core.catalogs import CSEPCatalog
from csep.utils.time_utils import decimal_year

from gefe.utils import MarkdownReport

class Model:
    def __init__(self, name, path, func, func_args, authors=None, doi=None, markdown=None):
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
        self.authors = authors
        self.doi = doi
        self.func = func
        self.func_args = func_args
        self.markdown = markdown
        self.forecast_unit = 1          # Model is defined as rates per 1 year
        self.forecasts = {}

    def create_forecast(self, start_date, test_date, name=None):
        """
        Creates a forecast from a model and a time window
        :param model: A model configuration dict
        :param test_date: A test date to calculate the horizon
        :return: A pycsep.core.forecasts.GriddedForecast object
        """

        time_horizon = decimal_year(test_date) - decimal_year(start_date)
        print(f"Loading model from {self.path}...")
        rates, region, mws = self.func(self.path)
        forecast = GriddedForecast(
            name=name,
            data=rates,
            region=region,
            magnitudes=mws,
            start_time=start_date,
            end_time=test_date
        )
        forecast = forecast.scale(time_horizon)
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
        self.func = func
        self.func_kwargs = func_kwargs      # todo Set default args?
        self.func_args = func_args
        self.plot_func = plot_func          # todo Should this function be assigned, the same way as TEST_TYPE???? (see line 8)
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

class Experiment:

    def __init__(self, start_date, end_date, test_date=None, catalog_reader=None, name=None):

        self.name = name
        self.start_date = start_date
        self.end_date = end_date
        self.test_date = test_date
        self.catalog_reader = catalog_reader

        self.models = []
        self.tests = []
        self.run_results = {}

    def get_run_struct(self, run_name=None):
        """
        Creates the run directory, and reads the file structure inside

        :param args: Dictionary containing the Experiment object and the Run arguments
        :return: run_folder: Path to the run
                 exists: flag if forecasts, catalogs and test_results if they exist already
                 target_paths: flag to each element of the experiment (catalog and evaluation results)
        """

        if self.test_date is None:
            raise RuntimeError("Test date must be set before running experiment.")

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
            'models': None,
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

        # store in experiment configuration
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

    def set_models(self, models):
        """
        Loads a model and its configurations to the experiment

        :param models: list of Model objects
        :return:
        """

        # todo checks:  Repeated model? Does model file exists?
        self.models = models

    def set_tests(self, tests):
        """
        Loads a test configuration to the experiment

        :param tests
        :return:
        """
        self.tests = tests

    def set_catalog_reader(self, loader):
        self.catalog_reader = loader

    def set_test_date(self, date):
        self.test_date = date

    def set_magnitude_range(self, mw_min, mw_max, mw_inc):
        self.magnitude_range = cleaner_range(mw_min, mw_max, mw_inc)

    def get_catalog(self):
        """ Returns filtered catalog either from a previous run or for a new run downloads from ISC gCMT catalogue.

        This function is passively optimized for the global experiment. Meaning that no filtering needs to
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
            catalog = self.catalog_reader(
                cat_id=self.test_date,
                start_datetime=self.start_date,
                end_datetime=self.test_date,
                min_mw=min_mag,
                verbose=True
            )
            # Quadtree rules require that forecasts and catalogs are within lats = [-85.0, 85.0]
            # todo: min/max lat should be set in config.py
            catalog = catalog.filter([
                'latitude > -85.0',
                'latitude < 85.0'
            ])
            catalog = catalog.filter('depth < 70')
            self.set_catalog(catalog)
        return catalog

    def set_catalog(self, catalog):
        self.catalog = catalog

    def run_test(self, test, write=True):
        # requires that test be fully configured, probably by calling enumerate_tests() first
        result = test.compute()
        if write:
            with open(test.path, 'w') as _file:
                json.dump(result.to_dict(), _file, indent=4)
        return result

    def prepare_all_tests(self):
        """ Prepare test to be run for the experiment by including runtime arguments like forecasts and catalogs

        :return tests: Complete list of evaluations to run for experiment
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
        if test.ref_model is not None:
            ref_model = self.get_model(test.ref_model)
            test_args = (forecast, self.get_forecast(ref_model), catalog)
        else:
            test_args = (forecast, catalog)
        return test_args

    def read_evaluation_result(self, test, models, target_paths):
        test_results = []
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
            test.plot_func(test_result, plot_args=test.plot_args, **test.plot_kwargs)
            pyplot.savefig(file_paths['figures'][test.name], dpi=dpi)
            if show:
                pyplot.show()

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
            text="We apply the following tests to each of the forecasts considered in this experiment. "
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
        excluded = ['run_results', 'magnitude_range']

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
