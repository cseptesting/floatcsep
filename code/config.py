# Python modules
import os
import datetime
import numpy
import matplotlib.pyplot as plt
import json

# pyCSEP modules
import csep.core.poisson_evaluations as poisson
from csep.core.catalogs import CSEPCatalog
import csep.utils.plots as plots
from csep.utils.time_utils import decimal_year
from csep.models import EvaluationResult
# Local modules
import evaluations
import utils
import accessors


test_type = {poisson.number_test: 'individual',
             poisson.magnitude_test: 'individual',
             poisson.spatial_test: 'individual',
             poisson.likelihood_test: 'individual',
             poisson.conditional_likelihood_test: 'individual',
             poisson.paired_t_test: 'comparative',
             poisson.w_test: 'comparative'}


experiment_config = {
    'num_sim': 10000,
    'verbose': True,
    'start_date': datetime.datetime(2020, 1, 1, 0, 0, 0),
    'end_date': datetime.datetime(2022, 12, 31, 23, 59, 59),
    'region': utils.global_region,
    'catalog': accessors.query_isc_gcmt,
    # Todo: Populate with all forecasts, here 'func' takes a model and returns a forecast
    'models': [
        {'name': 'GEAR1',
            'authors': ('Peter Bird, Yan Kagan, David Jackson, Ross Stein'),
            'func': utils.prepare_forecast,
            'func_args': {'model_path': '../models/GEAR1_csep.txt'},    #todo Clearly define absolute/relative paths >> @wsavran Im not 100% clear about handling this in a module.
            'doi': None,
            'markdown': '',
            # This list will get populated at run-time. For global experiment, we only have 1 forecast per model
            'forecasts': [
                            {
                                'start_date': '',
                                'end_date': '',
                                'plot_func': '',
                                'plot_args': '',
                                'forecast_filepath': '',
                                'figure_filepath': '',
                                'evaluations': [
                                    {
                                        'name': '',
                                        'config': '',
                                        'evaluation_path': '',
                                        'catalog_path': '',
                                        'figure_path': '',
                                        'result': '' # pyCSEP evaluation result object
                                    }
                                ]
                            }
                          ]
        },
        {
            'name': 'WHEEL',
            'authors': ('Jose A. Bayona and others'),
            'func': None,       # Function that returns a forecast from a model
            'func_args': None,  # Arguments to this function
            'doi': None,
            'markdown': '',
            # This will get populated at run-time. For global experiment, we only have 1 forecast per model
            'forecasts': [
                {
                    'start_date': '',
                    'end_date': '',
                    'plot_func': '',
                    'plot_args': '',
                    'forecast_filepath': '',
                    'figure_filepath': '',
                    'evaluations': [
                        {
                            'name': '',
                            'config': '',
                            'evaluation_path': '',
                            'catalog_path': '',
                            'figure_path': '',
                            'result': '' # pyCSEP evaluation result object
                        }
                    ]
                }
            ]
        }
    ],
    # Allows us to easily iterate over the evaluations that need to be run
    'evaluation_config': [
        {
            'name': 'Poisson N-Test',
            'func': poisson.number_test,
            'func_args': (),
            'plot_func': plots.plot_poisson_consistency_test,
            'plot_args': {},
            'markdown': ''
        },
        {
            'name': 'Poisson CL-Test',
            'func': poisson.conditional_likelihood_test,
            'plot_func': plots.plot_poisson_consistency_test,
            'plot_args': {},
            'markdown': ''
        },
        {
            'name': 'Poisson S-Test',
            'func': poisson.spatial_test,
            'plot_func': plots.plot_poisson_consistency_test,
            'plot_args': {},
            'markdown': ''
        },
        {
            'name': 'Poisson T-Test',
            'func': evaluations.matrix_poisson_t_test,
            'plot_func': utils.plot_matrix_comparative_test,
            'plot_args': {},
            'markdown': ''
        },
        {
            'name': 'Negative Binomial N-Test',
            'func': evaluations.negative_binomial_number_test,
            'plot_func': utils.plot_binary_consistency_test,
            'plot_args': {},
            'markdown': ''
        },
        {
            'name': 'Binomial CL-Test',
            'func': evaluations.binomial_conditional_likelihood_test,
            'plot_func': utils.plot_binary_consistency_test,
            'plot_args': {},
            'markdown': ''
        },
        {
            'name': 'Binomial S-Test',
            'func': evaluations.binomial_spatial_test,
            'plot_func': utils.plot_binary_consistency_test,
            'plot_args': {}, 
            'markdown': ''
        },
        {
            'name': 'Binomial T-Test',
            'func': evaluations.matrix_binary_t_test,
            'plot_func': utils.plot_matrix_comparative_test,
            'plot_args': {},
            'markdown': ''
        }
    ]
}

config_short = {
        'num_sim': 10,
        'verbose': True,
        'start_date': datetime.datetime(2020, 1, 1, 0, 0, 0),
        'end_date': datetime.datetime(2022, 12, 31, 23, 59, 59),
        'region': utils.global_region(2),
        # Todo: Populate with all forecasts, here 'func' takes a model and returns a forecast
        'models': [
            {'name': 'GEAR1',
                'authors': ('Peter Bird', 'Yan Kagan', 'David Jackson', 'Ross Stein'),
                'func': utils.prepare_forecast,
                'func_args': {'model_path': '../models/GEAR_resampled.txt', 'dh': 2},
                'doi': None,
                'markdown': '',
                # This list will get populated at run-time. For global experiment, we only have 1 forecast per model
                'forecasts': []
            },
            {'name': 'KJSS',
                'authors': ( 'Yan Kagan', 'David Jackson'),
                'func': utils.prepare_forecast,
                'func_args': {'model_path': '../models/KJSS_resampled.txt', 'dh': 2},
                'doi': None,
                'markdown': '',
                'forecasts': []
            },
            {'name': 'SHIFT2F_GSRM',
                'authors': ('Peter Bird', 'Kreemer'),
                'func': utils.prepare_forecast,
                'func_args': {'model_path': '../models/SHIFT2F_GSRM_resampled.txt', 'dh': 2},
                'doi': None, 'markdown': '',
                'forecasts': []
            },
            {'name': 'TEAMr',
                'authors': ('Jose Bayona'),
                'func': utils.prepare_forecast,
                'func_args': {'model_path': '../models/TEAMr_resampled.txt', 'dh': 2},
                'doi': None,
                'markdown': '',
                'forecasts': []
             },
            {'name': 'WHEELr',
                'authors': ('Jose Bayona'),
                'func': utils.prepare_forecast,
                'func_args': {'model_path': '../models/WHEELr_resampled.txt', 'dh': 2},
                'doi': None,
                'markdown': '',
                'forecasts': []
             },
        ],
        # Allows us to easily iterate over the evaluations that need to be run
        'evaluation_config': [
            {
                'name': 'Poisson N-Test',
                'func': poisson.number_test,
                'func_args': (),
                'plot_func': plots.plot_poisson_consistency_test,
                'plot_args': {},
                'markdown': ''
            }
        ]
    }


class Experiment:

    def __init__(self, start_date, end_date, region, cat_reader):

        self.start_date = start_date
        self.end_date = end_date
        self.region = region
        self.catalog_reader = cat_reader

        self.models = []
        self.tests = []
        self.evaluations = {}

    @staticmethod
    def get_run_struct(args):
        """
        Creates the run directory, and reads the file structure inside

        :param args: Dictionary containing the Experiment object and the Run arguments
        :return: run_folder: Path to the run
                 exists: flag if forecasts, catalogs and test_results if they exist already
                 target_paths: flag to each element of the experiment (catalog and evaluation results)
        """
        run_name = args['run_name']
        test_date = args['test_date']
        tests = [i['name'] for i in args['self'].tests]
        models = [i['name'] for i in args['self'].models]

        if run_name is None:
            run_name = test_date.isoformat()

        parent_dir = '../'  # todo Manage parent dir appropiately
        results_dir = 'results'
        run_folder = os.path.join(parent_dir, results_dir, run_name)
        folders = ['forecasts', 'catalog', 'evaluations', 'figures']

        folder_paths = {i: os.path.join(run_folder, i) for i in folders}

        for key, val in folder_paths.items():
            os.makedirs(val, exist_ok=True)

        files = {i: list(os.listdir(j)) for i, j in folder_paths.items()}
        exists = {'forecasts': False,  # Modify for time-dependent
                  'catalog': any(file for file in files['catalog']),
                  'evaluations': {test: {model: any(f'{test}_{model}' in file for file in files['evaluations'])
                                         for model in models}
                                  for test in tests}
                  }

        target_paths = {'forecasts': None, 'catalog': os.path.join(folder_paths['catalog'], 'catalog.json'),
                        'evaluations': {test: {model: os.path.join(folder_paths['evaluations'], f'{test}_{model}')
                                               for model in models}
                                        for test in tests}
                        }

        return run_folder, exists, target_paths

    def set_model(self, name, func, func_args, authors=None, doi=None, markdown=None, **kwargs):
        """
        Loads a model and its configurations to the experiment

        :param name: Name of the model
        :param func: Function that creates a forecast from the model
        :param func_args: Function arguments
        :param authors:
        :param doi:
        :param markdown: Template for the markdown
        :param kwargs:
        :return:
        """
        model = {'name': name,
                 'authors': authors,
                 'doi': doi,
                 'func': func,
                 'func_args': func_args,
                 'markdown': markdown,
                 'forecasts': []}

        # todo checks:  Repeated model? Does model file exists?
        self.models.append(model)

    def set_test(self, name, func, func_args, plot_func, plot_args=None, ref_model=None, **kwargs):
        """
        Loads a test configuration to the experiment

        :param name: Name of the test
        :param func: Function of the test evaluation
        :param func_args: Test function arguments
        :param plot_func: Test to plot the function
        :param plot_args: Arguments to the plot
        :param ref_model: Reference model for comparative tests
        :param kwargs:
        :return:
        """
        test = {'name': name,
                'func': func,
                'func_args': func_args,
                'plot_func': plot_func,
                'plot_args': plot_args}
        if ref_model:
            test['ref_model'] = ref_model

        self.tests.append(test)

    def create_forecast(self, model, test_date):
        """
        Creates a forecast from a model and a time window
        :param model: A model configuration dict
        :param test_date: A test date to calculate the horizon
        :return: A pycsep.core.forecasts.GriddedForecast object
        """
        time_horizon = decimal_year(test_date + datetime.timedelta(1)) - decimal_year(self.start_date)

        forecast = model['func'](name=model['name'],                             # todo add kwargs
                                 model_path=model['func_args']['model_path'],
                                 time_horizon=time_horizon,
                                 dh=model['func_args']['dh'])

        return forecast

    def get_catalog(self, test_date, path, min_mag=5.95, exists=None):
        """
        Gets the testing catalog from the catalog reader, query if does not exists, or parse from JSON if not
        #todo Should also pass arguments to a static catalog reader, or comcat
        :param test_date:
        :param path:
        :param min_mag:
        :param exists: If it exists in the structure, reads it
        :return:
        """
        if exists:
            catalog = CSEPCatalog.load_json(path)
        else:
            catalog = self.catalog_reader(cat_id=test_date,
                                          start_datetime=self.start_date,
                                          end_datetime=test_date,
                                          min_mw=min_mag, verbose=True)
            catalog.write_json(path)

        catalog.filter_spatial(self.region, update_stats=True, in_place=True)

        return catalog

    def run(self, test_date, new_run=False, run_name=None):
        """
        Main function of the experiment.
        - Creates the run folder structure
        - Reads the evaluation catalog
        - Run the tests for every model
        - Serializes the result
        - Reads a result if already exists
        - Creates the main evaluation dict, containing the experiment results

        :param test_date: Defines until when the experiment is run
        :param new_run: Flag to rerun an experiment
        :param run_name: Additional parameter if the run has a specific name. If not, the name is the test date
        :return:
        """
        run_folder, exists, target_paths = self.get_run_struct(locals())
        catalog = self.get_catalog(test_date, path=target_paths['catalog'], exists=exists['catalog'])
        eval = self.evaluations


        #### TODO, ALL FORECASTS SHOULD BE CREATED OUTSIDE THE LOOP
        for test in self.tests:

            if new_run or all(exists['evaluations'][test['name']].values()) is False:
                test_func = test.get('func')
                eval[test['name']] = {}
                if test_type[test_func] == 'individual':
                    for model in self.models:
                        forecast = self.create_forecast(model, test_date)
                        model_eval = test_func.__call__(forecast, catalog, **test.get('func_args'))
                        eval_path = target_paths['evaluations'][test['name']][model['name']]
                        with open(eval_path, 'w') as file_:
                            json.dump(model_eval.to_dict(), file_, indent=4)
                        eval[test['name']].update({'result': model_eval,
                                                   'path': eval_path})

                elif test_type[test.get('func')] == 'comparative':
                    ref_model = [i for i in self.models if i['name'] == test['ref_model']][0]
                    ref_forecast = self.create_forecast(ref_model, test_date)
                    eval[test['name']] = {'ref_model': ref_model['name']}
                    model_eval = []
                    for model in self.models:
                        if model['name'] == ref_model['name']:
                            continue
                        else:
                            forecast = self.create_forecast(model, test_date)
                            model_eval = test_func.__call__(forecast, ref_forecast, catalog)
                            eval_path = target_paths['evaluations'][test['name']][ref_model['name']]
                            with open(eval_path, 'w') as file_:
                                json.dump(model_eval.to_dict(), file_, indent=4)
                            eval[test['name']].update({'result': model_eval,
                                                       'path': eval_path})

                elif test_type[test_func] == 'matrix':
                    pass

            else:
                test_func = test.get('func')
                eval[test['name']] = {}
                if test_type[test_func] == 'individual':
                    for model in self.models:
                        eval_path = target_paths['evaluations'][test['name']][model['name']]
                        with open(eval_path, 'r') as file_:
                            model_eval = EvaluationResult.from_dict(json.load(file_))
                        eval[test['name']].update({'result': model_eval,
                                                   'path': eval_path})




