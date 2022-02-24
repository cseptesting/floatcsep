# Python modules
import datetime

# pyCSEP modules
import matplotlib.pyplot as plt
import csep.core.poisson_evaluations as poisson
import csep.utils.plots as plots
from csep.utils.calc import cleaner_range
from csep.utils.time_utils import decimal_year, datetime_to_utc_epoch


# Local modules
import evaluations
import utils
import accessors
import numpy


global_config = {'start_date': datetime.datetime(2020, 1, 1, 0, 0, 0)}

# Todo: Populate experiment configuration
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

    def __init__(self, start_date, end_date, region):

        self.start_date = start_date
        self.end_date = end_date
        self.region = region

        self.models = []
        self.tests = []
        self.results = []

    def set_model(self, name, func, func_args, authors=None, doi=None, markdown=None, **kwargs):

        model = {'name': name,
                 'authors': authors,
                 'doi': doi,
                 'func': func,
                 'func_args': func_args,
                 'markdown': markdown,
                 'forecasts': []}

        # todo checks:  Repeated model? Does model file exists?
        self.models.append(model)

    def set_test(self, name, func, func_args, plot_func, plot_args={}, **kwargs):

        test = {'name': name,
                'func': func,
                'func_args': func_args,
                'plot_func': plot_func,
                'plot_args': plot_args}
        # todo checks:  Repeated test? Does test results exists?
        self.tests.append(test)


    def time_horizon(self, test_date):
        """ Returns the time horizon in years given the test date

            Catalogs are filtered using 
                catalogs.filter(['datetime >= self.start_date', 'datetime < test_date'])
            
            Args:
                test_date (datetime): date to evaluate forecasts

            Returns:
                time_horizon (float): time horizon of experiment in years
        """

        # we are adding one day, bc tests are considered to occur at the end of the day specified by test_datetime.
        test_date_dec = decimal_year(test_date + datetime.timedelta(1))

        return test_date_dec - decimal_year(self.start_date)


    def create_forecast(self, model, test_date):

        time_horizon = self.time_horizon(test_date)
        forecast = model['func'](name=model['name'],                             # todo add kwargs
                                 model_path=model['func_args']['model_path'],
                                 time_horizon=time_horizon,
                                 dh=model['func_args']['dh'])

        return forecast

    def get_catalog(self, test_date, min_mag=5.95):

        catalog = accessors.query_isc_gcmt(start_datetime=self.start_date,
                                           end_datetime=test_date,
                                           min_mw=min_mag)   # modify
        catalog.filter_spatial(self.region, update_stats=True, in_place=True)
        self.catalog = catalog



    def run(self, test_date):

        self.get_catalog(test_date)

        for test in self.tests:
            results = []
            for model in self.models:
                forecast = self.create_forecast(model, test_date)
                # return forecast
                results.append(test.get('func').__call__(forecast, self.catalog, **test.get('func_args')))

        return results



if __name__ == '__main__':

    dh = 2
    mag_bins = cleaner_range(5.95, 8.95, 0.1)
    region = utils.global_region(dh, magnitudes=mag_bins)
    start_date = datetime.datetime(2020, 1, 1, 0, 0, 0)
    test_date = datetime.datetime(2021, 1, 1, 0, 0, 0)
    end_date = datetime.datetime(2022, 1, 1, 0, 0, 0)

    exp = Experiment(start_date, end_date, region)
    exp.set_model('GEAR1', utils.prepare_forecast, {'model_path': '../models/GEAR_resampled.txt', 'dh': dh})
    exp.set_model('KJSS', utils.prepare_forecast, {'model_path': '../models/KJSS_resampled.txt', 'dh': dh})
    exp.set_model('WHEELr', utils.prepare_forecast, {'model_path': '../models/WHEELr_resampled.txt', 'dh': dh})
    exp.set_test('Poisson CL', poisson.conditional_likelihood_test, {'num_simulations': 10, 'seed': 23},
                               plots.plot_poisson_consistency_test)

    a = exp.run(test_date)
    plots.plot_poisson_consistency_test(a)
    plt.show()
    print('asd')

