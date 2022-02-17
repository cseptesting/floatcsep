# Python modules
import datetime

# pyCSEP modules
from csep.core.regions import italy_testing_region
import csep.poisson_evaluations as poisson
import csep.utils.plots as plots

# Local modules
import tests

# Todo: Populate experiment configuration
experiment_config = {
    'num_sim': 10000,
    'verbose': True,
    'start_date': datetime.datetime(1,1,2022,0,0,0)
    'end_date': datetime.datetime(12,31,2023,23,59.999)
    'region': italy_testing_region
    # Todo: Populate with all forecasts, here 'func' takes a model and returns a forecast
    'models': [
        {
            'name': 'GEAR1',
            'authors': ('Peter Bird, Yan Kagan, David Jackson, Han Bao'),
            'func': None,
            'func_args': None,
            'doi': None,
            'markdown': ''
            # This list will get populated at run-time. For global experiment, we only have 1 forecast per model
            'forecasts': [
                {
                    'start_date': '',
                    'end_date': '',
                    'plot_func': '',
                    'plot_args': '',
                    'forecast_filepath': '',
                    'figure_filepath': ''
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
            'markdown': ''
            # This will get populated at run-time. For global experiment, we only have 1 forecast per model
            'forecasts': [
                {
                    'start_date': '',
                    'end_date': '',
                    'plot_func': '',
                    'plot_args': '',
                    'forecast_filepath': '',
                    'figure_filepath': ''
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
    ]
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
            'func_args': (num_simulations=experiment_config['num_sim'], verbse=experiment_config['verbose']),
            'plot_func': plots.plot_poisson_consistency_test,
            'plot_args': {},
            'markdown': ''
        }
        {
            'name': 'Poisson S-Test',
            'func': poisson.spatial_test,
            'func_args': (num_simulations=experiment_config['num_sim'], verbse=experiment_config['verbose']), 
            'plot_func': plots.plot_poisson_consistency_test,
            'plot_args': {},
            'markdown': ''
        },
        {
            'name': 'Poisson T-Test',
            'func': tests.matrix_poisson_t_test,
            'func_args': (),
            'plot_func': utils.plot_matrix_comparative_test,
            'plot_args': {},
            'markdown': ''
        },
        {
            'name': 'Negative Binomial N-Test',
            'func': tests.negative_binomial_number_test,
            'func_args': (),
            'plot_func': utils.plot_binary_consistency_test,
            'plot_args': {},
            'markdown': ''
        },
        {
            'name': 'Binomial CL-Test',
            'func': tests.binomial_conditional_likelihood_test,
            'func_args': (),
            'plot_func': utils.plot_binary_consistency_test,
            'plot_args': {},
            'markdown': ''
        },
        {
            'name': 'Binomial S-Test',
            'func': tests.binomial_spatial_test,
            'func_args': (),
            'plot_func': utils.plot_binary_consistency_test,
            'plot_args': {}, 
            'markdown': ''
        },
        {
            'name': 'Binomial T-Test',
            'func': tests.matrix_binary_t_test,
            'func_args': (),
            'plot_func': utils.plot_matrix_comparative_test,
            'plot_args': {},
            'markdown': ''
        }
    ]
}

class Experiment:
    def __init__(self, run_date, models, evaluation_config):
        self.start_date = global_config['start_date']
        self.end_date = run_date
        self.models = models
        self.evaluation_config = evaluation_config
        

    def time_horizon(self, test_date):
        """ Returns the time horizon in years given the test date

            Catalogs are filtered using 
                catalogs.filter(['datetime >= self.start_date', 'datetime < test_date'])
            
            Args:
                test_date (datetime): date to evaluate forecasts

            Returns:
                time_horizon (float): time horizon of experiment in years
        """
        pass
