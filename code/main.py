# python libraries
import datetime

# pyCSEP libraries
from csep.utils.calc import cleaner_range
import csep.core.poisson_evaluations as poisson
import csep.utils.plots as plots

# Local modules
import utils
import accessors
from config import Experiment
""" The main experiment code will go here. 
    
    Overall experiment steps:
        0. Setup directory structure for run-time 
            - Forecasts folder
            - Results folder
            - Observations folder
            - README.md goes in top-level 
            - Use test datetime for folder name

        1. Retrieve data from online repository (Zenodo and ISC)

            - Use experiment config class to determine the filepath of these models. If not found download, else skip
              downloading.
            - Download gCMT catalog from ISC. If catalog is found (the run is being re-created), then skip downloading and
              filtering.
                - Filter catalog according to experiment definition
                - Write ASCII version of catalog 
                - Update experiment manifest with filepath
            
        2. Prepare forecast files from models
            
            - Using same logic, only prepare these files if they are not found locally (ie, new run of the experiment)
            - The catalogs should be filtered to the same time horizon and region from experiment 

        3. Evaluate forecasts

            - Run the evaluations using the information stored in the experiment config
            - Update experiment class with information from evaluation runs

        4. Clean-up steps
            
            - Prepare Markdown report using the Experiment class 
            - Commit run results to GitLab (if running from authorized user)
"""



if __name__ == '__main__':
    ### Create the experiment configuration parameters
    dh = 2
    mag_bins = cleaner_range(5.95, 8.95, 0.1)
    region = utils.global_region(dh, magnitudes=mag_bins)
    start_date = datetime.datetime(2020, 1, 1, 0, 0, 0)
    test_date = datetime.datetime(2021, 1, 1, 0, 0, 0)
    end_date = datetime.datetime(2022, 1, 1, 0, 0, 0)

    ### Initialize
    exp = Experiment(start_date, end_date, region, accessors.query_isc_gcmt)

    ### Set the tests
    exp.set_test('Poisson_CL', poisson.conditional_likelihood_test,
                 {'num_simulations': 10, 'seed': 23}, plots.plot_poisson_consistency_test)
    exp.set_test('Poisson_N', poisson.conditional_likelihood_test,
                 {'num_simulations': 10, 'seed': 23}, plots.plot_poisson_consistency_test)
    exp.set_test('Poisson_T', poisson.paired_t_test,
                 {'num_simulations': 10, 'seed': 23}, plots.plot_comparison_test, ref_model='GEAR1')

    ### Set the models
    exp.set_model('GEAR1', utils.prepare_forecast, {'model_path': '../models/GEAR_resampled.txt', 'dh': dh})
    exp.set_model('KJSS', utils.prepare_forecast, {'model_path': '../models/KJSS_resampled.txt', 'dh': dh})
    exp.set_model('SHIFT2F_GSRM', utils.prepare_forecast, {'model_path': '../models/WHEELr_resampled.txt', 'dh': dh})
    exp.set_model('WHEELr', utils.prepare_forecast, {'model_path': '../models/WHEELr_resampled.txt', 'dh': dh})
    exp.set_model('TEAMr', utils.prepare_forecast, {'model_path': '../models/WHEELr_resampled.txt', 'dh': dh})

    ### Run the experiment
    exp.run(test_date, new_run=False)