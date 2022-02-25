# python libraries
import datetime
import numpy
import pandas
import os
import pickle


# pyCSEP libraries
from csep.utils.calc import cleaner_range
import csep.core.poisson_evaluations as poisson
import csep.utils.plots as plots
from csep.core.forecasts import GriddedForecast
from csep.core.regions import QuadtreeGrid2D
from csep.core.catalogs import CSEPCatalog
from csep.core.poisson_evaluations import spatial_test, number_test, conditional_likelihood_test, magnitude_test, w_test
from csep.utils.plots import plot_poisson_consistency_test, plot_comparison_test
from csep.utils.time_utils import decimal_year_to_utc_epoch
from csep.utils.constants import DAYS_PER_ASTRONOMICAL_YEAR
from csep.utils.basic_types import transpose_dict


# Local modules
import utils
import accessors
from config import Experiment
from evaluations import paired_ttest_point_process
from utils import quadtree_csv_loader


""" The main experiment code will go here. 

    Overall experiment steps:
        0. Setup directory structure for run-time 
            - Forecasts folder
            - Results folder
            - Observations folder
            - README.md goes in top-level 
            - Use test datetime for folder name

#For now just setup a simple directory structure inside the code folder 
#Improve later
"""

start_time = datetime.datetime(2014, 1, 1, 0, 0, 0)
end_time = datetime.datetime(2020, 1, 1, 0, 0, 0)
# start_epoch = datetime_to_utc_epoch(start_time)
# end_epoch = datetime_to_utc_epoch(end_time)
experiment_years = (end_time - start_time).days / DAYS_PER_ASTRONOMICAL_YEAR

# mbins = numpy.array([5.95])
# forecast_name = 'forecasts/helmstetter/helmstetter_' #, #'forecasts/wheel/wheel_' #'forecasts/uniform/uniform_'
catalog_name = 'catalog/cat_test.csv'

# list of forecasts to run
fore_fnames = {'GEAR1': 'global_quadtree_forecasts/GEAR1=', 'TEAM': 'global_quadtree_forecasts/TEAM=', 'WHEEL': 'global_quadtree_forecasts/WHEEL=',
               'SHIFT2F_GSRM': 'global_quadtree_forecasts/SHIFT2F_GSRM=', 'KJSS': 'global_quadtree_forecasts/KJSS='}

grid_names = {'N25L11': 'N25L11.csv', 'N10L11': 'N10L11.csv', 'SN10L11': 'SN10L11.csv', 'SN25L11': 'SN25L11.csv',

              }

# definitions of evaluations for forecast
evaluations = {'N-Test': {'func': number_test, 'type': 'consistency'}, 'S-Test': {'func': spatial_test, 'type': 'consistency'},
               'M-Test': {'func': magnitude_test, 'type': 'consistency'}, 'CL-Test': {'func': conditional_likelihood_test, 'type': 'consistency'},
               'T-Test': {'func': paired_ttest_point_process, 'type': 'comparative'}, 'W-Test': {'func': w_test, 'type': 'comparative'}}

experiment_tag = 'quadtree_global_experiment'

# comparative tests will be performed against this forecast
benchmark_forecast_name = 'global_quadtree_forecasts/GEAR1=N25L11.csv'
benchmark_short_name = 'GEAR1_N25L11'
# forecasts are provided as rates per m**2
area_fname = '../forecasts/area.dat'

# switch for storing results (can pickle or serialize result classes)
store_results = True

# Experiment name
experiment_time = datetime.datetime.today()
result_dir = 'results/'
# results_tag = 'results_quadtree_global_'+experiment_time.strftime("%Y,%m,%d")
# os.mkdir('results/'+results_tag)

# should results be plotted
plot_results = True
figure_dirname = ''

"""
        1. Retrieve data from online repository (Zenodo and ISC)

            - Use experiment config class to determine the filepath of these models. If not found download, else skip
              downloading.
            - Download gCMT catalog from ISC. If catalog is found (the run is being re-created), then skip downloading and
              filtering.
                - Filter catalog according to experiment definition
                - Write ASCII version of catalog 
                - Update experiment manifest with filepath

"""
# For now just using the available catalog in the subdirectorie.
# Improve later

# ----Read Catalog format
dfcat = pandas.read_csv(catalog_name)

column_name_mapper = {'lon': 'longitude', 'lat': 'latitude', 'mag': 'magnitude'}  # 'index': 'id'

# maps the column names to the dtype expected by the catalog class
dfcat = dfcat.reset_index().rename(columns=column_name_mapper)
# create the origin_times from decimal years
dfcat['origin_time'] = dfcat.apply(lambda row: decimal_year_to_utc_epoch(row.year), axis=1)  # ----
# add depth
# df['depth'] = 5
# create catalog from dataframe
catalog = CSEPCatalog.from_dataframe(dfcat)
print(catalog)

mw_min = 5.95
mw_max = 8.95
dmw = 0.1
mws = numpy.arange(mw_min, mw_max + dmw / 2, dmw)
catalog.magnitudes = mws  # We need this to filter catalog, just in case catalog is not filtered for magnitues.

"""           
        2. Prepare forecast files from models

            - Using same logic, only prepare these files if they are not found locally (ie, new run of the experiment)
            - The catalogs should be filtered to the same time horizon and region from experiment 

#For now simply directly read the forecasts from the folder
"""
results = {}
for grid_name, grid_fname in grid_names.items():
    # print(f"...working on {grid_name} available at {grid_fname}...")

    for fore_name, fore_fname in fore_fnames.items():
        print(f"...WORKING ON {fore_name}_{grid_name}...")
        # t0 = time.time()
        # class-level api to read forecast with custom format
        filename = fore_fname + grid_fname
        rates, qr, mbins = quadtree_csv_loader(filename)
        qr.get_cell_area()
        # Loading quadtree forecast into pyCSEP

        # rates = numpy.sum(rates, axis=1)
        # rates = rates.reshape(-1,1)
        # mmm = [5.95]
        forecast = GriddedForecast(data=rates, region=qr, magnitudes=mbins, name=fore_name + '_' + grid_name)
        # bind region from forecast to catalog
        catalog.region = forecast.region
        print(f"expected event count before scaling: {forecast.event_count}")
        # scale time-independent forecast to length of catalog
        forecast.scale(experiment_years)
        print(f"expected event count after scaling: {forecast.event_count}")
        # t1 = time.time()
        gridded_cat = catalog.spatial_counts()
        # makes sure we aren't missing any events
        assert catalog.event_count == numpy.sum(gridded_cat)
        # print(f"prepared {name} forecast in {t1-t0} seconds.")
        # store forecast for later use, need to handle this case
        if filename == benchmark_forecast_name:  # Keep the first file in the list as Benchmark
            benchmark_forecast = forecast  # print('benchmark forecast:', benchmark_forecast_name)
        # compute number_test
        print("")
        # print(f"...EVALUATING {fore_name}_{grid_name} FORECAST...")
        eval_results = {}
        for eval_name, eval_config in evaluations.items():
            print(f"Running {eval_name}")
            # t2 = time.time()
            # this could be simplified by having an Evaluation class.
            if eval_config['type'] == 'consistency':
                print(f"computing {eval_name}")
                # -----cltest = conditional_likelihood_test(forecast, catalog)
                eval_results[eval_name] = eval_config['func'](forecast, catalog)  # , seed = 123456  # t3 = time.time()
            elif eval_config['type'] == 'comparative':
                if filename == benchmark_forecast_name:
                    # print(f"skipping comparative testing with forecast {fore_name}_{grid_name}, because its the benchmark.")
                    break
                # handle comparison test
                eval_results[eval_name] = eval_config['func'](forecast, benchmark_forecast,
                                                              catalog)  # t3 = time.time()  # print(f"finished {eval_name} in {t3-t2} seconds.")
        results[fore_name + '_' + grid_name] = eval_results
        print("")

if store_results:
    with open(result_dir + experiment_tag + '.dat', 'wb') as f:
        pickle.dump(transpose_dict(results), f)

if plot_results:
    with open(result_dir + experiment_tag + '.dat', 'rb') as f:
        reordered_results = pickle.load(f)

    # Plot N-Test
    n_test_results = list(reordered_results['N-Test'].values())
    axn = plot_poisson_consistency_test(n_test_results, plot_args={'xlabel': 'Number of Earthquakes', 'title': 'N-Test'})
    axn.figure.tight_layout()
    axn.figure.savefig(result_dir + experiment_tag + 'N-Test.png')

    # Plot M-Test
    m_test_results = list(reordered_results['M-Test'].values())
    axm = plot_poisson_consistency_test(m_test_results, plot_args={'xlabel': 'M Simulated - M Observed', 'title': 'M-Test'}, normalize=True,
                                        one_sided_lower=True)
    axn.figure.tight_layout()
    axm.figure.savefig(result_dir + experiment_tag + 'M-Test.png')

    # Plot S-Test
    s_test_results = list(reordered_results['S-Test'].values())
    axs = plot_poisson_consistency_test(s_test_results, plot_args={'xlabel': 'S Simulated - S Observed', 'title': 'S-Test'}, normalize=True,
                                        one_sided_lower=True)
    axs.figure.tight_layout()
    axs.figure.savefig(result_dir + experiment_tag + 'S-Test.png')

    # Plot CL-Test
    cl_test_results = list(reordered_results['CL-Test'].values())
    axcl = plot_poisson_consistency_test(cl_test_results, plot_args={'xlabel': 'CL Simulated - CL Observed', 'title': 'CL-Test'}, normalize=True,
                                         one_sided_lower=True)
    axcl.figure.tight_layout()
    axcl.figure.savefig(result_dir + experiment_tag + 'CL-Test.png')

    # Plot T-Test
    t_test_results = list(reordered_results['T-Test'].values())
    ax = plot_comparison_test(t_test_results, plot_args={'title': 'T-Test', 'xlabel': 'Model', 'ylabel': 'Information Gain(Forecast - ' + benchmark_short_name})
    ax.figure.tight_layout()
    ax.figure.savefig(result_dir + experiment_tag + 'T-Test.png')

"""
        3. Evaluate forecasts

            - Run the evaluations using the information stored in the experiment config
            - Update experiment class with information from evaluation runs
"""

"""
        4. Clean-up steps

            - Prepare Markdown report using the Experiment class 
            - Commit run results to GitLab (if running from authorized user)
"""



# if __name__ == '__main__':
#     ### Create the experiment configuration parameters
#     dh = 2
#     mag_bins = cleaner_range(5.95, 8.95, 0.1)
#     region = utils.global_region(dh, magnitudes=mag_bins)
#     start_date = datetime.datetime(2020, 1, 1, 0, 0, 0)
#     test_date = datetime.datetime(2021, 1, 1, 0, 0, 0)
#     end_date = datetime.datetime(2022, 1, 1, 0, 0, 0)
#
#     ### Initialize
#     exp = Experiment(start_date, end_date, region, accessors.query_isc_gcmt)
#
#     ### Set the tests
#     exp.set_test('Poisson_CL', poisson.conditional_likelihood_test,
#                  {'num_simulations': 10, 'seed': 23}, plots.plot_poisson_consistency_test)
#     exp.set_test('Poisson_N', poisson.conditional_likelihood_test,
#                  {'num_simulations': 10, 'seed': 23}, plots.plot_poisson_consistency_test)
#     exp.set_test('Poisson_T', poisson.paired_t_test,
#                  {'num_simulations': 10, 'seed': 23}, plots.plot_comparison_test, ref_model='GEAR1')
#
#     ### Set the models
#     exp.set_model('GEAR1', utils.prepare_forecast, {'model_path': '../models/GEAR_resampled.txt', 'dh': dh})
#     exp.set_model('KJSS', utils.prepare_forecast, {'model_path': '../models/KJSS_resampled.txt', 'dh': dh})
#     exp.set_model('SHIFT2F_GSRM', utils.prepare_forecast, {'model_path': '../models/WHEELr_resampled.txt', 'dh': dh})
#     exp.set_model('WHEELr', utils.prepare_forecast, {'model_path': '../models/WHEELr_resampled.txt', 'dh': dh})
#     exp.set_model('TEAMr', utils.prepare_forecast, {'model_path': '../models/WHEELr_resampled.txt', 'dh': dh})
#
#     ### Run the experiment
#     exp.run(test_date, new_run=False)