###- This file is created from main.py. 
###- The purpose of this file is to create basic working of CSEP tests for Quadtree forecasts
###- For now, the data is being used from Repo.
import numpy
import pandas
from csep.core.regions import QuadtreeGrid2D
from csep.core.catalogs import CSEPCatalog
from csep.core.poisson_evaluations import spatial_test, number_test, conditional_likelihood_test, magnitude_test, w_test
from csep.utils.plots import plot_poisson_consistency_test
from csep.core.catalogs import CSEPCatalog
from csep.utils.time_utils import decimal_year_to_utc_epoch, datetime_to_utc_epoch
from csep.utils.constants import DAYS_PER_ASTRONOMICAL_YEAR
from csep.core.forecasts import GriddedForecast
import datetime
from tests import paired_ttest_point_process
import os


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

start_time = datetime.datetime(2014,1,1,0,0,0)
end_time = datetime.datetime(2020,1,1,0,0,0)
#start_epoch = datetime_to_utc_epoch(start_time)
#end_epoch = datetime_to_utc_epoch(end_time)
experiment_years = (end_time-start_time).days / DAYS_PER_ASTRONOMICAL_YEAR

#mbins = numpy.array([5.95])
#forecast_name = 'forecasts/helmstetter/helmstetter_' #, #'forecasts/wheel/wheel_' #'forecasts/uniform/uniform_' 
catalog_name = 'catalog/cat_test.csv'


# list of forecasts to run
fore_fnames = {'GEAR1': '/global_quadtree_forecasts/GEAR1=',
               'TEAM': '/global_quadtree_forecasts/TEAM=',
               'WHEEL': '/global_quadtree_forecasts/WHEEL=',
               'SHIFT2F_GSRM': '/global_quadtree_forecasts/SHIFT2F_GSRM=',
               'KJSS': '/forecasts/KJSS='}


grid_names = {'N50L11': 'N50L11.csv',
              'N10L11': 'N10L11.csv',
              'SN50L11': 'SN50L11.csv',
              'SN10L11': 'SN10L11.csv'}


# definitions of evaluations for forecast
evaluations = {'N-Test': {'func': number_test, 'type': 'consistency'},
               'S-Test': {'func': spatial_test, 'type': 'consistency'},
               'M-Test': {'func': magnitude_test, 'type': 'consistency'},
               'CL-Test': {'func': conditional_likelihood_test, 'type': 'consistency'},
               'T-Test': {'func': paired_ttest_point_process, 'type': 'comparative'},
               'W-Test': {'func': w_test, 'type': 'comparative'}}

experiment_tag = 'quadtree_global_experiment'

# comparative tests will be performed against this forecast
benchmark_forecast_name = 'global_quadtree_forecasts/GEAR1=N10L11.csv'

# forecasts are provided as rates per m**2
area_fname = '../forecasts/area.dat'

# switch for storing results (can pickle or serialize result classes)
store_results = True

#Experiment name
experiment_time = datetime.datetime.today()
results_tag = 'results_quadtree_global_'+experiment_time.strftime("%Y,%m,%d")
os.mkdir('results/'+results_tag)

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
#For now just using the available catalog in the subdirectorie. 
#Improve later  

#----Read Catalog format
dfcat = pandas.read_csv(catalog_name)

column_name_mapper = {
    'lon': 'longitude',
    'lat': 'latitude',
    'mag': 'magnitude'
    }  #'index': 'id'

# maps the column names to the dtype expected by the catalog class
dfcat = dfcat.reset_index().rename(columns=column_name_mapper)
# create the origin_times from decimal years
dfcat['origin_time'] = dfcat.apply(lambda row: decimal_year_to_utc_epoch(row.year), axis=1) #----
# add depth
#df['depth'] = 5
# create catalog from dataframe
catalog = CSEPCatalog.from_dataframe(dfcat)
print(catalog)




"""           
        2. Prepare forecast files from models
            
            - Using same logic, only prepare these files if they are not found locally (ie, new run of the experiment)
            - The catalogs should be filtered to the same time horizon and region from experiment 

#For now simply directly read the forecasts from the folder
"""




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

