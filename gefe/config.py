# Python modules
import datetime

# pyCSEP modules
import csep.core.poisson_evaluations as poisson
import csep.utils.plots as plots

# Local modules
import gefe.evaluations as evaluations
from gefe.accessors import query_isc_gcmt
from gefe.utils import quadtree_csv_loader
from gefe.models import (
    Experiment,
    Test,
    Model
)

# Experiment configuration below here:
######################################

expected_forecasts = {
    'WHEEL=SN25L11': './models/WHEEL=SN25L11.csv',
    'WHEEL=N50L11': './models/WHEEL=N50L11.csv',
    'GEAR1=N25L11': './models/GEAR1=N25L11.csv',
    'KJSS=N10L11': './models/KJSS=N10L11.csv',
    'TEAM=SN25L11': './models/TEAM=SN25L11.csv',
    'KJSS=SN100L11': './models/KJSS=SN100L11.csv',
    'GEAR1=SN25L11': './models/GEAR1=SN25L11.csv',
    'SHIFT2F_GSRM=N50L11': './models/SHIFT2F_GSRM=N50L11.csv',
    'TEAM=N10L11': './models/TEAM=N10L11.csv',
    'TEAM=SN50L11': './models/TEAM=SN50L11.csv',
    'GEAR1=SN50L11': './models/GEAR1=SN50L11.csv',
    'TEAM=N100L11': './models/TEAM=N100L11.csv',
    'SHIFT2F_GSRM=N25L11': './models/SHIFT2F_GSRM=N25L11.csv',
    'GEAR1=N100L11': './models/GEAR1=N100L11.csv',
    'SHIFT2F_GSRM=SN10L11': './models/SHIFT2F_GSRM=SN10L11.csv',
    'WHEEL=SN50L11': './models/WHEEL=SN50L11.csv',
    'KJSS=SN10L11': './models/KJSS=SN10L11.csv',
    'WHEEL=N100L11': './models/WHEEL=N100L11.csv',
    'WHEEL=N25L11': './models/WHEEL=N25L11.csv',
    'GEAR1=N50L11': './models/GEAR1=N50L11.csv',
    'KJSS=N25L11': './models/KJSS=N25L11.csv',
    'TEAM=SN10L11': './models/TEAM=SN10L11.csv',
    'GEAR1=SN10L11': './models/GEAR1=SN10L11.csv',
    'SHIFT2F_GSRM=N100L11': './models/SHIFT2F_GSRM=N100L11.csv',
    'TEAM=N50L11': './models/TEAM=N50L11.csv',
    'WHEEL=SN100L11': './models/WHEEL=SN100L11.csv',
    'SHIFT2F_GSRM=SN50L11': './models/SHIFT2F_GSRM=SN50L11.csv',
    'KJSS=N100L11': './models/KJSS=N100L11.csv',
    'WHEEL=SN10L11': './models/WHEEL=SN10L11.csv',
    'GEAR1=N10L11': './models/GEAR1=N10L11.csv',
    'KJSS=SN50L11': './models/KJSS=SN50L11.csv',
    'TEAM=SN100L11': './models/TEAM=SN100L11.csv',
    'KJSS=SN25L11': './models/KJSS=SN25L11.csv',
    'SHIFT2F_GSRM=SN100L11': './models/SHIFT2F_GSRM=SN100L11.csv',
    'WHEEL=N10L11': './models/WHEEL=N10L11.csv',
    'GEAR1=SN100L11': './models/GEAR1=SN100L11.csv',
    'TEAM=N25L11': './models/TEAM=N25L11.csv',
    'KJSS=N50L11': './models/KJSS=N50L11.csv',
    'SHIFT2F_GSRM=N10L11': './models/SHIFT2F_GSRM=N10L11.csv',
    'SHIFT2F_GSRM=SN25L11': './models/SHIFT2F_GSRM=SN25L11.csv'
}

# Create the experiment configuration parameters
start_date = datetime.datetime(2020, 1, 1, 0, 0, 0)
end_date = datetime.datetime(2022, 1, 1, 0, 0, 0)

default_test_kwargs = {
    'seed': 23,
    'num_simulations': 1000
}

# Initialize
exp = Experiment(start_date, end_date, name='Global Earthquake Forecasting Experiment -- Quadtree')
exp.set_catalog_reader(query_isc_gcmt)
exp.set_magnitude_range(5.95, 8.95, 0.1)

# Set the tests
# todo: add default plotting arguments to call normalize=True for consistency tests
# todo: we don't want to call show in plots
exp.set_tests([
    Test(
        name='Poisson_N',
        func=poisson.number_test,
        func_kwargs={},
        plot_func=plots.plot_poisson_consistency_test
    ),
    Test(
        name='Poisson_M',
        func=poisson.magnitude_test,
        func_kwargs=default_test_kwargs,
        plot_func=plots.plot_poisson_consistency_test
    ),
    Test(
        name='Poisson_S',
        func=poisson.spatial_test,
        func_kwargs=default_test_kwargs,
        plot_func=plots.plot_poisson_consistency_test
    ),
    Test(name='Poisson_CL',
         func=poisson.conditional_likelihood_test,
         func_kwargs=default_test_kwargs,
         plot_func=plots.plot_poisson_consistency_test
    ),
    Test(name='Poisson_T',
         func=evaluations.paired_ttest_point_process,
         func_kwargs={},
         plot_func=plots.plot_comparison_test,
         ref_model='GEAR1=N25L11'
    )
])

# Set the models
exp.set_models(
    [Model(name=name,
          path=path,
          func=quadtree_csv_loader,
          func_args=None)
    for name, path in expected_forecasts.items()]
)
