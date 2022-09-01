# Python modules
import datetime
from matplotlib import pyplot

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

pyplot.rcParams.update({
    'axes.labelweight': 'bold',
    'axes.titlesize': 20
})

# Experiment configuration below here:
######################################

expected_models = {
    'WHEEL=N10L11': './models/WHEEL=N10L11.csv',
    'GEAR1=N10L11': './models/GEAR1=N10L11.csv',
    'KJSS=N10L11': './models/KJSS=N10L11.csv',
    'TEAM=N10L11': './models/TEAM=N10L11.csv',
    'SHIFT2F_GSRM=N10L11': './models/SHIFT2F_GSRM=N10L11.csv',
    'WHEEL=N25L11': './models/WHEEL=N25L11.csv',
    'GEAR1=N25L11': './models/GEAR1=N25L11.csv',
    'KJSS=N25L11': './models/KJSS=N25L11.csv',
    'TEAM=N25L11': './models/TEAM=N25L11.csv',
    'SHIFT2F_GSRM=N25L11': './models/SHIFT2F_GSRM=N25L11.csv',
    'WHEEL=N50L11': './models/WHEEL=N50L11.csv',
    'GEAR1=N50L11': './models/GEAR1=N50L11.csv',
    'KJSS=N50L11': './models/KJSS=N50L11.csv',
    'TEAM=N50L11': './models/TEAM=N50L11.csv',
    'SHIFT2F_GSRM=N50L11': './models/SHIFT2F_GSRM=N50L11.csv',
    'WHEEL=N100L11': './models/WHEEL=N100L11.csv',
    'GEAR1=N100L11': './models/GEAR1=N100L11.csv',
    'KJSS=N100L11': './models/KJSS=N100L11.csv',
    'TEAM=N100L11': './models/TEAM=N100L11.csv',
    'SHIFT2F_GSRM=N100L11': './models/SHIFT2F_GSRM=N100L11.csv',
    'WHEEL=SN10L11': './models/WHEEL=SN10L11.csv',
    'GEAR1=SN10L11': './models/GEAR1=SN10L11.csv',
    'KJSS=SN10L11': './models/KJSS=SN10L11.csv',
    'TEAM=SN10L11': './models/TEAM=SN10L11.csv',
    'SHIFT2F_GSRM=SN10L11': './models/SHIFT2F_GSRM=SN10L11.csv',
    'WHEEL=SN25L11': './models/WHEEL=SN25L11.csv',
    'GEAR1=SN25L11': './models/GEAR1=SN25L11.csv',
    'KJSS=SN25L11': './models/KJSS=SN25L11.csv',
    'TEAM=SN25L11': './models/TEAM=SN25L11.csv',
    'SHIFT2F_GSRM=SN25L11': './models/SHIFT2F_GSRM=SN25L11.csv',
    'WHEEL=SN50L11': './models/WHEEL=SN50L11.csv',
    'GEAR1=SN50L11': './models/GEAR1=SN50L11.csv',
    'KJSS=SN50L11': './models/KJSS=SN50L11.csv',
    'TEAM=SN50L11': './models/TEAM=SN50L11.csv',
    'SHIFT2F_GSRM=SN50L11': './models/SHIFT2F_GSRM=SN50L11.csv',
    'WHEEL=SN100L11': './models/WHEEL=SN100L11.csv',
    'GEAR1=SN100L11': './models/GEAR1=SN100L11.csv',
    'KJSS=SN100L11': './models/KJSS=SN100L11.csv',
    'TEAM=SN100L11': './models/TEAM=SN100L11.csv',
    'SHIFT2F_GSRM=SN100L11': './models/SHIFT2F_GSRM=SN100L11.csv',
}

# Create the experiment configuration parameters
start_date = datetime.datetime(2014, 1, 1, 0, 0, 0)
end_date = datetime.datetime(2022, 1, 1, 0, 0, 0)

default_test_kwargs = {
    'seed': 23,
    'num_simulations': 10000
}

# Initialize
exp = None
# exp = Experiment(start_date, end_date, name='Global Earthquake Forecasting Experiment -- Quadtree')
# exp.set_catalog_reader(query_isc_gcmt)
# exp.set_magnitude_range(5.95, 8.95, 0.1)
# exp.set_depth_range(0, 70)
# # Set the tests
# # todo: finish markdown template strings for each test
# exp.set_tests([
#     Test(
#         name='Poisson_N',
#         func=poisson.number_test,
#         func_kwargs={},
#         plot_func=plots.plot_poisson_consistency_test,
#         plot_args={'title': r'$N-$test',
#                    'title_fontsize': 18,
#                    'figsize': (5, 8),
#                    'xlabel': 'Number of events',
#                    'linewidth': 0.7,
#                    'capsize': 2},
#         plot_kwargs={'normalize': True},
#         markdown=f'The results of N-test from {start_date} to {end_date}. '
#                  f'The test shows whether the number of observed earthquakes forecasted is consistent with the observed events. '
#                  f'The (green) boxes inside the confidence interval indicate the models passing N-Test.'
#     ),
#     Test(
#         name='Poisson_M',
#         func=poisson.magnitude_test,
#         func_kwargs=default_test_kwargs,
#         plot_func=plots.plot_poisson_consistency_test,
#         plot_args={'title': r'$M-$test',
#                    'title_fontsize': 18,
#                    'figsize': (5, 8),
#                    'xlabel': 'Log-likelihood',
#                    'linewidth': 0.7, 'capsize': 2},
#         plot_kwargs={'normalize': True,
#                      'one_sided_lower': True},
#         markdown=f'The results of M-test from {start_date} to {end_date}. '
#                  f'The test evaluates the magnitude distribution of the forecasts. '
#                  f'The (green) boxes inside the confidence interval indicate the models passing M-Test.'
#     ),
#     Test(
#         name='Poisson_S',
#         func=poisson.spatial_test,
#         func_kwargs=default_test_kwargs,
#         plot_func=plots.plot_poisson_consistency_test,
#         plot_args={'title': r'$S-$test',
#                    'title_fontsize': 18,
#                    'figsize': (5, 8),
#                    'xlabel': 'Log-likelihood',
#                    'linewidth': 0.7,
#                    'capsize': 2},
#         plot_kwargs={'normalize': True,
#                      'one_sided_lower': True},
#         markdown=f'The results of S-test from {start_date} to {end_date}. '
#                  f'The test evaluates the spatial distribution of the forecasts. '
#                  f'The (red) boxes lagging behind the confidence interval indicate the models failing to pass S-Test.'
#     ),
#     Test(name='Poisson_CL',
#          func=poisson.conditional_likelihood_test,
#          func_kwargs=default_test_kwargs,
#          plot_func=plots.plot_poisson_consistency_test,
#          plot_args={'title': r'$L_{n}-$test',
#                     'title_fontsize': 18,
#                     'figsize': (5, 8),
#                     'xlabel': 'Log-likelihood',
#                     'linewidth': 0.7,
#                     'capsize': 2},
#          plot_kwargs={'normalize': True,
#                       'one_sided_lower': True},
#          markdown=f'The results of CL-test from {start_date} to {end_date}. '
#                   f'The test simultaneously evaluates the spatial and magnitude distribution of the forecasts. '
#                   f'The (red) boxes lagging behind the confidence interval indicate the models failing to pass CL-Test.'
#          ),
#     Test(name='Poisson_T',
#          func=evaluations.paired_ttest_point_process,
#          func_kwargs={},
#          plot_func=plots.plot_comparison_test,
#          ref_model='GEAR1=SN50L11',
#          plot_args={'title': f'$T-$test',
#                     'title_fontsize': 18,
#                     'figsize': (8, 6),
#                     'ylabel': 'Information Gain per Earthquake',
#                     'ylabel_fontsize': 10,
#                     'xlabel': '',
#                     'linewidth': 1.2,
#                     'capsize': 2,
#                     'markersize': 3},
#          markdown=f'The results of comparitive T-test from {start_date} to {end_date} GEAR1 at Grid:SN50L11 as the benchmark.'
#                   f' The mean information gain per earthquake as is shown by circles, and the 95 percent confidence interval with vertical lines.'
#                   f' The models with information gain higher than zero are more informative than the benchmark model.'
#     )
# ])
#
# # Set the models
# exp.set_models(
#     [Model(name=name, path=path, func=quadtree_csv_loader, func_args=None) for name, path in expected_models.items()]
# )
