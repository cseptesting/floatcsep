import numpy
from csep.core.regions import QuadtreeGrid2D
from csep.core.catalogs import CSEPCatalog
from csep.models import EvaluationResult
import scipy


def matrix_poisson_t_test(forecasts, catalog, **kwargs):
    """ Computes Student's t-test for the information gain per earthquake over a list of forecasts
        
        Uses unique combinations of individual forecasts to perform pair-wise t-tests against all forecasts 
        provided to the function. 

        Args:
            forecasts (list of csep.GriddedForecast): list of forecasts to evaluate
            catalog (csep.AbstractBaseCatalog): evaluation catalog filtered consistent with forecast
            **kwargs: additional default arguments

        Returns:
            results (list of csep.EvaluationResult): iterable of evaluation results
    """

def negative_binomial_number_test(forecast, catalog):
    pass


def binomial_conditional_likelihood_test(forecast, catalog, **kwargs):
    pass


def binomial_conditional_likelihood_test(forecast, catalog, **kwargs):
    pass


def binomial_spatial_test(forecast, catalog, **kwargs):
    pass


def matrix_binary_t_test(forecasts, catalog, **kwargs):
    pass


def _log_likelihood_point_process(observation, forecast, cell_area):
    """
    Log-likelihood for point process
    Args:
        observation: Gridded observation data
        forecast: gridded forecast data
        cell_area: Cell area of each grid cell
    Returns:
        log-likelihood for point process

    """

    obs = observation[observation > 0]
    fcst = forecast[observation > 0]
    ca = cell_area[observation > 0]

    rate_density = []
    for i in range(len(obs)):
        #            print("i = :",i)
        #            print("Obs[i] = :", obs[i])
        counter = obs[i]
        while counter > 0:
            rate_density = numpy.append(rate_density, fcst[i] / ca[i])
            counter = counter - 1
    ll = numpy.sum(numpy.log(rate_density)) - sum(forecast)
    return ll[0]  # To get a scalar value instead of array


def _standard_deviation(gridded_forecast1, gridded_forecast2, gridded_observation1, gridded_observation2,
                              cell_area1, cell_area2):
    """
    Calculate Variance using forecast 1 and forecast 2. 
    But It is calculated using the forecast values corresponding to the non-zero observations.
    Equation is adapted from Rhoades et al. 2011


    Args:
        gridded_forecast1 : forecast
        gridded_forecast2 : benchmark_forecast
        gridded_observation1 : observation
        gridded_observation2 : observation according to benchmark

    Returns:
        Variance

    """

    N_obs = sum(gridded_observation1)
    obs1 = gridded_observation1[gridded_observation1 > 0]
    obs2 = gridded_observation2[gridded_observation2 > 0]

    fore1 = gridded_forecast1[gridded_observation1 > 0]
    fore2 = gridded_forecast2[gridded_observation2 > 0]

    ca1 = cell_area1[gridded_observation1 > 0]
    ca2 = cell_area2[gridded_observation2 > 0]

    target_fore1 = []
    target_fore2 = []

    for i in range(len(obs1)):
        #            print("i = :",i)
        #            print("Obs[i] = :", obs[i])
        counter = obs1[i]
        while counter > 0:
            target_fore1 = numpy.append(target_fore1, fore1[i] / ca1[i])
            counter = counter - 1

    for i in range(len(obs2)):
        #            print("i = :",i)
        #            print("Obs[i] = :", obs[i])
        counter = obs2[i]
        while counter > 0:
            target_fore2 = numpy.append(target_fore2, fore2[i] / ca2[i])
            counter = counter - 1

    # print('Length of Fore 1: ', len(target_fore1))
    # print('Length of Fore 2: ', len(target_fore2))

    X1 = numpy.log(target_fore1)
    X2 = numpy.log(target_fore2)
    first_term = (numpy.sum(numpy.power((X1 - X2), 2))) / (N_obs - 1)
    second_term = numpy.power(numpy.sum(X1 - X2), 2) / (numpy.power(N_obs, 2) - N_obs)
    forecast_variance = first_term - second_term


    return forecast_variance


# Function for T test based on Point process LL
def paired_ttest_point_process(forecast, benchmark_forecast, observed_catalog, alpha=0.05):
    """
    Function for T test based on Point process LL.
    Works for comparing forecasts for different grids

    Args:    
        forecast (csep.core.forecasts.GriddedForecast): nd-array storing gridded rates, axis=-1 should be the magnitude column
        benchmark_forecast (csep.core.forecasts.GriddedForecast): nd-array storing gridded rates, axis=-1 should be the magnitude column
        observed_catalog (csep.core.catalogs.AbstractBaseCatalog): number of observed earthquakes, should be whole number and >= zero.
        alpha (float): tolerance level for the type-i error rate of the statistical test
        scale (bool): if true, scale forecasted rates down to a single day
    Returns:
        evaluation_result: csep.core.evaluations.EvaluationResult
    """
    # Treat every forecast separately to calculate LL and then take diff of LL
    # Forecast 1
    gridded_forecast1 = forecast.data
    gridded_observation1 = forecast.region._get_spatial_counts(observed_catalog)
    if forecast.region.cell_area == []: #If region object has not computed cell area already.
        forecast.region.get_cell_area()
    cell_area1 = forecast.region.cell_area
    ll1 = _log_likelihood_point_process(gridded_observation1, gridded_forecast1, cell_area1)
    # print('This is LL1 :', ll1)

    # Forecast 2
    gridded_forecast2 = benchmark_forecast.data
    gridded_observation2 = benchmark_forecast.region._get_spatial_counts(observed_catalog)
    cell_area2 = benchmark_forecast.region.cell_area
    ll2 = _log_likelihood_point_process(gridded_observation2, gridded_forecast2, cell_area2)
    # print('This is LL2 :', ll2)
    assert sum(gridded_observation1) == sum(gridded_observation2), 'Sum of Gridded Catalog is not same'

    # print('Forecast to Compare: ', len(gridded_forecast1))
    # print('Forecast Benchmark: ', len(gridded_forecast2))

    N_obs = sum(gridded_observation1)

    information_gain = (ll1 - ll2) / N_obs

    forecast_variance = _standard_deviation(gridded_forecast1, gridded_forecast2, gridded_observation1,
                                                  gridded_observation2, cell_area1, cell_area2)

    forecast_std = numpy.sqrt(forecast_variance)
    t_statistic = information_gain / (forecast_std / numpy.sqrt(N_obs))

    # Obtaining the Critical Value of T from T distribution.
    df = N_obs - 1
    t_critical = scipy.stats.t.ppf(1 - (alpha / 2), df)  # Assuming 2-Tail Distribution  for 2 tail, divide 0.05/2.

    # Computing Information Gain Interval.
    ig_lower = information_gain - (t_critical * forecast_std / numpy.sqrt(N_obs))
    ig_upper = information_gain + (t_critical * forecast_std / numpy.sqrt(N_obs))

    # If T value greater than T critical, Then both Lower and Upper Confidence Interval limits will be greater than Zero.
    # If above Happens, Then It means that Forecasting Model 1 is better than Forecasting Model 2.
    out = {'t_statistic': t_statistic,
           't_critical': t_critical,
           'information_gain': information_gain,
           'ig_lower': ig_lower,
           'ig_upper': ig_upper}

    result = EvaluationResult()
    result.name = 'Paired T-Test'
    result.test_distribution = (out['ig_lower'], out['ig_upper'])
    result.observed_statistic = out['information_gain']
    result.quantile = (out['t_statistic'], out['t_critical'])
    result.sim_name = (forecast.name, benchmark_forecast.name)
    result.obs_name = observed_catalog.name
    result.status = 'normal'
    result.min_mw = numpy.min(forecast.magnitudes)
    return result