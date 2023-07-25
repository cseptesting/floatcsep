import numpy
import scipy.stats
from matplotlib import pyplot
from csep.models import EvaluationResult
from csep.core.poisson_evaluations import _simulate_catalog, paired_t_test, \
    w_test, _poisson_likelihood_test
from csep.core.exceptions import CSEPCatalogException
from typing import Sequence
from csep.core.forecasts import GriddedForecast
from csep.core.catalogs import CSEPCatalog


def binomial_spatial_test(
        gridded_forecast: GriddedForecast,
        observed_catalog: CSEPCatalog,
        num_simulations=1000,
        seed=None,
        random_numbers=None,
        verbose=False):
    """
    Performs the binary spatial test on the Forecast using the Observed Catalogs.
    Note: The forecast and the observations should be scaled to the same time period before calling this function. This increases
    transparency as no assumptions are being made about the length of the forecasts. This is particularly important for
    gridded forecasts that supply their forecasts as rates.
    Args:
        gridded_forecast: csep.core.forecasts.GriddedForecast
        observed_catalog: csep.core.catalogs.Catalog
        num_simulations (int): number of simulations used to compute the quantile score
        seed (int): used fore reproducibility, and testing
        random_numbers (numpy.ndarray): random numbers used to override the random number generation. injection point for testing.
    Returns:
        evaluation_result: csep.core.evaluations.EvaluationResult
    """

    # grid catalog onto spatial grid
    gridded_catalog_data = observed_catalog.spatial_counts()

    # simply call likelihood test on catalog data and forecast
    qs, obs_ll, simulated_ll = _binomial_likelihood_test(
        gridded_forecast.spatial_counts(), gridded_catalog_data,
        num_simulations=num_simulations,
        seed=seed,
        random_numbers=random_numbers,
        use_observed_counts=True,
        verbose=verbose, normalize_likelihood=True)

    # populate result data structure
    result = EvaluationResult()
    result.test_distribution = simulated_ll
    result.name = 'Binary S-Test'
    result.observed_statistic = obs_ll
    result.quantile = qs
    result.sim_name = gridded_forecast.name
    result.obs_name = observed_catalog.name
    result.status = 'normal'
    try:
        result.min_mw = numpy.min(gridded_forecast.magnitudes)
    except AttributeError:
        result.min_mw = -1
    return result


def binary_paired_t_test(forecast: GriddedForecast,
                         benchmark_forecast: GriddedForecast,
                         observed_catalog: CSEPCatalog,
                         alpha=0.05, scale=False):
    """
    Computes the binary t-test for gridded earthquake forecasts.

    This score is positively oriented, meaning that positive values of the information gain indicate that the
    forecast is performing better than the benchmark forecast.

    Args:
        forecast (csep.core.forecasts.GriddedForecast): nd-array storing gridded rates, axis=-1 should be the magnitude column
        benchmark_forecast (csep.core.forecasts.GriddedForecast): nd-array storing gridded rates, axis=-1 should be the magnitude
        column
        observed_catalog (csep.core.catalogs.AbstractBaseCatalog): number of observed earthquakes, should be whole number and >= zero.
        alpha (float): tolerance level for the type-i error rate of the statistical test
        scale (bool): if true, scale forecasted rates down to a single day

    Returns:
        evaluation_result: csep.core.evaluations.EvaluationResult
    """

    # needs some pre-processing to put the forecasts in the context that is required for the t-test. this is different
    # for cumulative forecasts (eg, multiple time-horizons) and static file-based forecasts.
    target_event_rate_forecast1p, n_fore1 = forecast.target_event_rates(
        observed_catalog, scale=scale)
    target_event_rate_forecast2p, n_fore2 = benchmark_forecast.target_event_rates(
        observed_catalog, scale=scale)

    target_event_rate_forecast1 = forecast.data.ravel()[
        numpy.unique(numpy.nonzero(
            observed_catalog.spatial_magnitude_counts().ravel()))]
    target_event_rate_forecast2 = benchmark_forecast.data.ravel()[
        numpy.unique(numpy.nonzero(
            observed_catalog.spatial_magnitude_counts().ravel()))]

    # call the primative version operating on ndarray
    out = _binary_t_test_ndarray(
        target_event_rate_forecast1,
        target_event_rate_forecast2,
        observed_catalog.event_count,
        n_fore1,
        n_fore2,
        observed_catalog,
        alpha=alpha
    )

    # storing this for later
    result = EvaluationResult()
    result.name = 'binary paired T-Test'
    result.test_distribution = (out['ig_lower'], out['ig_upper'])
    result.observed_statistic = out['information_gain']
    result.quantile = (out['t_statistic'], out['t_critical'])
    result.sim_name = (forecast.name, benchmark_forecast.name)
    result.obs_name = observed_catalog.name
    result.status = 'normal'
    result.min_mw = numpy.min(forecast.magnitudes)
    return result


def sequential_likelihood(
        gridded_forecasts: Sequence[GriddedForecast],
        observed_catalogs: Sequence[CSEPCatalog],
        seed: int = None,
        random_numbers=None, ):
    """
    Performs the likelihood test on Gridded Forecast using an Observed Catalog.

    Note: The forecast and the observations should be scaled to the same time period before calling this function. This increases
    transparency as no assumptions are being made about the length of the forecasts. This is particularly important for
    gridded forecasts that supply their forecasts as rates.

    Args:
        gridded_forecasts: list csep.core.forecasts.GriddedForecast
        observed_catalogs: list csep.core.catalogs.Catalog
        timewindows: list str.
        seed (int): used fore reproducibility, and testing
        random_numbers (numpy.ndarray): random numbers used to override the random number generation.
                               injection point for testing.
    Returns:
        evaluation_result: csep.core.evaluations.EvaluationResult
    """

    # grid catalog onto spatial grid
    # grid catalog onto spatial grid

    likelihoods = []

    for gridded_forecast, observed_catalog in zip(gridded_forecasts,
                                                  observed_catalogs):
        try:
            _ = observed_catalog.region.magnitudes
        except CSEPCatalogException:
            observed_catalog.region = gridded_forecast.region

        gridded_catalog_data = observed_catalog.spatial_magnitude_counts()

        # simply call likelihood test on catalog and forecast
        qs, obs_ll, simulated_ll = _poisson_likelihood_test(
            gridded_forecast.data, gridded_catalog_data,
            num_simulations=1,
            seed=seed,
            random_numbers=random_numbers,
            use_observed_counts=False,
            normalize_likelihood=False)
        likelihoods.append(obs_ll)
        # populate result data structure

    result = EvaluationResult()
    result.test_distribution = numpy.arange(len(gridded_forecasts))
    result.name = 'Sequential Likelihood'
    result.observed_statistic = likelihoods
    result.quantile = 1
    result.sim_name = gridded_forecast.name
    result.obs_name = observed_catalog.name
    result.status = 'normal'
    result.min_mw = numpy.min(gridded_forecast.magnitudes)

    return result


def sequential_information_gain(
        gridded_forecasts: Sequence[GriddedForecast],
        benchmark_forecasts: Sequence[GriddedForecast],
        observed_catalogs: Sequence[CSEPCatalog],
        seed: int = None,
        random_numbers: Sequence = None):
    """

    Args:
        gridded_forecasts: list csep.core.forecasts.GriddedForecast
        benchmark_forecasts: list csep.core.forecasts.GriddedForecast
        observed_catalogs: list csep.core.catalogs.Catalog
        timewindows: list str.
        seed (int): used fore reproducibility, and testing
        random_numbers (numpy.ndarray): random numbers used to override the random number generation.
                               injection point for testing.

    Returns:
        evaluation_result: csep.core.evaluations.EvaluationResult
    """

    # grid catalog onto spatial grid
    # grid catalog onto spatial grid

    information_gains = []

    for gridded_forecast, reference_forecast, observed_catalog in zip(
            gridded_forecasts, benchmark_forecasts,
            observed_catalogs):
        try:
            _ = observed_catalog.region.magnitudes
        except CSEPCatalogException:
            observed_catalog.region = gridded_forecast.region

        gridded_catalog_data = observed_catalog.spatial_magnitude_counts()

        # simply call likelihood test on catalog and forecast
        qs, obs_ll, simulated_ll = _poisson_likelihood_test(
            gridded_forecast.data, gridded_catalog_data,
            num_simulations=1,
            seed=seed,
            random_numbers=random_numbers,
            use_observed_counts=False,
            normalize_likelihood=False)
        qs, ref_ll, simulated_ll = _poisson_likelihood_test(
            reference_forecast.data, gridded_catalog_data,
            num_simulations=1,
            seed=seed,
            random_numbers=random_numbers,
            use_observed_counts=False,
            normalize_likelihood=False)

        information_gains.append(obs_ll - ref_ll)

    result = EvaluationResult()

    result.test_distribution = numpy.arange(len(gridded_forecasts))
    result.name = 'Sequential Likelihood'
    result.observed_statistic = information_gains
    result.quantile = 1
    result.sim_name = gridded_forecast.name
    result.obs_name = observed_catalog.name
    result.status = 'normal'
    result.min_mw = numpy.min(gridded_forecast.magnitudes)
    return result


def vector_poisson_t_w_test(
        forecast: GriddedForecast,
        benchmark_forecasts: Sequence[GriddedForecast],
        catalog: CSEPCatalog):
    """

        Computes Student's t-test for the information gain per earthquake over
        a list of forecasts and w-test for normality

        Uses all ref_forecasts to perform pair-wise t-tests against the
        forecast provided to the function.

        Args:
            forecast (csep.GriddedForecast): forecast to evaluate
            benchmark_forecasts (list of csep.GriddedForecast): list of forecasts to evaluate
            catalog (csep.AbstractBaseCatalog): evaluation catalog filtered consistent with forecast
            **kwargs: additional default arguments

        Returns:
            results (list of csep.EvaluationResult): iterable of evaluation results
    """
    results_t = []
    results_w = []

    for bmf_j in benchmark_forecasts:
        results_t.append(paired_t_test(forecast, bmf_j, catalog))
        results_w.append(w_test(forecast, bmf_j, catalog))
    result = EvaluationResult()
    result.name = 'Paired T-Test'
    result.test_distribution = 'normal'
    result.observed_statistic = [t.observed_statistic for t in results_t]
    result.quantile = (
        [numpy.abs(t.quantile[0]) - t.quantile[1] for t in results_t],
        [w.quantile for w in results_w])
    result.sim_name = forecast.name
    result.obs_name = catalog.name
    result.status = 'normal'
    result.min_mw = numpy.min(forecast.magnitudes)

    return result


def brier_score(forecast, catalog, spatial_only=False, binary=True):
    def p_series(lambd, niters=500):

        tol = 1e-8
        p2 = 0
        for i in range(niters):
            inc = scipy.stats.poisson.pmf(i, lambd) ** 2
            if numpy.all(numpy.abs((inc - p2) / p2) >= tol) or p2 == 0:
                p2 += inc
            else:
                break
        return p2

    if spatial_only:
        data = forecast.spatial_counts()
        obs = catalog.spatial_counts()
    else:
        data = forecast.data
        obs = catalog.spatial_magnitude_counts()

    if binary:
        obs = (obs >= 1).astype(int)
        prob_success = 1 - scipy.stats.poisson.cdf(0, data)
        brier = []

        for p, o in zip(prob_success.ravel(), obs.ravel()):

            if o == 0:
                brier.append(-2 * p ** 2)
            else:
                brier.append(-2 * (p - 1) ** 2)
        brier = numpy.sum(brier)
    else:
        prob_success = scipy.stats.poisson.pmf(obs, data)
        brier = 2 * (prob_success) - p_series(data) - 1
        brier = numpy.sum(brier)

    for n_dim in obs.shape:
        brier /= n_dim

    result = EvaluationResult(
        name='Brier score',
        observed_statistic=brier,
        test_distribution=[0],
        sim_name=forecast.name
    )
    return result


def _nbd_number_test_ndarray(fore_cnt, obs_cnt, variance, epsilon=1e-6):
    """
    Computes delta1 and delta2 values from the Negative Binomial (NBD) number test.

    Args:
        fore_cnt (float): parameter of negative binomial distribution coming from expected value of the forecast
        obs_cnt (float): count of earthquakes observed during the testing period.
        variance (float): variance parameter of negative binomial distribution coming from historical catalog.
        A variance value of approximately 23541 has been calculated using M5.95+ earthquakes observed worldwide from 1982 to 2013.
        epsilon (float): tolerance level to satisfy the requirements of two-sided p-value

    Returns
        result (tuple): (delta1, delta2)
    """
    var = variance
    mean = fore_cnt
    upsilon = 1.0 - ((var - mean) / var)
    tau = (mean ** 2 / (var - mean))

    delta1 = 1.0 - scipy.stats.nbinom.cdf(obs_cnt - epsilon, tau, upsilon,
                                          loc=0)
    delta2 = scipy.stats.nbinom.cdf(obs_cnt + epsilon, tau, upsilon, loc=0)

    return delta1, delta2


def negative_binomial_number_test(gridded_forecast, observed_catalog,
                                  variance):
    """
    Computes "negative binomial N-Test" on a gridded forecast.

    Computes Number (N) test for Observed and Forecasts. Both data sets are expected to be in terms of event counts.
    We find the Total number of events in Observed Catalog and Forecasted Catalogs. Which are then employed to compute the
    probablities of
    (i) At least no. of events (delta 1)
    (ii) At most no. of events (delta 2) assuming the negative binomial distribution.

    Args:
        gridded_forecast:   Forecast of a Model (Gridded) (Numpy Array)
                    A forecast has to be in terms of Average Number of Events in Each Bin
                    It can be anything greater than zero
        observed_catalog:   Observed (Gridded) seismicity (Numpy Array):
                    An Observation has to be Number of Events in Each Bin
                    It has to be a either zero or positive integer only (No Floating Point)
        variance:   Variance parameter of negative binomial distribution obtained from historical catalog.

    Returns:
        out (tuple): (delta_1, delta_2)
    """
    result = EvaluationResult()

    # observed count
    obs_cnt = observed_catalog.event_count

    # forecasts provide the expeceted number of events during the time horizon of the forecast
    fore_cnt = gridded_forecast.event_count

    epsilon = 1e-6

    # stores the actual result of the number test
    delta1, delta2 = _nbd_number_test_ndarray(fore_cnt, obs_cnt, variance,
                                              epsilon=epsilon)

    # store results
    result.test_distribution = ('negative_binomial', fore_cnt)
    result.name = 'NBD N-Test'
    result.observed_statistic = obs_cnt
    result.quantile = (delta1, delta2)
    result.sim_name = gridded_forecast.name
    result.obs_name = observed_catalog.name
    result.status = 'normal'
    result.min_mw = numpy.min(gridded_forecast.magnitudes)

    return result


def binomial_joint_log_likelihood_ndarray(forecast, catalog):
    """
    Computes Bernoulli log-likelihood scores, assuming that earthquakes follow a binomial distribution.

    Args:
        forecast:   Forecast of a Model (Gridded) (Numpy Array)
                    A forecast has to be in terms of Average Number of Events in Each Bin
                    It can be anything greater than zero
        catalog:    Observed (Gridded) seismicity (Numpy Array):
                    An Observation has to be Number of Events in Each Bin
                    It has to be a either zero or positive integer only (No Floating Point)
    """
    # First, we mask the forecast in cells where we could find log=0.0 singularities:
    forecast_masked = numpy.ma.masked_where(forecast.ravel() <= 0.0,
                                            forecast.ravel())

    # Then, we compute the log-likelihood of observing one or more events given a Poisson distribution, i.e., 1 - Pr(0)
    target_idx = numpy.nonzero(catalog.ravel())
    y = numpy.zeros(forecast_masked.ravel().shape)
    y[target_idx[0]] = 1
    first_term = y * (numpy.log(1.0 - numpy.exp(-forecast_masked.ravel())))

    # Also, we estimate the log-likelihood in cells no events are observed:
    second_term = (1 - y) * (-forecast_masked.ravel().data)
    # Finally, we sum both terms to compute the joint log-likelihood score:
    return sum(first_term.data + second_term.data)


def _binomial_likelihood_test(forecast_data, observed_data,
                              num_simulations=1000, random_numbers=None,
                              seed=None, use_observed_counts=True,
                              verbose=True, normalize_likelihood=False):
    """
    Computes binary conditional-likelihood test from CSEP using an efficient simulation based approach.
    Args:
        forecast_data (numpy.ndarray): nd array where [:, -1] are the magnitude bins.
        observed_data (numpy.ndarray): same format as observation.
        num_simulations: default number of simulations to use for likelihood based simulations
        seed: used for reproducibility of the prng
        random_numbers (numpy.ndarray): can supply an explicit list of random numbers, primarily used for software testing
        use_observed_counts (bool): if true, will simulate catalogs using the observed events, if false will draw from poisson
        distribution
    """

    # Array-masking that avoids log singularities:
    forecast_data = numpy.ma.masked_where(forecast_data <= 0.0, forecast_data)

    # set seed for the likelihood test
    if seed is not None:
        numpy.random.seed(seed)

    # used to determine where simulated earthquake should be placed, by definition of cumsum these are sorted
    sampling_weights = numpy.cumsum(forecast_data.ravel()) / numpy.sum(
        forecast_data)

    # data structures to store results
    sim_fore = numpy.zeros(sampling_weights.shape)
    simulated_ll = []
    n_obs = len(numpy.unique(numpy.nonzero(observed_data.ravel())))
    n_fore = numpy.sum(forecast_data)
    expected_forecast_count = int(n_obs)

    if use_observed_counts and normalize_likelihood:
        scale = n_obs / n_fore
        expected_forecast_count = int(n_obs)
        forecast_data = scale * forecast_data

    # main simulation step in this loop
    for idx in range(num_simulations):
        if use_observed_counts:
            num_events_to_simulate = int(n_obs)
        else:
            num_events_to_simulate = int(
                numpy.random.poisson(expected_forecast_count))

        if random_numbers is None:
            sim_fore = _simulate_catalog(num_events_to_simulate,
                                         sampling_weights, sim_fore)
        else:
            sim_fore = _simulate_catalog(num_events_to_simulate,
                                         sampling_weights, sim_fore,
                                         random_numbers=random_numbers[idx, :])

        # compute joint log-likelihood
        current_ll = binomial_joint_log_likelihood_ndarray(forecast_data.data,
                                                           sim_fore)

        # append to list of simulated log-likelihoods
        simulated_ll.append(current_ll)

        # just be verbose
        if verbose:
            if (idx + 1) % 100 == 0:
                print(f'... {idx + 1} catalogs simulated.')
                target_idx = numpy.nonzero(observed_data.ravel())

    # observed joint log-likelihood
    obs_ll = binomial_joint_log_likelihood_ndarray(forecast_data.data,
                                                   observed_data)

    # quantile score
    qs = numpy.sum(simulated_ll <= obs_ll) / num_simulations

    # float, float, list
    return qs, obs_ll, simulated_ll


def binomial_conditional_likelihood_test(
        gridded_forecast: GriddedForecast,
        observed_catalog: CSEPCatalog,
        num_simulations=1000, seed=None,
        random_numbers=None, verbose=False):
    """
    Performs the binary conditional likelihood test on Gridded Forecast using an Observed Catalog.

    Normalizes the forecast so the forecasted rate are consistent with the observations. This modification
    eliminates the strong impact differences in the number distribution have on the forecasted rates.

    Note: The forecast and the observations should be scaled to the same time period before calling this function. This increases
    transparency as no assumptions are being made about the length of the forecasts. This is particularly important for
    gridded forecasts that supply their forecasts as rates.

    Args:
    gridded_forecast: csep.core.forecasts.GriddedForecast
    observed_catalog: csep.core.catalogs.Catalog
    num_simulations (int): number of simulations used to compute the quantile score
    seed (int): used fore reproducibility, and testing
    random_numbers (numpy.ndarray): random numbers used to override the random number generation. injection point for testing.

    Returns:
    evaluation_result: csep.core.evaluations.EvaluationResult
    """

    # grid catalog onto spatial grid
    try:
        _ = observed_catalog.region.magnitudes
    except CSEPCatalogException:
        observed_catalog.region = gridded_forecast.region

    gridded_catalog_data = observed_catalog.spatial_magnitude_counts()

    # simply call likelihood test on catalog data and forecast
    qs, obs_ll, simulated_ll = _binomial_likelihood_test(gridded_forecast.data,
                                                         gridded_catalog_data,
                                                         num_simulations=num_simulations,
                                                         seed=seed,
                                                         random_numbers=random_numbers,
                                                         use_observed_counts=True,
                                                         verbose=verbose,
                                                         normalize_likelihood=False)

    # populate result data structure
    result = EvaluationResult()
    result.test_distribution = simulated_ll
    result.name = 'Binary CL-Test'
    result.observed_statistic = obs_ll
    result.quantile = qs
    result.sim_name = gridded_forecast.name
    result.obs_name = observed_catalog.name
    result.status = 'normal'
    result.min_mw = numpy.min(gridded_forecast.magnitudes)

    return result


def _binary_t_test_ndarray(target_event_rates1, target_event_rates2, n_obs,
                           n_f1, n_f2, catalog, alpha=0.05):
    """
    Computes binary T test statistic by comparing two target event rate distributions.

    We compare Forecast from Model 1 and with Forecast of Model 2. Information Gain per Active Bin (IGPA) is computed, which is then
    employed to compute T statistic. Confidence interval of Information Gain can be computed using T_critical. For a complete
    explanation see Rhoades, D. A., et al., (2011). Efficient testing of earthquake forecasting models. Acta Geophysica, 59(4),
    728-747. doi:10.2478/s11600-011-0013-5, and Bayona J.A. et al., (2022). Prospective evaluation of multiplicative hybrid earthquake
    forecasting models in California. doi: 10.1093/gji/ggac018.

    Args:
        target_event_rates1 (numpy.ndarray): nd-array storing target event rates
        target_event_rates2 (numpy.ndarray): nd-array storing target event rates
        n_obs (float, int, numpy.ndarray): number of observed earthquakes, should be whole number and >= zero.
        n_f1 (float): Total number of forecasted earthquakes by Model 1
        n_f2 (float): Total number of forecasted earthquakes by Model 2
        catalog: csep.core.catalogs.Catalog
        alpha (float): tolerance level for the type-i error rate of the statistical test

    Returns:
        out (dict): relevant statistics from the t-test
    """
    # Some Pre Calculations -  Because they are being used repeatedly.
    N_p = n_obs
    N = len(numpy.unique(numpy.nonzero(
        catalog.spatial_magnitude_counts().ravel())))  # Number of active bins
    N1 = n_f1
    N2 = n_f2
    X1 = numpy.log(target_event_rates1)  # Log of every element of Forecast 1
    X2 = numpy.log(target_event_rates2)  # Log of every element of Forecast 2

    # Information Gain, using Equation (17)  of Rhoades et al. 2011
    information_gain = (numpy.sum(X1 - X2) - (N1 - N2)) / N

    # Compute variance of (X1-X2) using Equation (18)  of Rhoades et al. 2011
    first_term = (numpy.sum(numpy.power((X1 - X2), 2))) / (N - 1)
    second_term = numpy.power(numpy.sum(X1 - X2), 2) / (numpy.power(N, 2) - N)
    forecast_variance = first_term - second_term

    forecast_std = numpy.sqrt(forecast_variance)
    t_statistic = information_gain / (forecast_std / numpy.sqrt(N))

    # Obtaining the Critical Value of T from T distribution.
    df = N - 1
    t_critical = scipy.stats.t.ppf(1 - (alpha / 2),
                                   df)  # Assuming 2-Tail Distribution  for 2 tail, divide 0.05/2.

    # Computing Information Gain Interval.
    ig_lower = information_gain - (t_critical * forecast_std / numpy.sqrt(N))
    ig_upper = information_gain + (t_critical * forecast_std / numpy.sqrt(N))

    # If T value greater than T critical, Then both Lower and Upper Confidence Interval limits will be greater than Zero.
    # If above Happens, Then It means that Forecasting Model 1 is better than Forecasting Model 2.
    return {'t_statistic': t_statistic,
            't_critical': t_critical,
            'information_gain': information_gain,
            'ig_lower': ig_lower,
            'ig_upper': ig_upper}


def log_likelihood_point_process(observation, forecast, cell_area):
    """
    Log-likelihood for point process

    """
    forecast_density = forecast / cell_area.reshape(-1, 1)
    observation = observation.ravel()
    forecast_density = forecast_density.ravel()
    obs = observation[observation > 0]
    fcst = forecast_density[observation > 0]
    rate_density = []
    for i in range(len(obs)):
        counter = obs[i]
        while counter > 0:
            rate_density = numpy.append(rate_density, fcst[i])
            counter = counter - 1
    ll = numpy.sum(numpy.log(rate_density)) - sum(forecast)
    return ll[0]  # To get a scalar value instead of array


def _standard_deviation(gridded_forecast1, gridded_forecast2,
                        gridded_observation1, gridded_observation2, cell_area1,
                        cell_area2):
    """
    Calculate Variance using forecast 1 and forecast 2.
    But It is calculated using the forecast values corresponding to the non-zero observations.
    The same process is repeated as repeated during calculation of Point Process LL.
    After we get forecast rates for non-zeros observations, then Pooled Variance is calculated.


    Parameters
    ----------
        gridded_forecast1 : forecast
        gridded_forecast2 : benchmark_forecast
        gridded_observation1 : observation
        gridded_observation2 : observation according to benchmark

    Returns
    -------
        Variance

    """

    N_obs = numpy.sum(gridded_observation1)

    forecast_density1 = (gridded_forecast1 / cell_area1.reshape(-1, 1)).ravel()
    forecast_density2 = (gridded_forecast2 / cell_area2.reshape(-1, 1)).ravel()

    gridded_observation1 = gridded_observation1.ravel()
    gridded_observation2 = gridded_observation2.ravel()

    obs1 = gridded_observation1[gridded_observation1 > 0]
    obs2 = gridded_observation2[gridded_observation2 > 0]

    fore1 = forecast_density1[gridded_observation1 > 0]
    fore2 = forecast_density2[gridded_observation2 > 0]

    target_fore1 = []
    target_fore2 = []

    for i in range(len(obs1)):
        counter = obs1[i]
        while counter > 0:
            target_fore1 = numpy.append(target_fore1, fore1[i])
            counter = counter - 1

    for i in range(len(obs2)):
        counter = obs2[i]
        while counter > 0:
            target_fore2 = numpy.append(target_fore2, fore2[i])
            counter = counter - 1

    X1 = numpy.log(target_fore1)
    X2 = numpy.log(target_fore2)
    first_term = (numpy.sum(numpy.power((X1 - X2), 2))) / (N_obs - 1)
    second_term = numpy.power(numpy.sum(X1 - X2), 2) / (
            numpy.power(N_obs, 2) - N_obs)
    forecast_variance = first_term - second_term

    return forecast_variance


def paired_ttest_point_process(
        forecast: GriddedForecast,
        benchmark_forecast: GriddedForecast,
        observed_catalog: CSEPCatalog,
        alpha=0.05):
    """
    Function for T test based on Point process LL.
    Works for comparing forecasts for different grids

    Parameters
    ----------
        forecast (csep.core.forecasts.GriddedForecast): nd-array storing gridded rates, axis=-1 should be the magnitude column
        benchmark_forecast (csep.core.forecasts.GriddedForecast): nd-array storing gridded rates, axis=-1 should be the magnitude column
        observed_catalog (csep.core.catalogs.AbstractBaseCatalog): number of observed earthquakes, should be whole number and >= zero.
        alpha (float): tolerance level for the type-i error rate of the statistical test
        scale (bool): if true, scale forecasted rates down to a single day
    Returns:
        evaluation_result: csep.core.evaluations.EvaluationResult
    """

    gridded_forecast1 = numpy.array(forecast.data)

    observed_catalog.region = forecast.region

    gridded_observation1 = forecast.region._get_spatial_magnitude_counts(
        observed_catalog, mag_bins=forecast.magnitudes)
    cell_area1 = numpy.array(forecast.region.cell_area)
    ll1 = log_likelihood_point_process(gridded_observation1, gridded_forecast1,
                                       cell_area1)

    # Forecast 2
    gridded_forecast2 = numpy.array(benchmark_forecast.data)
    gridded_observation2 = benchmark_forecast.region._get_spatial_magnitude_counts(
        observed_catalog,
        mag_bins=forecast.magnitudes)
    cell_area2 = numpy.array(benchmark_forecast.region.cell_area)
    ll2 = log_likelihood_point_process(gridded_observation2, gridded_forecast2,
                                       cell_area2)

    assert numpy.sum(gridded_observation1) == numpy.sum(
        gridded_observation2), 'Sum of Gridded Catalog is not same'

    N_obs = numpy.sum(gridded_observation1)

    information_gain = (ll1 - ll2) / N_obs

    forecast_variance = _standard_deviation(gridded_forecast1,
                                            gridded_forecast2,
                                            gridded_observation1,
                                            gridded_observation2, cell_area1,
                                            cell_area2)

    forecast_std = numpy.sqrt(forecast_variance)
    t_statistic = information_gain / (forecast_std / numpy.sqrt(N_obs))

    # Obtaining the Critical Value of T from T distribution.
    df = N_obs - 1
    t_critical = scipy.stats.t.ppf(1 - (alpha / 2),
                                   df)  # Assuming 2-Tail Distribution  for 2 tail, divide 0.05/2.

    # Computing Information Gain Interval.
    ig_lower = information_gain - (
            t_critical * forecast_std / numpy.sqrt(N_obs))
    ig_upper = information_gain + (
            t_critical * forecast_std / numpy.sqrt(N_obs))

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


def plot_negbinom_consistency_test(eval_results, normalize=False, axes=None,
                          one_sided_lower=False, variance=None, plot_args=None,
                          show=False):
    """ Plots results from CSEP1 tests following the CSEP1 convention.

    Note: All of the evaluations should be from the same type of evaluation, otherwise the results will not be
          comparable on the same figure.

    Args:
        eval_results (list): Contains the tests results :class:`csep.core.evaluations.EvaluationResult` (see note above)
        normalize (bool): select this if the forecast likelihood should be normalized by the observed likelihood. useful
                          for plotting simulation based simulation tests.
        one_sided_lower (bool): select this if the plot should be for a one sided test
        plot_args(dict): optional argument containing a dictionary of plotting arguments, with keys as strings and items as described below

    Optional plotting arguments:
        * figsize: (:class:`list`/:class:`tuple`) - default: [6.4, 4.8]
        * title: (:class:`str`) - default: name of the first evaluation result type
        * title_fontsize: (:class:`float`) Fontsize of the plot title - default: 10
        * xlabel: (:class:`str`) - default: 'X'
        * xlabel_fontsize: (:class:`float`) - default: 10
        * xticks_fontsize: (:class:`float`) - default: 10
        * ylabel_fontsize: (:class:`float`) - default: 10
        * color: (:class:`float`/:class:`None`) If None, sets it to red/green according to :func:`_get_marker_style` - default: 'black'
        * linewidth: (:class:`float`) - default: 1.5
        * capsize: (:class:`float`) - default: 4
        * hbars:  (:class:`bool`)  Flag to draw horizontal bars for each model - default: True
        * tight_layout: (:class:`bool`) Set matplotlib.figure.tight_layout to remove excess blank space in the plot - default: True

    Returns:
        ax (:class:`matplotlib.pyplot.axes` object)
    """

    try:
        results = list(eval_results)
    except TypeError:
        results = [eval_results]
    results.reverse()
    # Parse plot arguments. More can be added here
    if plot_args is None:
        plot_args = {}
    figsize = plot_args.get('figsize', None)
    title = plot_args.get('title', results[0].name)
    title_fontsize = plot_args.get('title_fontsize', None)
    xlabel = plot_args.get('xlabel', '')
    xlabel_fontsize = plot_args.get('xlabel_fontsize', None)
    xticks_fontsize = plot_args.get('xticks_fontsize', None)
    ylabel_fontsize = plot_args.get('ylabel_fontsize', None)
    color = plot_args.get('color', 'black')
    linewidth = plot_args.get('linewidth', None)
    capsize = plot_args.get('capsize', 4)
    hbars = plot_args.get('hbars', True)
    tight_layout = plot_args.get('tight_layout', True)
    percentile = plot_args.get('percentile', 95)
    plot_mean = plot_args.get('mean', False)

    if axes is None:
        fig, ax = pyplot.subplots(figsize=figsize)
    else:
        ax = axes
        fig = ax.get_figure()

    xlims = []

    for index, res in enumerate(results):
        var = variance
        observed_statistic = res.observed_statistic
        mean = res.test_distribution[1]
        upsilon = 1.0 - ((var - mean) / var)
        tau = (mean ** 2 / (var - mean))
        plow = scipy.stats.nbinom.ppf((1 - percentile / 100.) / 2., tau,
                                       upsilon)
        phigh = scipy.stats.nbinom.ppf(1 - (1 - percentile / 100.) / 2.,
                                      tau, upsilon)

        if not numpy.isinf(
                observed_statistic):  # Check if test result does not diverges
            percentile_lims = numpy.array([[mean - plow, phigh - mean]]).T
            ax.plot(observed_statistic, index,
                    _get_marker_style(observed_statistic, (plow, phigh),
                                      one_sided_lower))
            ax.errorbar(mean, index, xerr=percentile_lims,
                        fmt='ko' * plot_mean, capsize=capsize,
                        linewidth=linewidth, ecolor=color)
            # determine the limits to use
            xlims.append((plow, phigh, observed_statistic))
            # we want to only extent the distribution where it falls outside of it in the acceptable tail
            if one_sided_lower:
                if observed_statistic >= plow and phigh < observed_statistic:
                    # draw dashed line to infinity
                    xt = numpy.linspace(phigh, 99999, 100)
                    yt = numpy.ones(100) * index
                    ax.plot(xt, yt, linestyle='--', linewidth=linewidth,
                            color=color)

        else:
            print('Observed statistic diverges for forecast %s, index %i.'
                  ' Check for zero-valued bins within the forecast' % (
                      res.sim_name, index))
            ax.barh(index, 99999, left=-10000, height=1, color=['red'],
                    alpha=0.5)

    try:
        ax.set_xlim(*_get_axis_limits(xlims))
    except ValueError:
        raise ValueError(
            'All EvaluationResults have infinite observed_statistics')
    ax.set_yticks(numpy.arange(len(results)))
    ax.set_yticklabels([res.sim_name for res in results],
                       fontsize=ylabel_fontsize)
    ax.set_ylim([-0.5, len(results) - 0.5])
    if hbars:
        yTickPos = ax.get_yticks()
        if len(yTickPos) >= 2:
            ax.barh(yTickPos, numpy.array([99999] * len(yTickPos)),
                    left=-10000,
                    height=(yTickPos[1] - yTickPos[0]), color=['w', 'gray'],
                    alpha=0.2, zorder=0)
    ax.set_title(title, fontsize=title_fontsize)
    ax.set_xlabel(xlabel, fontsize=xlabel_fontsize)
    ax.tick_params(axis='x', labelsize=xticks_fontsize)
    if tight_layout:
        ax.figure.tight_layout()
        fig.tight_layout()

    if show:
        pyplot.show()

    return ax


def _get_marker_style(obs_stat, p, one_sided_lower):
    """Returns matplotlib marker style as fmt string"""
    if obs_stat < p[0] or obs_stat > p[1]:
        # red circle
        fmt = 'ro'
    else:
        # green square
        fmt = 'gs'
    if one_sided_lower:
        if obs_stat < p[0]:
            fmt = 'ro'
        else:
            fmt = 'gs'
    return fmt


def _get_axis_limits(pnts, border=0.05):
    """Returns a tuple of x_min and x_max given points on plot."""
    x_min = numpy.min(pnts)
    x_max = numpy.max(pnts)
    xd = (x_max - x_min) * border
    return (x_min - xd, x_max + xd)

