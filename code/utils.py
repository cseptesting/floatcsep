import numpy
import pandas
from csep.core.regions import CartesianGrid2D, compute_vertices
from csep.core.forecasts import GriddedForecast
from csep.utils.plots import plot_spatial_dataset
import itertools
from csep.models import Polygon


def plot_matrix_comparative_test(evaluation_results, plot_args=None):
    """ Produces matrix plot for comparative tests for all forecasts

        Args:
            evaluation_results (list of result objects): paired t-test results
            plot_args (dict): plotting arguments for function

        Returns:
            ax (matplotlib.Axes): handle for figure
    """
    pass


def plot_binary_consistency_test(evaluation_result, plot_args=None):
    """ Plots consistency test result for binary evaluations

        Note: We might be able to recycle poisson here, but lets wrap it

        Args:
            evaluation_results (list of result objects): paired t-test results
            plot_args (dict): plotting arguments for function

        Returns:
            ax (matplotlib.Axes): handle for figure
    """
    pass


def global_region(dh=0.1, name="global", magnitudes=None):
    """ Creates a global region used for evaluating gridded forecasts on the global scale.

    Modified from csep.core.regions.global_region

    The gridded region corresponds to the

    Args:
        dh:

    Returns:
        csep.utils.CartesianGrid2D:
    """
    # generate latitudes

    lons = numpy.arange(-180.0, 180, dh)
    lats = numpy.arange(-90, 90, dh)
    coords = itertools.product(lons,lats)
    region = CartesianGrid2D([Polygon(bbox) for bbox in compute_vertices(coords, dh)], dh, name=name)
    if magnitudes is not None:
        region.magnitudes = magnitudes
    return region


def resample_block_model(model_path, resample_path, k):
    """

    Creates a resampled version of a model for code testing purposes

    :param model_path: Original model
    :param resample_path: Path to resampled model
    :param k: Grouping number of cells, creating blocks (k times k). Must be factor of 3600 and 1800

    """
    db = pandas.read_csv(model_path, header=0, sep=' ', escapechar='#')
    data = db.to_numpy()

    with open(model_path, 'r') as model:                                    # Read the magnitude columns in the forecast
        header = model.readline()[2:]
        magnitudes = [float(i) for i in header.split(' ')[6:]]

    nx = 3600
    ny = 1800
    nm = len(magnitudes)

    if nx % k != 0 or ny % k != 0:
        raise Exception('Grouping factor k must be factor of 3600 and 1800')

    ordered = data[:, 6:].reshape((nx, ny, nm))
    resampled = ordered.reshape(int(nx/k), k, int(ny/k), k, nm).sum(axis=(1, 3))  # blocks, n, blocks, n
    rates = resampled.reshape(int(nx/k) * int(ny/k), nm)

    region = global_region(0.1 * k)
    origin = region.origins()

    cells = numpy.vstack((origin[:, 0], origin[:, 0] + region.dh,
                          origin[:, 1], origin[:, 1] + region.dh,
                          numpy.zeros(int(nx * ny / k**2)), 70 * numpy.ones(int(nx * ny / k**2)))).T

    new_array = numpy.hstack((cells, rates))
    header = 'lon_min lon_max lat_min lat_max depth_min depth_max ' + ' '.join([str(i) for i in magnitudes])

    numpy.savetxt(resample_path, new_array, fmt=6 * ['%.1f'] + 31 * ['%.16e'], header=header)


def resample_models(k=20):


    """
    Resamples all forecast to low resolution for code testing purposes

    :param k: resample factor
    :return:
    """

    model_original = '../models/GEAR1_csep.txt'
    model_resampled = '../models/GEAR_resampled.txt'
    resample_block_model(model_original, model_resampled, k=k)

    model_original = '../models/KJSS_csep.txt'
    model_resampled = '../models/KJSS_resampled.txt'
    resample_block_model(model_original, model_resampled, k=k)

    model_original = '../models/SHIFT2F_GSRM_csep.txt'
    model_resampled = '../models/SHIFT2F_GSRM_resampled.txt'
    resample_block_model(model_original, model_resampled, k=k)

    model_original = '../models/TEAMr_csep.txt'
    model_resampled = '../models/TEAMr_resampled.txt'
    resample_block_model(model_original, model_resampled, k=k)

    model_original = '../models/WHEELr_csep.txt'
    model_resampled = '../models/WHEELr_resampled.txt'
    resample_block_model(model_original, model_resampled, k=k)


def plot_forecast_lowres(forecast, plot_args, k=4):

    """
    Plot a reduced resolution plot. The forecast values are kept the same, but cells are enlarged
    :param forecast: GriddedForecast object
    :param plot_args: arguments to be passed to plot_spatial_dataset
    :param k: Resampling factor. Selects cells every k row and k columns.

    """

    print('\tPlotting Forecast')
    plot_args['title'] = forecast.name
    region = forecast.region
    coords = region.origins()
    dataset = numpy.log10(forecast.spatial_counts(cartesian=True))[::k, ::k]
    region.xs = numpy.unique(region.get_cartesian(coords[:, 0])[0, ::k])
    region.ys = numpy.unique(region.get_cartesian(coords[:, 1])[::k, 0])
    plot_spatial_dataset(dataset, region, set_global=True, plot_args=plot_args)


def prepare_forecast(model_path, time_horizon, dh=0.1, name=None, **kwargs):
    """ Returns a forecast from a time-independent model

        Note: For the time-independent global models the rates should be 'per year'

        Args:
            model_path (str): filepath of model
            time_horizon (float): time horizon of forecast in years     # todo < Should this be a pair of datetimes instead of the scale? In such way, the forecast object has time bounds, and can be identified with a catalog.
            dh (float): spatial cell size
            name (str): name of the model
            **kwargs: other keyword arguments

        Returns:
            forecast (GriddedForecast): pycsep forecast object with rates scaled to time horizon
    """

    if name is None:                                                        # Get name from file if none is provided
        name = model_path.split('_csep.txt')[0].split('/')[-1]

    print(f'Loading Forecast {name}')

    db = pandas.read_csv(model_path, header=0, sep=' ', escapechar='#')
    data = db.to_numpy()
    with open(model_path, 'r') as model:                                    # Read the magnitude columns in the forecast
        magnitudes = [float(i) for i in model.readline().split(' ')[7:]]   # todo check it works with escapechar #

    region = global_region(dh)             #todo: Hard coded here, but should be eventually able to read the region? e.g test italy using gear
    rates = data[:, 6:]

    forecast = GriddedForecast(data=rates, region=region, magnitudes=magnitudes, name=name, **kwargs)
    forecast.scale(time_horizon)
    print(f'\t Total {forecast.event_count:.4f} events forecasted in {time_horizon:.2f} years')

    return forecast

