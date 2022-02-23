import numpy
import pandas
from csep.core.regions import CartesianGrid2D
from csep.core.forecasts import GriddedForecast
from csep.utils.plots import plot_spatial_dataset



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


def plot_lowsampled_forecast(forecast, plot_args, k=4):

    """
    Wrapper to plot quickly global forecast when no large image resolution is required, where k controles the

    """
    print('\tPlotting Forecast')
    plot_args['title'] = forecast.name
    region = forecast.region
    coords = region.origins()
    dataset = numpy.log10(forecast.spatial_counts(cartesian=True))[::k, ::k]
    region.xs = numpy.unique(region.get_cartesian(coords[:, 0])[0, ::k])
    region.ys = numpy.unique(region.get_cartesian(coords[:, 1])[::k, 0])
    plot_spatial_dataset(dataset, region, set_global=True, plot_args=plot_args)


def prepare_forecast(model_path, time_horizon, name=None, dh=0.1, **kwargs):
    """ Returns a forecast from a time-independent model

        Note: For the time-independent global models the rates should be 'per year'

        Args:
            model_path (str): filepath of model
            time_horizon (float): time horizon of forecast in years     # todo < Should this be a pair of datetimes instead of the scale? In such way, the forecast object has time bounds, and can be identified with a catalog.
            name (str): name of the model                               # todo:  Should dh be obtained from the experiment properties? Fixed? or just as optional arg?
            dh (float): Cell size
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
        magnitudes = [float(i) for i in model.readline().split(',')[6:]]

    region = CartesianGrid2D.from_origins(data[:, [0, 2]], dh)              # Create the region from the lat-lon (low-corner) of cells
    rates = data[:, 6:]

    forecast = GriddedForecast(data=rates, region=region, magnitudes=magnitudes, name=name, **kwargs)
    forecast.scale(time_horizon)
    print(f'\t Total {forecast.event_count:.4f} events forecasted in {time_horizon:.2f} years')

    return forecast

