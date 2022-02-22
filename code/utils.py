import numpy
from csep.core.regions import QuadtreeGrid2D

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

def aggregate_quadtree_forecast(cart_forecast, quadtree_region):
    """ Aggregates conventional forecast onto quadtree region

        Args:
            cart_forecast (GriddedForecast): gridded forecast with regular grid
            quadtree_region (QuadtreeRegion2D): desired quadtree region 

        Returns:
            qt_forecast (GriddedForecast): gridded forecast on quadtree grid
    """
    pass

def prepare_forecast(model_path, time_horizon, **kwargs):
    """ Returns a forecast from a time-independent model

        Note: For the time-independent global models the rates should be 'per year'

        Args:
            model_path (str): filepath of model
            time_horizon (float): time horizon of forecast in years
            **kwargs: other keyword arguments

        Returns:
            forecast (GriddedForecast): pycsep forecast object with rates scaled to time horizon
    """
    pass


def load_quadtree_forecast_csv(csv_fname):
    """ Load quadtree forecasted stored as csv file
        The format expects forecast as a comma separated file, in which first column corresponds to quadtree grid cell (quadkey).
        The second and thrid columns indicate depth range.
        The corresponding enteries in the respective row are forecast rates corresponding to the magnitude bins.
        The first line of forecast is a header, and its format is listed here:
            'Quadkey', depth_min, depth_max, Mag_0, Mag_1, Mag_2, Mag_3 , ....
             Quadkey is a string. Rest of the values are floats.
        For the purposes of defining region objects quadkey is used.
        We assume that the starting value of magnitude bins are provided in the header.
        Args:
            csv_fname: file name of csep forecast in csv format
        Returns:
            rates, region, mws (numpy.ndarray, QuadtreeRegion2D, numpy.ndarray): rates, region, and magnitude bins needed
                                                                                 to define QuadTree forecasts
     """

    data = numpy.genfromtxt(csv_fname, dtype='str', delimiter=',')
    quadkeys = data[1:, 0]
    mws = data[0, 3:]
    rates = data[1:, 3:]
    rates = rates.astype(float)
    region = QuadtreeGrid2D.from_quadkeys(quadkeys, magnitudes=mws)

    return rates, region, mws 