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
