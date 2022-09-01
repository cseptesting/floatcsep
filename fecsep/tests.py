from matplotlib import pyplot
import cartopy
from fecsep.utils import prepare_forecast, plot_forecast_lowres, resample_block_model
from fecsep.accessors import query_isc_gcmt
from matplotlib import pyplot
import datetime


def test_catalog_query_plot():

    start_datetime = datetime.datetime(2020, 1, 1)
    end_datetime = datetime.datetime(2021, 1, 1)
    catalog = query_isc_gcmt(start_datetime=start_datetime, end_datetime=end_datetime, min_mw=5.95, verbose=True)
    catalog.plot(set_global=True)
    pyplot.show()


def test_forecast_read_plot():
    model = '../models/GEAR1_csep.txt'
    forecast = prepare_forecast(model, 3, name='GEAR')

    plot_args = {'figsize': (10, 6),
                 'coastline': True,
                 'linecolor': 'grey',
                 'basemap': None,
                 'projection': cartopy.crs.Robinson(central_longitude=-179),
                 'grid_labels': False,
                 'cmap': 'magma',
                 'clim': [-8, -1],
                 'clabel': r'$\log_{10}N_{eq}\left(M_L \geq 5.95\right)$ per '
                           r'$0.1^\circ\times 0.1^\circ $ per year'}

    plot_forecast_lowres(forecast, plot_args, k=6)
    pyplot.show()


def test_resample_model():

    model_original = '../models/GEAR1_csep.txt'
    model_resampled = '../models/GEAR_resampled.txt'
    resample_block_model(model_original, model_resampled, k=20)

    plot_args = {'figsize': (10, 6),
                 'coastline': True,
                 'linecolor': 'grey',
                 'basemap': None,
                 'projection': cartopy.crs.Robinson(central_longitude=-179),
                 'grid_labels': False,
                 'cmap': 'magma',
                 'clim': [-8, -1],
                 'clabel': r'$\log_{10}N_{eq}\left(M_L \geq 5.95\right)$ per '
                           r'$2^\circ\times 2^\circ $ per year'}

    forecast = prepare_forecast(model_resampled, time_horizon=1, dh=0.1*20)
    forecast.plot(set_global=True, plot_args=plot_args)
    pyplot.show()


if __name__ == '__main__':

    test_catalog_query_plot()
    test_forecast_read_plot()
    test_resample_model()

