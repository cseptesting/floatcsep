from matplotlib import pyplot
import cartopy
from utils import prepare_forecast, plot_lowsampled_forecast
from accessors import query_isc_gcmt
from matplotlib import pyplot



def test_catalog_query_plot():

    catalog = query_isc_gcmt(start_year=2019, start_month=8, verbose=True)
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

    plot_lowsampled_forecast(forecast, plot_args, k=6)
    pyplot.show()


if __name__ == '__main__':

    test_catalog_query_plot()
    test_forecast_read_plot()


