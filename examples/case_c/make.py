import numpy as np
from datetime import datetime
from csep import query_comcat
from csep.core.regions import california_relm_region
from csep.utils.time_utils import decimal_year
from csep.core.catalogs import CSEPCatalog
from csep.core.regions import CartesianGrid2D
from fecsep.utils import cleaner_range


def write_forecast(rates, region, magnitudes, name):
    with open(name + '.csv', 'w') as file_:
        file_.write(
            f'lon_min, lon_max, lat_min, lat_max, depth_min, depth_max, {", ".join(magnitudes.astype(str))}')
        for cell, rate in zip(region.origins(), rates):
            file_.write(
                f'\n{cell[0]}, {cell[0] + dh}, {cell[1]}, {cell[1] + dh}, 0, 30, ')
            file_.write(', '.join(rate.astype(str)))


def MFD(a_val, b_val, mags):
    mfd = np.array([10 ** (a_val - b_val * i) - 10 ** (a_val - b_val * j)
                    for i, j in
                    zip(mags[:-1], mags[1:])] + [
                       10 ** (a_val - b_val * mags[-1])])
    return mfd


start = datetime(2010, 1, 1)
end = datetime(2020, 1, 1)
region = california_relm_region()
bbox = region.get_bbox()
min_mag = 4.0
max_mag = 8.0
magnitudes = cleaner_range(min_mag, max_mag, 4)[0]

region = CartesianGrid2D.from_origins(np.genfromtxt('region.txt'),
                                      magnitudes=magnitudes)
dh = region.dh
origins = region.origins()
events = []

catalog = CSEPCatalog.load_json('catalog.json')

cat_filtered = catalog.filter_spatial(region=region)

# Forecast c (-almost- perfect forecast)
rates = cat_filtered.spatial_magnitude_counts() / (
        decimal_year(end) - decimal_year(start))
write_forecast(rates, region, magnitudes, 'models/model_c')

# Forecast a (GR-1 uniform)
b = 1
a = np.log10(cat_filtered.get_number_of_events() / (
        decimal_year(end) - decimal_year(start))) + min_mag * b
rates = 0.5 * np.ones((2, 1)) * MFD(a, b, magnitudes)
write_forecast(rates, region, magnitudes, 'models/model_a')

# Forecast a (GR-1 spatial inhomogeneous)
spatial_pdf = cat_filtered.spatial_magnitude_counts().sum(axis=1) / \
              cat_filtered.get_number_of_events()
rates = spatial_pdf[:, None] * MFD(a, b, magnitudes)
write_forecast(rates, region, magnitudes, 'models/model_b')

# Forecast a (GR-fit spatial inhomogeneous)
cumsum = np.flip(np.cumsum(np.flip(cat_filtered.magnitude_counts()))) / (
        decimal_year(end) - decimal_year(start))
b_, a_ = np.polyfit(magnitudes[cumsum != 0.],
                    np.log10(cumsum[cumsum != 0.]), deg=1)

rates = spatial_pdf[:, None] * MFD(a_, -b_, magnitudes)
write_forecast(rates, region, magnitudes, 'models/model_d')
