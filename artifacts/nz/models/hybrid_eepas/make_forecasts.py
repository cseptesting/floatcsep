import os.path
import numpy
import pandas
import argparse


def truncated_GR(bval, mag_bin, mmin, mmax, n=1):

    aval = numpy.log10(n) + bval * mmin
    mag_bins = numpy.arange(mmin, mmax + mag_bin, mag_bin)
    mag_weights = (10 ** (aval - bval * (mag_bins - mag_bin / 2.)) - 10 ** (aval - bval * (mag_bins + mag_bin / 2.))) / \
                  (10 ** (aval - bval * (mmin - mag_bin / 2.)) - 10 ** (aval - bval * (mmax + mag_bin / 2.)))
    return aval, mag_bins, numpy.array(mag_weights)


def make(raw, forecast_path, N=None, bval=None, mmin=5.0, mmax=8.0):

    _, mws, mw_weights = truncated_GR(bval, mag_bin=0.1, mmin=mmin, mmax=mmax, n=N)
    data = pandas.read_csv(raw, header=0)
    data.rename(columns=lambda x: x.strip(), inplace=True)
    if data.columns.shape[0] <= 12:
        spatial_rate = data['rate'].to_numpy()
        rates = spatial_rate[:, numpy.newaxis] * mw_weights[numpy.newaxis, :]
        data.drop(['rate', 'mask', 'm_min', 'm_max', 'mask', 'dispersion'], axis=1, inplace=True) # only poisson

    else:
        spatial_rate = data[[f'{i:.1f}' for i in mws]].to_numpy()
        rates = []
        for i in spatial_rate:
            mi_rates = i*mw_weights
            rates.append(mi_rates)
        rates = numpy.array(rates)
        data.drop(['dispersion'], axis=1, inplace=True) # only poisson

    data[[f'{i:.1f}' for i in mws]] = rates
    data.to_csv(forecast_path, index=False)


def main():

    parser = argparse.ArgumentParser(description='Creates New Zealand models for a given yearly rate and b-val')
    parser.add_argument('-N', type=float, nargs='?',
                        help='Total yearly rate')
    parser.add_argument('-b', type=float, nargs='?',
                        help='b value')
    args = parser.parse_args()

    N = args.N
    b = args.b
    for name in ['ao_eepas.csv', 'at_eepas.csv', 'm_eepas.csv']:
        new_path = f'_N{N:.1f}_b{b:.2f}'.join(os.path.splitext(name))
        make(name, new_path, N, b)

if __name__ == '__main__':
    main()

