import h5py
import pandas
import argparse, os
import numpy


def quadtree_to_hdf5(filename):
    """

    Args:
        *args:

    Returns:

    """

    hdf5_filename = f'{os.path.splitext(filename)[0]}.hdf5'

    with open(filename, 'r') as file_:
        qt_header = file_.readline().split(',')
        fmts = [str] + [float] * (len(qt_header) - 1)
    qt_formats = {i:j for i, j in zip(qt_header, fmts)}
    data = pandas.read_csv(filename, header=0, dtype=qt_formats)

    quadkeys = [i.encode('ascii', 'ignore') for i in data.tile]
    m = numpy.array(data.keys()[3:]).astype(float)
    rates = data[m.astype(str)].to_numpy()

    with h5py.File(hdf5_filename, 'a') as hf:
        hf.require_dataset('rates', shape=rates.shape, dtype=float)
        hf['rates'][:] = rates
        hf.require_dataset('magnitudes', shape=m.shape, dtype=float)
        hf['magnitudes'][:] = m
        hf.require_dataset('quadkeys', shape=len(quadkeys), dtype='S16')
        hf['quadkeys'][:] = quadkeys



def serialize():

    parser = argparse.ArgumentParser()
    parser.add_argument("--format", help="format")
    parser.add_argument("--filename", help="Model forecast name")
    args = parser.parse_args()

    if args.format == 'quadtree':
        quadtree_to_hdf5(args.filename)
    print('serializing ready')


if __name__ == '__main__':

    serialize()
