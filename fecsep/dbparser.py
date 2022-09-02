import h5py
import pandas
import argparse, os
import numpy
import itertools

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

def dat_to_hdf5(filename):
    """
    from csep.load_ascii
    Args:
        *args:

    Returns:

    """

    hdf5_filename = f'{os.path.splitext(filename)[0]}.hdf5'
    # Load data
    data = numpy.loadtxt(filename)
    all_polys = data[:, :4]
    all_poly_mask = data[:, -1]
    sorted_idx = numpy.sort(numpy.unique(all_polys, return_index=True, axis=0)[1], kind='stable')
    unique_poly = all_polys[sorted_idx]
    poly_mask = all_poly_mask[sorted_idx]
    all_mws = data[:, -4]
    sorted_idx = numpy.sort(numpy.unique(all_mws, return_index=True)[1], kind='stable')
    mws = all_mws[sorted_idx]
    bboxes = numpy.array([tuple(itertools.product(bbox[:2], bbox[2:])) for bbox in unique_poly])
    dh = float(unique_poly[0, 3] - unique_poly[0, 2])


    n_mag_bins = len(mws)
    rates = data[:, -2].reshape(len(bboxes), n_mag_bins)

    with h5py.File(hdf5_filename, 'a') as hf:
        hf.require_dataset('rates', shape=rates.shape, dtype=float)
        hf['rates'][:] = rates
        hf.require_dataset('magnitudes', shape=mws.shape, dtype=float)
        hf['magnitudes'][:] = mws
        hf.require_dataset('bboxes', shape=bboxes.shape, dtype=float)
        hf['bboxes'][:] = bboxes
        hf.require_dataset('dh', shape=(1,), dtype=float)
        hf['dh'][:] = dh
        hf.require_dataset('poly_mask', shape=poly_mask.shape, dtype=float)
        hf['poly_mask'][:] = poly_mask


def serialize():
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", help="format")
    parser.add_argument("--filename", help="Model forecast name")
    args = parser.parse_args()

    if args.format == 'quadtree':
        quadtree_to_hdf5(args.filename)
    if args.format == 'dat':
        dat_to_hdf5(args.filename)
    print('serializing ready')


if __name__ == '__main__':

    serialize()
