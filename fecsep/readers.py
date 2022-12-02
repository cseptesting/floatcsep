import h5py
import pandas
import csep
import argparse, os
import numpy
import xml.etree.ElementTree as et
import itertools
from csep.models import Polygon
from csep.core.regions import QuadtreeGrid2D, CartesianGrid2D
import time


class ForecastParsers:

    @staticmethod
    def dat(filename):

        data = numpy.loadtxt(filename)
        all_polys = data[:, :4]
        all_poly_mask = data[:, -1]
        sorted_idx = numpy.sort(
            numpy.unique(all_polys, return_index=True, axis=0)[1],
            kind='stable')
        unique_poly = all_polys[sorted_idx]
        poly_mask = all_poly_mask[sorted_idx]
        all_mws = data[:, -4]
        sorted_idx = numpy.sort(numpy.unique(all_mws, return_index=True)[1],
                                kind='stable')
        mws = all_mws[sorted_idx]
        bboxes = numpy.array(
            [tuple(itertools.product(bbox[:2], bbox[2:])) for bbox in
             unique_poly])
        dh = float(unique_poly[0, 3] - unique_poly[0, 2])

        n_mag_bins = len(mws)
        rates = data[:, -2].reshape(len(bboxes), n_mag_bins)

        region = CartesianGrid2D(
            [Polygon(bbox) for bbox in bboxes], dh, mask=poly_mask)

        return rates, region, mws

    @staticmethod
    def xml(filename):
        name = filename.split('.')[1]
        author = filename.split('.')[0].split('-')[0].capitalize()
        print('Processing model %s of author %s' % (name, author))
        tree = et.parse(filename)
        root = tree.getroot()

        data_Hij = []
        m_bins = []

        for children in list(root[0]):
            if 'modelName' in children.tag:
                name_xml = children.text
            elif 'author' in children.tag:
                author_xml = children.text
            elif 'forecastStartDate' in children.tag:
                start_date = children.text.replace('Z', '')
            elif 'forecastEndDate' in children.tag:
                end_date = children.text.replace('Z', '')
            elif 'defaultMagBinDimension' in children.tag:
                m_bin_width = float(children.text)
            elif 'lastMagBinOpen' in children.tag:
                lastmbin = float(children.text)
            elif 'defaultCellDimension' in children.tag:
                cell_dim = {i[0]: float(i[1]) for i in children.attrib.items()}
            elif 'depthLayer' in children.tag:
                depth = {i[0]: float(i[1]) for i in root[0][9].attrib.items()}
                cells = root[0][9]

        for cell in cells:
            cell_data = []
            m_cell_bins = []
            for i, m in enumerate(cell.iter()):
                if i == 0:
                    cell_data.extend([float(m.attrib['lon']),
                                      float(m.attrib['lat'])])
                else:
                    cell_data.append(float(m.text))
                    m_cell_bins.append(float(m.attrib['m']))
            data_Hij.append(cell_data)
            m_bins.append(m_cell_bins)
        try:
            data_Hij = numpy.array(data_Hij)
            m_bins = numpy.array(m_bins)
        except:
            raise Exception('Data is not square ')

        magnitudes = m_bins[0, :]
        rates = data_Hij[:, -len(magnitudes):]
        all_polys = numpy.vstack((data_Hij[:, 0] - cell_dim['lonRange'] / 2.,
                                  data_Hij[:, 0] + cell_dim['lonRange'] / 2.,
                                  data_Hij[:, 1] - cell_dim['latRange'] / 2.,
                                  data_Hij[:, 1] + cell_dim[
                                      'latRange'] / 2.)).T
        bboxes = numpy.array(
            [tuple(itertools.product(bbox[:2], bbox[2:])) for bbox in
             all_polys])
        dh = float(all_polys[0, 3] - all_polys[0, 2])
        poly_mask = numpy.ones(bboxes.shape[0])

        region = CartesianGrid2D(
            [Polygon(bbox) for bbox in bboxes], dh, mask=poly_mask)

        return rates, region, magnitudes

    @staticmethod
    def quadtree(filename):
        with open(filename, 'r') as file_:
            qt_header = file_.readline().split(',')
            fmts = [str] + [float] * (len(qt_header) - 1)
        qt_formats = {i: j for i, j in zip(qt_header, fmts)}
        data = pandas.read_csv(filename, header=0, dtype=qt_formats)

        quadkeys = numpy.array(
            [i.encode('ascii', 'ignore') for i in data.tile])
        magnitudes = numpy.array(data.keys()[3:]).astype(float)
        rates = data[magnitudes.astype(str)].to_numpy()

        region = QuadtreeGrid2D.from_quadkeys(
            quadkeys.astype(str), magnitudes=magnitudes)
        region.get_cell_area()

        return rates, region, magnitudes

    @staticmethod
    def csv(filename):
        def is_mag(num):
            try:
                m = float(num)
                if m > -1 and m < 12.:
                    return True
                else:
                    return False
            except ValueError:
                return False

        with open(filename, 'r') as file_:
            line = file_.readline()
            if len(line.split(',')) > 3:
                sep = ','
            else:
                sep = ' '

        if 'tile' in line:
            rates, region, magnitudes = ForecastParsers.quadtree(filename)
            return rates, region, magnitudes

        data = pandas.read_csv(filename, header=0, sep=sep, escapechar='#',
                               skipinitialspace=True)

        data.columns = [i.strip() for i in data.columns]
        magnitudes = numpy.array([float(i) for i in data.columns if is_mag(i)])
        rates = data[[i for i in data.columns if is_mag(i)]].to_numpy()
        all_polys = data[
            ['lon_min', 'lon_max', 'lat_min', 'lat_max']].to_numpy()
        bboxes = numpy.array(
            [tuple(itertools.product(bbox[:2], bbox[2:])) for bbox in
             all_polys])
        dh = float(all_polys[0, 3] - all_polys[0, 2])

        try:
            poly_mask = data['mask']
        except:
            poly_mask = numpy.ones(bboxes.shape[0])

        region = CartesianGrid2D(
            [Polygon(bbox) for bbox in bboxes], dh, mask=poly_mask)

        return rates, region, magnitudes

    @staticmethod
    def hdf5(filename):
        start = time.process_time()

        with h5py.File(filename, 'r') as db:
            rates = db['rates'][:]
            magnitudes = db['magnitudes'][:]

            if 'quadkeys' in db.keys():
                region = QuadtreeGrid2D.from_quadkeys(
                    db['quadkeys'][:].astype(str), magnitudes=magnitudes)
                region.get_cell_area()
            else:
                dh = db['dh'][:][0]
                bboxes = db['bboxes'][:]
                poly_mask = db['poly_mask'][:]
                region = CartesianGrid2D(
                    [Polygon(bbox) for bbox in bboxes], dh, mask=poly_mask)

        print(f'Loading from hdf5 took: {time.process_time() - start}')

        return rates, region, magnitudes


class HDF5Serializer:

    @staticmethod
    def grid2hdf5(filename, ext, hdf5_filename=None):
        start = time.process_time()
        parser = getattr(ForecastParsers, ext)

        rates, region, mag = parser(filename)
        bboxes = numpy.array([i.points for i in region.polygons])

        with h5py.File(hdf5_filename, 'a') as hf:
            hf.require_dataset('rates', shape=rates.shape, dtype=float)
            hf['rates'][:] = rates
            hf.require_dataset('magnitudes', shape=mag.shape,
                               dtype=float)
            hf['magnitudes'][:] = mag
            hf.require_dataset('bboxes', shape=bboxes.shape, dtype=float)
            hf['bboxes'][:] = bboxes
            hf.require_dataset('dh', shape=(1,), dtype=float)
            hf['dh'][:] = region.dh
            hf.require_dataset('poly_mask', shape=region.poly_mask.shape,
                               dtype=float)
            hf['poly_mask'][:] = region.poly_mask
        print(f'Serializing from csv took: {time.process_time() - start}')


def serialize():
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", help="format")
    parser.add_argument("--filename", help="Model forecast name")
    args = parser.parse_args()

    if args.format == 'quadtree':
        HDF5Serializer.quadtree(args.filename)
    if args.format == 'dat':
        HDF5Serializer.dat(args.filename)
    if args.format == 'csep' or args.format == 'csv':
        HDF5Serializer.csv(args.filename)
    if args.format == 'xml':
        HDF5Serializer.xml(args.filename)


if __name__ == '__main__':
    serialize()
