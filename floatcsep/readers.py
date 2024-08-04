import argparse
import logging
import os.path
import time
import xml.etree.ElementTree as eTree

import h5py
import numpy
import pandas
from csep.core.regions import QuadtreeGrid2D, CartesianGrid2D
from csep.models import Polygon

log = logging.getLogger(__name__)


class ForecastParsers:

    @staticmethod
    def dat(filename):
        data = numpy.loadtxt(filename)
        all_polys = data[:, :4]
        all_poly_mask = data[:, -1]
        sorted_idx = numpy.sort(
            numpy.unique(all_polys, return_index=True, axis=0)[1], kind="stable"
        )
        unique_poly = all_polys[sorted_idx]
        poly_mask = all_poly_mask[sorted_idx]
        all_mws = data[:, -4]
        sorted_idx = numpy.sort(numpy.unique(all_mws, return_index=True)[1], kind="stable")
        mws = all_mws[sorted_idx]
        bboxes = [((i[0], i[2]), (i[0], i[3]), (i[1], i[3]), (i[1], i[2])) for i in unique_poly]
        dh = float(unique_poly[0, 3] - unique_poly[0, 2])

        n_mag_bins = len(mws)
        rates = data[:, -2].reshape(len(bboxes), n_mag_bins)

        region = CartesianGrid2D([Polygon(bbox) for bbox in bboxes], dh, mask=poly_mask)

        return rates, region, mws

    @staticmethod
    def xml(filename, verbose=False):
        tree = eTree.parse(filename)
        root = tree.getroot()
        metadata = {}
        data_ijm = []
        m_bins = []
        cells = []
        cell_dim = {}
        for k, children in enumerate(list(root[0])):
            if "modelName" in children.tag:
                name_xml = children.text
                metadata["name"] = name_xml
            elif "author" in children.tag:
                author_xml = children.text
                metadata["author"] = author_xml
            elif "forecastStartDate" in children.tag:
                start_date = children.text.replace("Z", "")
                metadata["forecastStartDate"] = start_date
            elif "forecastEndDate" in children.tag:
                end_date = children.text.replace("Z", "")
                metadata["forecastEndDate"] = end_date
            elif "defaultMagBinDimension" in children.tag:
                m_bin_width = float(children.text)
                metadata["defaultMagBinDimension"] = m_bin_width
            elif "lastMagBinOpen" in children.tag:
                lastmbin = float(children.text)
                metadata["lastMagBinOpen"] = lastmbin
            elif "defaultCellDimension" in children.tag:
                cell_dim = {i[0]: float(i[1]) for i in children.attrib.items()}
                metadata["defaultCellDimension"] = cell_dim
            elif "depthLayer" in children.tag:
                depth = {i[0]: float(i[1]) for i in root[0][k].attrib.items()}
                cells = root[0][k]
                metadata["depthLayer"] = depth
        if verbose:
            log.debug(f"Forecast with metadata:\n{metadata}")

        for cell in cells:
            cell_data = []
            m_cell_bins = []
            for i, m in enumerate(cell.iter()):
                if i == 0:
                    cell_data.extend([float(m.attrib["lon"]), float(m.attrib["lat"])])
                else:
                    cell_data.append(float(m.text))
                    m_cell_bins.append(float(m.attrib["m"]))
            data_ijm.append(cell_data)
            m_bins.append(m_cell_bins)
        try:
            data_ijm = numpy.array(data_ijm)
            m_bins = numpy.array(m_bins)
        except (TypeError, ValueError):
            raise Exception("Data is not square")

        magnitudes = m_bins[0, :]
        rates = data_ijm[:, -len(magnitudes) :]
        all_polys = numpy.vstack(
            (
                data_ijm[:, 0] - cell_dim["lonRange"] / 2.0,
                data_ijm[:, 0] + cell_dim["lonRange"] / 2.0,
                data_ijm[:, 1] - cell_dim["latRange"] / 2.0,
                data_ijm[:, 1] + cell_dim["latRange"] / 2.0,
            )
        ).T
        bboxes = [((i[0], i[2]), (i[0], i[3]), (i[1], i[3]), (i[1], i[2])) for i in all_polys]
        dh = float(all_polys[0, 3] - all_polys[0, 2])
        poly_mask = numpy.ones(len(bboxes))

        region = CartesianGrid2D([Polygon(bbox) for bbox in bboxes], dh, mask=poly_mask)

        return rates, region, magnitudes

    @staticmethod
    def quadtree(filename):
        with open(filename, "r") as file_:
            qt_header = file_.readline().split(",")
            formats = [str]
            for i in range(len(qt_header) - 1):
                formats.append(float)

        qt_formats = {i: j for i, j in zip(qt_header, formats)}
        data = pandas.read_csv(filename, header=0, dtype=qt_formats)

        quadkeys = numpy.array([i.encode("ascii", "ignore") for i in data.tile])
        magnitudes = numpy.array(data.keys()[3:]).astype(float)
        rates = data[magnitudes.astype(str)].to_numpy()

        region = QuadtreeGrid2D.from_quadkeys(quadkeys.astype(str), magnitudes=magnitudes)
        region.get_cell_area()

        return rates, region, magnitudes

    @staticmethod
    def csv(filename):
        def is_mag(num):
            try:
                m = float(num)
                if -1 < m < 12.0:
                    return True
                else:
                    return False
            except ValueError:
                return False

        with open(filename, "r") as file_:
            line = file_.readline()
            if len(line.split(",")) > 3:
                sep = ","
            else:
                sep = " "

        if "tile" in line:
            rates, region, magnitudes = ForecastParsers.quadtree(filename)
            return rates, region, magnitudes

        data = pandas.read_csv(
            filename, header=0, sep=sep, escapechar="#", skipinitialspace=True
        )

        data.columns = [i.strip() for i in data.columns]
        magnitudes = numpy.array([float(i) for i in data.columns if is_mag(i)])
        rates = data[[i for i in data.columns if is_mag(i)]].to_numpy()
        all_polys = data[["lon_min", "lon_max", "lat_min", "lat_max"]].to_numpy()
        bboxes = [((i[0], i[2]), (i[0], i[3]), (i[1], i[3]), (i[1], i[2])) for i in all_polys]
        dh = float(all_polys[0, 3] - all_polys[0, 2])

        try:
            poly_mask = data["mask"]
        except KeyError:
            poly_mask = numpy.ones(len(bboxes))

        region = CartesianGrid2D([Polygon(bbox) for bbox in bboxes], dh, mask=poly_mask)

        return rates, region, magnitudes

    @staticmethod
    def hdf5(filename, group=""):
        start = time.process_time()

        with h5py.File(filename, "r") as db:
            rates = db[f"{group}/rates"][:]
            magnitudes = db[f"{group}/magnitudes"][:]

            if "quadkeys" in db.keys():
                region = QuadtreeGrid2D.from_quadkeys(
                    db[f"{group}/quadkeys"][:].astype(str), magnitudes=magnitudes
                )
                region.get_cell_area()
            else:
                dh = db[f"{group}/dh"][:][0]
                bboxes = db[f"{group}/bboxes"][:]
                poly_mask = db[f"{group}/poly_mask"][:]
                region = CartesianGrid2D([Polygon(bbox) for bbox in bboxes], dh, mask=poly_mask)

        log.debug(f"Loading from hdf5 {filename} took:" f" {time.process_time() - start:.2f}")

        return rates, region, magnitudes


class HDF5Serializer:
    @staticmethod
    def grid2hdf5(rates, region, mag, grp="", hdf5_filename=None, **kwargs):
        start = time.process_time()

        bboxes = numpy.array([i.points for i in region.polygons])

        with h5py.File(hdf5_filename, "a") as hfile:

            hfile.require_dataset(f"{grp}/rates", shape=rates.shape, dtype=float)
            hfile[f"{grp}/rates"][:] = rates
            hfile.require_dataset(f"{grp}/magnitudes", shape=mag.shape, dtype=float)
            hfile[f"{grp}/magnitudes"][:] = mag
            hfile.require_dataset(f"{grp}/bboxes", shape=bboxes.shape, dtype=float)
            hfile[f"{grp}/bboxes"][:] = bboxes
            hfile.require_dataset(f"{grp}/dh", shape=(1,), dtype=float)
            try:
                hfile[f"{grp}/dh"][:] = region.dh
            except AttributeError:
                raise AttributeError(
                    "Quadtree can not be dropped to HDF5"
                    "(not needed, because file is already"
                    " low sized"
                )
            hfile.require_dataset(f"{grp}/poly_mask", shape=region.poly_mask.shape, dtype=float)
            hfile[f"{grp}/poly_mask"][:] = region.poly_mask

            if kwargs:
                for key, v in kwargs.items():
                    if isinstance(v, (float, int, str)):
                        dtype = type(v)
                        shape = (1,)
                    elif isinstance(v, numpy.ndarray):
                        shape = v.shape
                        dtype = v.dtype
                    else:
                        shape = len(v)
                        dtype = type(v[0])
                    hfile.require_dataset(f"{grp}/{key}", shape=shape, dtype=dtype)
                    hfile[f"{grp}/{key}"][:] = v

        log.debug(f"Storing to hdf5 {hdf5_filename} took:" f"{time.process_time() - start:2f}")


def check_format(filename, fmt=None, func=None):
    if fmt is None:
        fmt = os.path.splitext(filename)[-1][1:]

    if fmt == "xml":
        max_lines = 40
        bin_ = False
        with open(filename, "r") as f_:
            for i in range(max_lines):
                line_ = f_.readline()
                if "<bin" in line_ and "</bin>" in line_:
                    bin_ = True

        error_msg = (
            "File does not specify rates per magnitude bin."
            " Example correct format:\n <cell lat='0.1'"
            " lon'0.1'>\n <bin m='5.0'>1.0e-1</bin>\n"
            "<bin m='5.1'>1.0e-1</bin>\n </cell>"
        )
        if not bin_:
            raise LookupError(error_msg)
        tree = eTree.parse(filename)
        root = tree.getroot()
        index = False

        if "forecastData" not in root[0].tag:
            raise IndentationError(
                'Attribute "forecastData" is not found at '
                'the correct tree indentation level (1)"'
            )

        for i, j in enumerate(list(root[0])):
            if "depthLayer" in j.tag:
                index = i

        if isinstance(index, int) and (index is not False):
            cell_keys = list(root[0][index][0].attrib.keys())
            bin_ = root[0][index][0][0].attrib

            if "lat" not in cell_keys or "lon" not in cell_keys:
                raise KeyError(error_msg)
            if "m" not in bin_:
                raise KeyError(error_msg)

        else:
            raise LookupError("Attribute 'depthLayer' not present in" " 'forecastData' node")

    elif fmt == "csv":
        pass
    elif fmt == "qtree":
        pass
    elif fmt == "dat":
        pass
    elif fmt == "hdf5":
        pass
    elif func:
        pass


def serialize():
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", help="format")
    parser.add_argument("--filename", help="Model forecast name")
    args = parser.parse_args()

    if args.format == "quadtree":
        ForecastParsers.quadtree(args.filename)
    if args.format == "dat":
        ForecastParsers.dat(args.filename)
    if args.format == "csep" or args.format == "csv":
        ForecastParsers.csv(args.filename)
    if args.format == "xml":
        ForecastParsers.xml(args.filename)


if __name__ == "__main__":
    serialize()
