import copy
import tempfile
import unittest
import os
import numpy
import csep.utils.datasets
import pytest

from floatcsep.utils import readers


class TestForecastParsers(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls._path = os.path.dirname(__file__)
        cls._dir = os.path.join(cls._path, "../artifacts", "models")

    @classmethod
    def tearDownClass(cls) -> None:
        fname = os.path.join(cls._dir, "model.hdf5")
        if os.path.isfile(fname):
            os.remove(fname)

    def test_parse_csv(self):
        fname = os.path.join(self._dir, "model.csv")
        numpy.seterr(all="ignore")
        rates, region, mags = readers.ForecastParsers.csv(fname)

        rts = numpy.array([[1.0, 0.1], [1.0, 0.1], [1.0, 0.1], [1.0, 0.1]])
        orgs = numpy.array([[0.0, 0.0], [0.1, 0], [0.0, 0.1], [0.1, 0.1]])
        poly_2 = numpy.array([[0.0, 0.1], [0.0, 0.2], [0.1, 0.2], [0.1, 0.1]])

        numpy.testing.assert_allclose(rts, rates)
        numpy.testing.assert_allclose(orgs, region.origins())
        numpy.testing.assert_almost_equal(0.1, region.dh)
        numpy.testing.assert_allclose([5.0, 5.1], mags)
        numpy.testing.assert_allclose(poly_2, region.polygons[2].points)

    def test_parse_dat(self):
        fname = csep.utils.datasets.helmstetter_mainshock_fname
        rates, region, mags = readers.ForecastParsers.dat(fname)
        forecast = csep.load_gridded_forecast(fname)

        self.assertEqual(forecast.region, region)
        numpy.testing.assert_allclose(forecast.magnitudes, mags)
        numpy.testing.assert_allclose(forecast.data, rates)

    def test_parse_csv_qtree(self):
        fname = os.path.join(self._dir, "qtree", "TEAM=N10L11.csv")
        numpy.seterr(all="ignore")

        rates, region, mags = readers.ForecastParsers.csv(fname)

        poly = numpy.array(
            [[-180.0, 66.51326], [-180.0, 79.171335], [-135.0, 79.171335], [-135.0, 66.51326]]
        )

        numpy.testing.assert_allclose(115.96694121688556, rates.sum())
        numpy.testing.assert_allclose([-177.1875, 51.179343], region.origins()[123])
        self.assertEqual(8089, rates.shape[0])
        numpy.testing.assert_allclose(poly, region.polygons[2].points)

        rates2, region2, mags2 = readers.ForecastParsers.quadtree(fname)
        numpy.testing.assert_allclose(rates, rates2)
        numpy.testing.assert_allclose(
            [i.points for i in region.polygons], [i.points for i in region2.polygons]
        )
        numpy.testing.assert_allclose(mags, mags2)

    def test_parse_xml(self):
        fname = os.path.join(
            self._path,
            "../../tutorials",
            "case_e",
            "models",
            "gulia-wiemer.ALM.italy.10yr.2010-01-01.xml",
        )

        numpy.seterr(all="ignore")
        rates, region, mags = readers.ForecastParsers.xml(fname)

        orgs = numpy.array([12.6, 38.3])
        poly = numpy.array([[12.6, 38.3], [12.6, 38.4], [12.7, 38.4], [12.7, 38.3]])
        mags_ = numpy.arange(5, 9.05, 0.1)

        numpy.testing.assert_allclose(16.185424321406536, rates.sum())
        numpy.testing.assert_allclose(2.8488248e-05, rates[4329, 15])
        numpy.testing.assert_allclose(orgs, region.origins()[4329])
        numpy.testing.assert_almost_equal(0.1, region.dh)
        numpy.testing.assert_allclose(mags_, mags)
        numpy.testing.assert_allclose(poly, region.polygons[4329].points)

    def test_serialize_hdf5(self):
        numpy.seterr(all="ignore")
        fname = os.path.join(self._dir, "model.csv")
        rates, region, mags = readers.ForecastParsers.csv(fname)

        fname_db = os.path.join(self._dir, "model.hdf5")
        readers.HDF5Serializer.grid2hdf5(rates, region, mags, hdf5_filename=fname_db)
        self.assertTrue(os.path.isfile(fname_db))
        size = os.path.getsize(fname_db)
        self.assertLessEqual(4500, size)
        self.assertGreaterEqual(5000, size)

    def test_parse_hdf5(self):
        fname = os.path.join(self._dir, "model_h5.hdf5")
        rates, region, mags = readers.ForecastParsers.hdf5(fname)

        orgs = numpy.array([[0.0, 0.0], [0.1, 0], [0.0, 0.1], [0.1, 0.1]])
        poly_3 = numpy.array([[0.1, 0.1], [0.1, 0.2], [0.2, 0.2], [0.2, 0.1]])
        numpy.testing.assert_allclose(4.4, rates.sum())
        numpy.testing.assert_allclose(orgs, region.origins())
        numpy.testing.assert_almost_equal(0.1, region.dh)
        numpy.testing.assert_allclose([5.0, 5.1], mags)
        numpy.testing.assert_allclose(poly_3, region.polygons[3].points)

    def test_checkformat_xml(self):
        def save(xml_list):
            name_ = os.path.join(tempfile.tempdir, "tmpxml.xml")
            with open(name_, "w") as file_:
                for i in xml_list:
                    file_.write(i + "\n")
            return name_

        forecast_xml = [
            "<?xml version='1.0' encoding='UTF-8'?>",
            "<CSEPForecast xmlns=''>",
            "<forecastData publicID=''>",
            "<depthLayer max='30.0' min='0.0'>",
            "<cell lat='0.1' lon='0.1'>",
            "<bin m='5.0' mk='1'>1.6773966e-008</bin>",
            "</cell>",
            "</depthLayer>",
            "</forecastData>",
            "</CSEPForecast>",
        ]

        filename = save(forecast_xml)

        try:
            readers.check_format(filename, fmt="xml")
        except (IndentationError, IndexError, KeyError):
            self.fail("Format check failed")

        xml_fail = copy.deepcopy(forecast_xml)
        xml_fail[3] = "<depthayer max='30.0' min='0.0'>"
        xml_fail[-3] = "</depthayer>"
        filename = save(xml_fail)
        with pytest.raises(LookupError):
            readers.check_format(filename, fmt="xml")

        xml_fail = copy.deepcopy(forecast_xml)
        xml_fail[4] = "<cell Lat='0.1' Lon='0.1'>"
        filename = save(xml_fail)
        with pytest.raises(KeyError):
            readers.check_format(filename, fmt="xml")

        xml_fail = copy.deepcopy(forecast_xml)
        xml_fail[5] = "<mbin m='5.0'>1.6773966e-008</mbin>"
        filename = save(xml_fail)
        with pytest.raises(LookupError):
            readers.check_format(filename, fmt="xml")

        xml_fail = copy.deepcopy(forecast_xml)
        xml_fail[5] = "<bin a='5.0'>1.6773966e-008</bin>"
        filename = save(xml_fail)
        with pytest.raises(KeyError):
            readers.check_format(filename, fmt="xml")

        xml_fail = copy.deepcopy(forecast_xml)
        xml_fail[2] = ""
        xml_fail[-2] = ""
        filename = save(xml_fail)
        with pytest.raises(IndentationError):
            readers.check_format(filename)

    # @classmethod
    # def tearDownClass(cls) -> None:
    #     fname = os.path.join(cls._dir, 'model.hdf5')
    #
    #     if os.path.isfile(fname):
    #         os.remove(fname)
