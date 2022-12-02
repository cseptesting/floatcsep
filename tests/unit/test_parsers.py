import unittest
import os
import numpy
import csep.utils.datasets
from fecsep import readers


class TestForecastParsers(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls._path = os.path.dirname(__file__)
        cls._dir = os.path.join(cls._path, 'artifacts', 'models')

    def test_parse_csv(self):
        fname = os.path.join(self._dir, 'model.csv')
        numpy.seterr(all="ignore")
        rates, region, mags = readers.ForecastParsers.csv(fname)

        rts = numpy.array([[1., 0.1],
                           [1., 0.1],
                           [1., 0.1],
                           [1., 0.1]])
        orgs = numpy.array([[0., 0.],
                            [0.1, 0],
                            [0., 0.1],
                            [0.1, 0.1]])
        poly_2 = numpy.array([[0., 0.1],
                              [0., 0.2],
                              [0.1, 0.1],
                              [0.1, 0.2]])

        numpy.testing.assert_allclose(rts, rates)
        numpy.testing.assert_allclose(orgs, region.origins())
        numpy.testing.assert_almost_equal(0.1, region.dh)
        numpy.testing.assert_allclose([5., 5.1], mags)
        numpy.testing.assert_allclose(poly_2, region.polygons[2].points)

    def test_parse_dat(self):
        fname = csep.utils.datasets.helmstetter_mainshock_fname
        rates, region, mags = readers.ForecastParsers.dat(fname)
        forecast = csep.load_gridded_forecast(fname)

        self.assertEqual(forecast.region, region)
        numpy.testing.assert_allclose(forecast.magnitudes, mags)
        numpy.testing.assert_allclose(forecast.data, rates)

    def test_parse_csv_qtree(self):
        fname = os.path.join(self._dir, 'qtree', 'TEAM=N10L11.csv')
        numpy.seterr(all="ignore")

        rates, region, mags = readers.ForecastParsers.csv(fname)

        poly = numpy.array([[-180., 66.51326],
                            [-180., 79.171335],
                            [-135., 79.171335],
                            [-135., 66.51326]])

        numpy.testing.assert_allclose(115.96694121688556, rates.sum())
        numpy.testing.assert_allclose([-177.1875, 51.179343],
                                      region.origins()[123])
        self.assertEqual(8089, rates.shape[0])
        numpy.testing.assert_allclose(poly, region.polygons[2].points)

        rates2, region2, mags2 = readers.ForecastParsers.quadtree(fname)
        numpy.testing.assert_allclose(rates, rates2)
        numpy.testing.assert_allclose([i.points for i in region.polygons],
                                      [i.points for i in region2.polygons])
        numpy.testing.assert_allclose(mags, mags2)

    def test_parse_xml(self):
        fname = os.path.join(self._path, '../../examples', 'case_e',
                             'models',
                             'gulia-wiemer.ALM.italy.10yr.2010-01-01.xml')

        numpy.seterr(all="ignore")
        rates, region, mags = readers.ForecastParsers.xml(fname)

        orgs = numpy.array([12.6, 38.3])
        poly = numpy.array([[12.6, 38.3],
                            [12.6, 38.4],
                            [12.7, 38.3],
                            [12.7, 38.4]])
        mags_ = numpy.arange(5, 9.05, 0.1)

        numpy.testing.assert_allclose(16.185424321406536, rates.sum())
        numpy.testing.assert_allclose(2.8488248e-05, rates[4329, 15])
        numpy.testing.assert_allclose(orgs, region.origins()[4329])
        numpy.testing.assert_almost_equal(0.1, region.dh)
        numpy.testing.assert_allclose(mags_, mags)
        numpy.testing.assert_allclose(poly, region.polygons[4329].points)

    def test_serialize_hdf5(self):
        numpy.seterr(all="ignore")
        fname = os.path.join(self._dir, 'model.csv')
        fname_db = os.path.join(self._dir, 'model.hdf5')
        readers.HDF5Serializer.grid2hdf5(fname, 'csv', fname_db)
        self.assertTrue(os.path.isfile(fname_db))
        size = os.path.getsize(fname_db)
        self.assertEqual(4640, size)

    def test_parse_hdf5(self):
        fname = os.path.join(self._dir, 'model_h5.hdf5')
        rates, region, mags = readers.ForecastParsers.hdf5(fname)

        orgs = numpy.array([[0., 0.],
                            [0.1, 0],
                            [0., 0.1],
                            [0.1, 0.1]])
        poly_3 = numpy.array([[0.1, 0.1],
                              [0.1, 0.2],
                              [0.2, 0.1],
                              [0.2, 0.2]])
        numpy.testing.assert_allclose(4.4, rates.sum())
        numpy.testing.assert_allclose(orgs, region.origins())
        numpy.testing.assert_almost_equal(0.1, region.dh)
        numpy.testing.assert_allclose([5., 5.1], mags)
        numpy.testing.assert_allclose(poly_3, region.polygons[3].points)

    @classmethod
    def tearDownClass(cls) -> None:
        fname = os.path.join(cls._dir, 'model.hdf5')

        if os.path.isfile(fname):
            os.remove(fname)
