import os.path
import numpy.testing
from datetime import datetime
from unittest import TestCase
from floatcsep.model import Model


class TestModelFromFile(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        path = os.path.dirname(__file__)
        cls._path = path
        cls._dir = os.path.join(path, '../', 'artifacts', 'models')
        cls._alm_fn = os.path.join(
            path, '../../examples',
            'case_e',
            'models',
            'gulia-wiemer.ALM.italy.10yr.2010-01-01.xml'
        )

    @staticmethod
    def init_model(name, path, **kwargs):
        """ Make model without Registry """
        # model = Model.__new__(Model)
        # Model.__init__.__wrapped__(self=model, name=name,
        #                            path=path, **kwargs)
        model = Model(name, path, **kwargs)
        # ext = os.path.splitext(path)[-1][1:]
        # model.fmt = ext
        # model.dir = os.path.dirname(path) if ext else path

        return model

    def test_forecast_ti_from_csv(self):
        """ Parses forecast from csv file """
        name = 'mock'
        fname = os.path.join(self._dir, 'model.csv')
        model = self.init_model(name, fname)
        start = datetime(1900, 1, 1)
        end = datetime(2000, 1, 1)
        model.stage([[start, end]])
        model.forecast_from_file(start, end)
        numpy.testing.assert_almost_equal(440., model.forecasts[
            '1900-01-01_2000-01-01'].data.sum())

    def test_forecast_ti_from_xml(self):
        """ Parses forecast from XML file """

        name = 'ALM'
        fname = self._alm_fn
        numpy.seterr(all="ignore")
        model = Model(name, fname)
        start = datetime(1900, 1, 1)
        end = datetime(2000, 1, 1)
        model.stage([[start, end]])
        model.forecast_from_file(start, end)

        numpy.testing.assert_almost_equal(1618.5424321406535, model.forecasts[
            '1900-01-01_2000-01-01'].data.sum())

    def test_forecast_ti_from_xml2hdf5(self):
        """ reads from xml, drops to db, makes forecast from db """
        name = 'ALM'
        fname = self._alm_fn
        numpy.seterr(all="ignore")

        model = self.init_model(name=name, path=fname, use_db=True)
        start = datetime(1900, 1, 1)
        end = datetime(2000, 1, 1)
        model.stage([[start, end]])
        model.forecast_from_file(start, end)

        numpy.testing.assert_almost_equal(1618.5424321406535, model.forecasts[
            '1900-01-01_2000-01-01'].data.sum())

    def test_forecast_ti_from_hdf5(self):
        """ reads from hdf5, scale in runtime """
        name = 'mock'
        fname = os.path.join(self._dir, 'model_h5.hdf5')
        model = self.init_model(name=name, path=fname, use_db=True)
        model.stage()

        start = datetime(2020, 1, 1)
        end = datetime(2023, 1, 1)
        model.stage([[start, end]])
        model.forecast_from_file(start, end)
        numpy.testing.assert_almost_equal(13.2, model.forecasts[
            '2020-01-01_2023-01-01'].data.sum())

    @classmethod
    def tearDownClass(cls) -> None:
        alm_db = os.path.join(cls._path, '../../examples', 'case_e',
                              'models',
                              'gulia-wiemer.ALM.italy.10yr.2010-01-01.hdf5')
        if os.path.isfile(alm_db):
            os.remove(alm_db)
