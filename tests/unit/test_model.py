import os.path
import tempfile
from datetime import datetime

import numpy.testing

from fecsep import readers
from unittest import TestCase
import filecmp

from fecsep.model import Model
import shutil


class TestModel(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls._path = os.path.dirname(__file__)
        cls._dir = os.path.join(cls._path, 'artifacts', 'models')
        cls._alm_fn = os.path.join(
                               cls._path, '../../examples',
                               'case_e',
                               'models',
                               'gulia-wiemer.ALM.italy.10yr.2010-01-01.xml'
                                   )

    @staticmethod
    def assertEqualModel(exp_a, exp_b):

        keys_a = list(exp_a.__dict__.keys())
        keys_b = list(exp_a.__dict__.keys())

        if keys_a != keys_b:
            raise AssertionError('Models are not equal')

        for i in keys_a:
            if not (getattr(exp_a, i) == getattr(exp_b, i)):
                raise AssertionError('Models are not equal')

    def test_from_filesystem(self):
        """ init from file, check attributes"""
        name = 'mock'
        fname = os.path.join(self._dir, 'model.csv')
        model = Model(name, fname, use_db=False)
        self.assertEqual(name, model.name)
        self.assertEqual(fname, model.path)
        self.assertEqual('ti', model._class)
        self.assertEqual('csv', model._fmt)
        self.assertEqual('file', model._src)
        self.assertEqual(self._dir, model._dir)
        self.assertEqual(1, model.forecast_unit)

        self.assertIs(None, model.db_func)

    def test_from_filesystem_DB(self):
        """ init from file, check for hdf5 db atrrs """
        name = 'mock'
        fname = os.path.join(self._dir, 'model.csv')
        model = Model(name, fname, use_db=True)

        self.assertTrue(os.path.isfile(model.dbpath))
        model.rm_db()
        self.assertFalse(os.path.isfile(model.dbpath))

    def test_from_zenodo(self):
        """ downloads model from zenodo """

        name = 'mock_zenodo'
        filename_ = 'TEAM=N10L11.csv'
        dir_ = os.path.join(tempfile.tempdir, 'mock')
        if os.path.isdir(dir_):
            shutil.rmtree(dir_)
        path_ = os.path.join(dir_, filename_)
        zenodo_id = 6289795

        # Initialize from zenodo id
        model_a = Model(name, path_, zenodo_id=zenodo_id)

        # Initialize from the files downloaded
        path = os.path.join(self._path, 'artifacts', 'models',
                            'qtree', filename_)
        model_b = Model(name, path, zenodo_id=6289795)

        self.assertEqual('csv', model_a._fmt)
        self.assertEqual('csv', model_b._fmt)
        self.assertEqual(model_a.name, model_b.name)
        self.assertTrue(filecmp.cmp(model_a.path, model_b.path))
        self.assertFalse(model_a.dbpath)

    def test_from_git(self):
        name = 'mock_git'
        _dir = 'template_'
        path_ = os.path.join(tempfile.tempdir, _dir)
        giturl = 'https://git.gfz-potsdam.de/csep-group/' \
                 'rise_italy_experiment/models/template.git'
        model_a = Model(name, path_, giturl=giturl)
        path = os.path.join(self._dir, 'template')
        model_b = Model(name, path)
        self.assertEqual(model_a.name, model_b.name)
        self.assertEqual('bin', model_b._src)
        dircmp = filecmp.dircmp(model_a.path, model_b.path).common
        self.assertGreater(len(dircmp), 8)
        shutil.rmtree(path_)

    def test_from_dict(self):
        name = 'mock'
        fname = os.path.join(self._dir, 'model.csv')

        dict_ = {'mock':
                 {'path': fname,
                  'forecast_unit': 5,
                  'authors': ['Darwin, C.', 'Bell, J.', 'Et, Al.'],
                  'doi': '10.1010/10101010',
                  'giturl': 'should not be accessed, bc filesystem exists',
                  'zenodo_id': 'should not be accessed, bc filesystem exists'
                  }
                 }

        # Similar to import from YML
        model_a = Model.from_dict(dict_)

        # Import from normal py dict structure
        py_dict = {'name': 'mock', **dict_['mock']}
        model_b = Model.from_dict(py_dict)

        self.assertEqual(name, model_a.name)
        self.assertEqual(fname, model_a.path)
        self.assertEqual('csv', model_a._fmt)
        self.assertEqual('file', model_a._src)
        self.assertEqual(self._dir, model_a._dir)

        self.assertEqualModel(model_a, model_b)

    def test_create_forecasts(self):
        """ Selects TI or TD appropiately """
        pass

    def test_forecast_ti_from_file(self):
        """ reads from file, scale in runtime """
        name = 'mock'
        fname = os.path.join(self._dir, 'model.csv')
        model = Model(name, fname, use_db=False)

        start = datetime(1900, 1, 1)
        end = datetime(2000, 1, 1)
        model.forecast_from_file(start, end)

        numpy.testing.assert_almost_equal(440., model.forecasts[
            '1900-01-01_2000-01-01'].data.sum())

    def test_forecast_ti_from_xml(self):
        """ reads from file, scale in runtime """
        name = 'ALM'
        fname = self._alm_fn

        numpy.seterr(all="ignore")

        model = Model(name, fname, use_db=False)

        start = datetime(1900, 1, 1)
        end = datetime(2000, 1, 1)
        model.forecast_from_file(start, end)

        numpy.testing.assert_almost_equal(1618.5424321406535, model.forecasts[
            '1900-01-01_2000-01-01'].data.sum())

    def test_forecast_ti_from_xml2hdf5(self):
        """ reads from file, drops to db, reads from db, scale in runtime """
        name = 'ALM'
        fname = self._alm_fn

        numpy.seterr(all="ignore")

        model = Model(name, fname, use_db=True)

        start = datetime(1900, 1, 1)
        end = datetime(2000, 1, 1)
        model.forecast_from_file(start, end)

        numpy.testing.assert_almost_equal(1618.5424321406535, model.forecasts[
            '1900-01-01_2000-01-01'].data.sum())

    def test_forecast_ti_from_db(self):
        """ reads from file, scale in runtime """
        name = 'mock'
        fname = os.path.join(self._dir, 'model.csv')
        model = Model(name, fname, use_db=True)

        start = datetime(2020, 1, 1)
        end = datetime(2023, 1, 1)
        model.forecast_from_file(start, end)
        numpy.testing.assert_almost_equal(13.2, model.forecasts[
            '2020-01-01_2023-01-01'].data.sum())

        assert True

    def test_make_forecast_ti_dbrt(self):
        """ reads from generated db, scale in runtime """
        pass

    def test_make_forecast_ti_db_fs(self):
        """ reads from generated db, scale and drops to db """
        pass

    def test_todict(self):
        pass

    def test_make_forecast_td(self):
        pass

    @classmethod
    def tearDownClass(cls) -> None:

        alm_db = os.path.join(cls._path, '../../examples', 'case_e',
                              'models',
                              'gulia-wiemer.ALM.italy.10yr.2010-01-01.hdf5')
        os.remove(alm_db)
