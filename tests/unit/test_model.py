import os.path
import tempfile
from datetime import datetime
from csep.utils.time_utils import decimal_year

import numpy.testing

from fecsep.utils import timewindow_str
from unittest import TestCase
import filecmp

from fecsep.model import Model
import shutil


class TestModel(TestCase):
    _path = os.path.dirname(__file__)

    def assertEqualModel(self, exp_a, exp_b):
        pass

    def test_init_from_filesystem(self):
        name = 'mock'
        path = os.path.join(TestModel._path, 'artifacts',
                            'models', 'model.csv')
        model = Model(name, path)
        self.assertEqual(name, model.name)
        self.assertEqual(path, model.path)
        self.assertEqual('csv', model.format)
        self.assertEqual(os.path.dirname(path), model._dir)
        self.assertTrue(model.rm_db())

    def test_init_from_zenodo(self):
        name = 'mock_zenodo'
        filename_ = 'TEAM=N10L11.csv'
        dir_ = os.path.join(tempfile.tempdir, 'mock')
        if os.path.isdir(dir_):
            shutil.rmtree(dir_)
        path_ = os.path.join(dir_, filename_)
        zenodo_id = 6289795
        model_a = Model(name, path_, zenodo_id=zenodo_id)

        path = os.path.join(TestModel._path, 'artifacts', 'models',
                            'qtree', filename_)
        model_b = Model(name, path)

        self.assertEqual('csv', model_a.format)
        self.assertEqual('csv', model_b.format)
        self.assertEqual(model_a.name, model_b.name)
        self.assertTrue(filecmp.cmp(model_a.path, model_b.path))
        self.assertTrue(model_a.rm_db())
        self.assertTrue(model_b.rm_db())

    def test_init_from_git(self):
        name = 'mock_git'
        _dir = 'template_'
        path_ = os.path.join(tempfile.tempdir, _dir)
        giturl = 'https://git.gfz-potsdam.de/csep-group/rise_italy_experiment/' \
                 'models/template.git'
        model_a = Model(name, path_, giturl=giturl)
        path = os.path.join(TestModel._path, 'artifacts', 'models',
                            'template')
        model_b = Model(name, path)
        self.assertEqual(model_a.name, model_b.name)
        self.assertEqual('src', model_b.format)
        dircmp = filecmp.dircmp(model_a.path, model_b.path).common
        self.assertGreater(len(dircmp), 8)
        shutil.rmtree(path_)

    def test_from_dict(self):
        name = 'mock'
        path = os.path.join(TestModel._path, 'artifacts',
                            'models', 'model.csv')

        dict_ = {'mock':
                     {'path': os.path.join(TestModel._path, 'artifacts',
                                           'models', 'model.csv'),
                      'forecast_unit': 5,
                      'authors': ['Darwin, C.', 'Bell, J.', 'Et, Al.'],
                      'doi': '10.1010/10101010',
                      'giturl': 'should not be accessed, bc filesystem exists',
                      'zenodo_id': 'should not be accessed, bc filesystem exists'
                      }}

        model = Model.from_dict(dict_)
        self.assertEqual(name, model.name)
        self.assertEqual(path, model.path)
        self.assertEqual('csv', model.format)
        self.assertEqual(os.path.dirname(path), model._dir)
        self.assertTrue(model.rm_db())

    def test_create_forecast_ti(self):
        name = 'mock'
        path = os.path.join(os.path.dirname(__file__),
                            'artifacts', 'models', 'model.csv')
        rates = numpy.array([[1., 0.1],
                             [1., 0.1],
                             [1., 0.1],
                             [1., 0.1]])

        model = Model(name, path)
        start = datetime(2020, 1, 1)
        end = datetime(2021, 1, 1)
        model.create_forecast(start, end)
        tstr = timewindow_str([start, end])

        self.assertIn(tstr, model.forecasts)
        self.assertEqual(f'{name}_{tstr}', model.forecasts[tstr].name)
        numpy.testing.assert_equal(4.4, model.forecasts[tstr].event_count)
        numpy.testing.assert_allclose(rates, model.forecasts[tstr].data)

        end = datetime(2020, 3, 1)
        model.create_forecast(start, end)
        tstr = timewindow_str([start, end])

        self.assertIn(tstr, model.forecasts)
        self.assertEqual(f'{name}_{tstr}', model.forecasts[tstr].name)
        numpy.testing.assert_almost_equal(0.72131147541031171,
                                          model.forecasts[tstr].event_count)
        numpy.testing.assert_allclose(rates * (decimal_year(end) -
                                               decimal_year(start)),
                                      model.forecasts[tstr].data)
        self.assertTrue(model.rm_db())
