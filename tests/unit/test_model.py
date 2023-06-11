import os.path
import tempfile
from datetime import datetime

import csep.core.regions
import numpy.testing

from unittest import TestCase
from unittest.mock import patch
import filecmp

from floatcsep.model import Model
from floatcsep.utils import str2timewindow
import shutil


class TestModel(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        path = os.path.dirname(__file__)
        cls._path = path
        cls._dir = os.path.normpath(
            os.path.join(path, '../artifacts', 'models'))
        cls._alm_fn = os.path.join(
            path, '../../examples',
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

    @staticmethod
    def initmodel_noreg(name, path, **kwargs):
        """ Instantiates a model without using the @register deco,
        but mocks Model.Registry() attrs"""

        ### Save for registry:
        # model = Model.__new__(Model)
        # Model.__init__.__wrapped__(self=model, name=name,
        #                            path=path, **kwargs)
        model = Model(name, path, **kwargs)
        ext = os.path.splitext(path)[-1][1:]

        return model

    def test_from_filesystem(self):
        """ init from file, check base attributes"""
        name = 'mock'
        fname = os.path.join(self._dir, 'model.csv')

        # Initialize without Registry
        model = self.initmodel_noreg(name=name, path=fname)

        self.assertEqual(name, model.name)
        self.assertEqual(fname, model.path)
        self.assertEqual('ti', model.model_class)
        self.assertEqual(1, model.forecast_unit)

    def test_from_zenodo(self):
        """ downloads model from zenodo, checks with test artifacts"""

        name = 'mock_zenodo'
        filename_ = 'datapackage.json'
        dir_ = os.path.join(tempfile.tempdir, 'mock')

        if os.path.isdir(dir_):
            shutil.rmtree(dir_)
        path_ = os.path.join(dir_, filename_)

        zenodo_id = 7096870
        # Initialize from zenodo id
        model_a = self.initmodel_noreg(name=name, path=path_,
                                       zenodo_id=zenodo_id)
        model_a.stage()

        # Initialize from the artifact files (same as downloaded)
        dir_art = os.path.join(self._path, '../artifacts', 'models',
                               'zenodo_test')
        path = os.path.join(dir_art, filename_)
        model_b = self.initmodel_noreg(name=name, path=path,
                                       zenodo_id=zenodo_id)
        model_b.stage()

        self.assertEqual(os.path.basename(model_a.path('path')),
                         os.path.basename(model_b.path('path')))
        self.assertEqual(model_a.name, model_b.name)
        self.assertTrue(filecmp.cmp(model_a.path('path'),
                                    model_b.path('path')))

    def test_fail_zenodo(self):
        name = 'mock_zenodo'
        filename_ = 'model_notreal.csv'  # File not found in repository
        dir_ = os.path.join(tempfile.tempdir, 'zenodo_notreal')
        if os.path.isdir(dir_):
            shutil.rmtree(dir_)
        path_ = os.path.join(dir_, filename_)

        # Initialize from zenodo id
        model = self.initmodel_noreg(name=name, path=path_, zenodo_id=4739912)

        with self.assertRaises(FileNotFoundError):
            model.get_source(model.zenodo_id, model.giturl)

    def test_from_git(self):
        """ clones model from git, checks with test artifacts"""
        name = 'mock_git'
        _dir = 'git_template'
        path_ = os.path.join(tempfile.tempdir, _dir)
        giturl = 'https://git.gfz-potsdam.de/csep-group/' \
                 'rise_italy_experiment/models/template.git'
        model_a = self.initmodel_noreg(name=name, path=path_,
                                       giturl=giturl)
        model_a.stage()
        path = os.path.join(self._dir, 'template')
        model_b = self.initmodel_noreg(name=name, path=path)
        model_b.stage()
        self.assertEqual(model_a.name, model_b.name)
        dircmp = filecmp.dircmp(model_a.dir, model_b.dir).common
        self.assertGreater(len(dircmp), 8)
        shutil.rmtree(path_)

    def test_fail_git(self):
        name = 'mock_git'
        filename_ = 'attr.c'
        dir_ = os.path.join(tempfile.tempdir, 'git_notreal')
        if os.path.isdir(dir_):
            shutil.rmtree(dir_)
        path_ = os.path.join(dir_, filename_)

        # Initialize from git url
        model = self.initmodel_noreg(
            name=name, path=path_,
            giturl='https://github.com/github/testrepo')

        with self.assertRaises(FileNotFoundError):
            model.get_source(model.zenodo_id, model.giturl, branch='master')

    def test_from_dict(self):
        """ test that '__init__' and 'from_dict' instantiates
        identical objets"""

        name = 'mock'
        fname = os.path.join(self._dir, 'model.csv')

        dict_ = {'mock':
                 {'model_path': fname,
                  'forecast_unit': 5,
                  'authors': ['Darwin, C.', 'Bell, J.', 'Et, Al.'],
                  'doi': '10.1010/10101010',
                  'giturl': 'should not be accessed, bc filesystem exists',
                  'zenodo_id': 'should not be accessed, bc filesystem '
                               'exists'
                  }
                 }

        # Has to be instantiated with registry
        model_a = Model.from_dict(dict_)

        # Import from normal py dict structure
        py_dict = {'name': 'mock', **dict_['mock']}
        model_b = Model.from_dict(py_dict)

        self.assertEqual(name, model_a.name)
        self.assertEqual(fname, model_a.path.path)
        self.assertEqual('csv', model_a.path.fmt)
        self.assertEqual(self._dir, model_a.dir)

        self.assertEqualModel(model_a, model_b)

        with self.assertRaises(IndexError):
            Model.from_dict({'model': 1, 'no_name': 2})
        with self.assertRaises(IndexError):
            Model.from_dict({'model_1': {'name': 'quack'},
                             'model_2': {'name': 'moo'}})

    @patch('floatcsep.model.Model.forecast_from_func')
    @patch('floatcsep.model.Model.forecast_from_file')
    def test_create_forecast(self, mock_file, mock_func):

        model = self.initmodel_noreg('mock', 'mockfile.csv')
        model.create_forecast('2020-01-01_2021-01-01')
        self.assertTrue(mock_file.called)

        model = self.initmodel_noreg('mock', 'mockbins', model_class='td')

        model.path.build_tree([str2timewindow('2020-01-01_2021-01-01')])
        model.create_forecast('2020-01-01_2021-01-01')
        self.assertTrue(mock_func.called)

    def test_forecast_from_file(self):
        """ reads from file, scale in runtime """
        _rates = numpy.array([[1., 0.1],
                              [1., 0.1]])
        _mags = numpy.array([5.0, 5.1])
        origins = numpy.array([[0., 0.],
                               [0., 1.]])
        _region = csep.core.regions.CartesianGrid2D.from_origins(origins)

        def forecast_(_):
            return _rates, _region, _mags

        name = 'mock'
        fname = os.path.join(self._dir, 'model.csv')

        with patch('floatcsep.readers.ForecastParsers.csv', forecast_):
            model = self.initmodel_noreg(name, fname)
            start = datetime(1900, 1, 1)
            end = datetime(2000, 1, 1)
            model.path.build_tree([[start, end]])
            model.forecast_from_file(start, end)
            numpy.testing.assert_almost_equal(220., model.forecasts[
                '1900-01-01_2000-01-01'].data.sum())

    def test_argprep(self):
        model_path = os.path.join(self._dir, 'td_model')
        with open(os.path.join(model_path, 'input', 'args.txt'), 'w') as args:
            args.write('start_date = foo\nend_date = bar')

        model = self.initmodel_noreg('a',
                                     model_path,
                                     func='func')
        start = datetime(2000, 1, 1)
        end = datetime(2000, 1, 2)
        model.stage([[start, end]])
        model.prepare_args(start, end)

        with open(os.path.join(model_path, 'input', 'args.txt'), 'r') as args:
            self.assertEqual(args.readline(),
                             f'start_date = {start.isoformat()}\n')
            self.assertEqual(args.readline(),
                             f'end_date = {end.isoformat()}\n')
        model.prepare_args(start, end, n_sims=400)
        with open(os.path.join(model_path, 'input', 'args.txt'), 'r') as args:
            self.assertEqual(args.readlines()[2],
                             f'n_sims = 400\n')

        model.prepare_args(start, end, n_sims=200)
        with open(os.path.join(model_path, 'input', 'args.txt'), 'r') as args:
            self.assertEqual(args.readlines()[2],
                             f'n_sims = 200\n')

    def test_get_forecast(self):

        model = self.initmodel_noreg('mock', 'mockfile.csv')
        model.forecasts = {'a': 1, 'moo': 1, 'cuack': 1}

        self.assertEqual(1, model.get_forecast('a'))
        self.assertEqual([1, 1], model.get_forecast(['moo', 'cuack']))
        with self.assertRaises(KeyError):
            model.get_forecast(['woof'])
        with self.assertRaises(ValueError):
            model.get_forecast('meaow')

    def test_todict(self):

        fname = os.path.join(self._dir, 'model.csv')
        dict_ = {
            'path': fname,
            'forecast_unit': 5,
            'authors': ['Darwin, C.', 'Bell, J.', 'Et, Al.'],
            'doi': '10.1010/10101010',
            'giturl': 'should not be accessed, bc filesystem exists',
            'zenodo_id': 'should not be accessed, bc filesystem exists',
            'model_class': 'ti'
        }
        model = self.initmodel_noreg(name='mock', **dict_)
        model_dict = model.as_dict()
        eq = True

        for k, v in dict_.items():
            if k not in list(model_dict['mock'].keys()):
                eq = False
            else:
                if v != model_dict['mock'][k]:
                    eq = False
        excl = ['path', 'giturl', 'forecast_unit']
        keys = list(model.as_dict(excluded=excl).keys())
        for i in excl:
            if i in keys and i != 'path':  # path always gets printed
                eq = False
        self.assertTrue(eq)

    def test_init_db(self):
        pass

    def test_rm_db(self):
        pass
