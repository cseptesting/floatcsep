import os.path
import tempfile
from unittest import TestCase
import filecmp
from fecsep.core import Model
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
        self.assertEqual(os.path.dirname(path), model._dir)

    def test_init_from_zenodo(self):
        name = 'mock_zenodo'
        filename_ = 'TEAM=N10L11.csv'
        path_ = os.path.join(tempfile.tempdir, filename_)
        zenodo_id = 6289795
        model_a = Model(name, path_, zenodo_id=zenodo_id)

        path = os.path.join(TestModel._path, 'artifacts', 'models',
                            'qtree', filename_)
        model_b = Model(name, path)

        self.assertEqual(model_a.name, model_b.name)
        self.assertTrue(filecmp.cmp(model_a.path, model_b.path))

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
        dircmp = filecmp.dircmp(model_a.path, model_b.path).common
        self.assertGreater(len(dircmp), 8)
        shutil.rmtree(path_)
