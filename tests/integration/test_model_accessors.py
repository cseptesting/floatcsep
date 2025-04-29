import filecmp
import os.path
import shutil
import tempfile
from datetime import datetime
from unittest import TestCase
from unittest.mock import patch

import numpy.testing
import csep.core.regions
from csep.core.forecasts import GriddedForecast

from floatcsep.infrastructure.environments import EnvironmentManager
from floatcsep.model import TimeIndependentModel, TimeDependentModel
from floatcsep.utils.helpers import timewindow2str


class TestModelFromFile(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        path = os.path.dirname(__file__)
        cls._path = path
        cls._dir = os.path.join(path, "../", "artifacts", "models")
        cls._alm_fn = os.path.join(
            path,
            "../../tutorials",
            "case_e",
            "models",
            "gulia-wiemer.ALM.italy.10yr.2010-01-01.xml",
        )

    @staticmethod
    def init_model(name, path, **kwargs):
        model = TimeIndependentModel(name, path, **kwargs)
        return model

    def run_forecast_test(self, name, fname, start, end, expected_sum, use_db=False):
        model = self.init_model(name=name, path=fname, use_db=use_db)
        model.stage([[start, end]])
        model.get_forecast(timewindow2str([start, end]))
        numpy.testing.assert_almost_equal(
            expected_sum,
            model.repository.forecasts[
                f"{start.strftime('%Y-%m-%d')}_{end.strftime('%Y-%m-%d')}"
            ].data.sum(),
        )

    def test_forecast_ti_from_csv(self):
        """Parses forecast from csv file"""
        name = "mock"
        fname = os.path.join(self._dir, "model.csv")
        start = datetime(1900, 1, 1)
        end = datetime(2000, 1, 1)
        expected_sum = 440.0
        self.run_forecast_test(name, fname, start, end, expected_sum)

    def test_forecast_ti_from_xml(self):
        """Parses forecast from XML file"""
        name = "ALM"
        fname = self._alm_fn
        start = datetime(1900, 1, 1)
        end = datetime(2000, 1, 1)
        expected_sum = 1618.5424321406535
        self.run_forecast_test(name, fname, start, end, expected_sum)

    def test_forecast_ti_from_xml2hdf5(self):
        """reads from xml, drops to db, makes forecast from db"""
        name = "ALM"
        fname = self._alm_fn
        start = datetime(1900, 1, 1)
        end = datetime(2000, 1, 1)
        expected_sum = 1618.5424321406535
        self.run_forecast_test(name, fname, start, end, expected_sum, use_db=True)

    def test_forecast_ti_from_hdf5(self):
        """reads from hdf5, scale in runtime"""
        name = "mock"
        fname = os.path.join(self._dir, "model_h5.hdf5")
        start = datetime(2020, 1, 1)
        end = datetime(2023, 1, 1)
        expected_sum = 13.2
        self.run_forecast_test(name, fname, start, end, expected_sum, use_db=True)

    @classmethod
    def tearDownClass(cls) -> None:
        alm_db = os.path.join(
            cls._path,
            "../../tutorials",
            "case_e",
            "models",
            "gulia-wiemer.ALM.italy.10yr.2010-01-01.hdf5",
        )
        if os.path.isfile(alm_db):
            os.remove(alm_db)


class TestModelFromGit(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        path = os.path.dirname(__file__)
        cls._path = path
        cls._dir = os.path.normpath(os.path.join(path, "../artifacts", "models"))

    @staticmethod
    def init_model(name, model_path, **kwargs):
        """Instantiates a model without using the @register deco,
        but mocks Model.Registry() attrs"""

        model = TimeDependentModel(name=name, model_path=model_path, **kwargs)

        return model

    @patch.object(EnvironmentManager, "create_environment")
    @patch("floatcsep.infrastructure.registries.ModelFileRegistry.build_tree")
    def test_from_git(self, mock_build_tree, mock_create_environment):
        """clones model from git, checks with test artifacts"""
        mock_build_tree.return_value = None
        mock_create_environment.return_value = None
        name = "mock_git"
        _dir = "git_template"
        path_ = os.path.join(tempfile.gettempdir(), _dir)
        if os.path.exists(path_):
            shutil.rmtree(path_)
        os.makedirs(path_, exist_ok=True)
        giturl = "https://github.com/pabloitu/" "template.git"
        model_a = self.init_model(name=name, model_path=path_, giturl=giturl, force_stage=True)
        model_a.stage()

        path = os.path.join(self._dir, "template")
        model_b = self.init_model(name=name, model_path=path)
        model_b.stage()
        self.assertEqual(model_a.name, model_b.name)
        dircmp = filecmp.dircmp(model_a.registry.dir, model_b.registry.dir).common
        self.assertGreater(len(dircmp), 8)
        shutil.rmtree(path_)

    def test_fail_git(self):
        name = "mock_git"
        filename_ = "attr.c"
        dir_ = os.path.join(tempfile.tempdir, "git_notreal")
        if os.path.isdir(dir_):
            shutil.rmtree(dir_)
        path_ = os.path.join(dir_, filename_)

        # Initialize from git url
        model = self.init_model(
            name=name, model_path=path_, giturl="https://github.com/github/testrepo"
        )

        with self.assertRaises(FileNotFoundError):
            model.get_source(model.zenodo_id, model.giturl, branch="master")


class TestModelFromZenodo(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        path = os.path.dirname(__file__)
        cls._path = path
        cls._dir = os.path.normpath(os.path.join(path, "../artifacts", "models"))

    @staticmethod
    def init_model(name, model_path, **kwargs):
        """Instantiates a model without using the @register deco,
        but mocks Model.Registry() attrs"""

        model = TimeIndependentModel(name=name, model_path=model_path, **kwargs)
        return model

    @patch("floatcsep.infrastructure.registries.ModelFileRegistry.build_tree")
    def test_zenodo(self, mock_buildtree):
        """downloads model from zenodo, checks with test artifacts"""
        mock_buildtree.return_value = None

        name = "mock_zenodo"
        filename_ = "dummy.txt"
        dir_ = os.path.join(tempfile.tempdir, "mock")

        if os.path.isdir(dir_):
            shutil.rmtree(dir_)
        path_ = os.path.join(dir_, filename_)

        zenodo_id = 13117711
        # Initialize from zenodo id
        model_a = self.init_model(name=name, model_path=path_, zenodo_id=zenodo_id)
        model_a.stage()

        # Initialize from the artifact files (same as downloaded)
        dir_art = os.path.join(self._path, "../artifacts", "models", "zenodo_test")
        path = os.path.join(dir_art, filename_)
        model_b = self.init_model(name=name, model_path=path, zenodo_id=zenodo_id)
        model_b.stage()

        self.assertEqual(
            os.path.basename(model_a.registry.get_attr("path")),
            os.path.basename(model_b.registry.get_attr("path")),
        )
        self.assertEqual(model_a.name, model_b.name)
        self.assertTrue(filecmp.cmp(model_a.registry.get_attr("path"), model_b.registry.get_attr("path")))

    def test_zenodo_fail(self):
        name = "mock_zenodo"
        filename_ = "model_notreal.csv"  # File not found in repository
        dir_ = os.path.join(tempfile.tempdir, "zenodo_notreal")
        if os.path.isdir(dir_):
            shutil.rmtree(dir_)
        path_ = os.path.join(dir_, filename_)

        # Initialize from zenodo id
        model = self.init_model(name=name, model_path=path_, zenodo_id=13117711)

        with self.assertRaises(FileNotFoundError):
            model.get_source(model.zenodo_id, model.giturl)
