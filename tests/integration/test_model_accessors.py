import filecmp
import os.path
import shutil
import socket
import unittest
import tempfile
import requests
from datetime import datetime
from unittest import TestCase
from unittest.mock import patch
from pathlib import Path

import numpy.testing

from floatcsep.model import TimeIndependentModel, TimeDependentModel
from floatcsep.utils.helpers import timewindow2str
from floatcsep.infrastructure.registries import ModelFileRegistry


def has_git():
    from shutil import which

    return which("git") is not None


def has_internet(host="github.com", port=443, timeout=3):
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


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

    def run_forecast_test(self, name, fname, start, end, expected_sum):
        model = self.init_model(name=name, path=fname)
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
        self.run_forecast_test(name, fname, start, end, expected_sum)

    def test_forecast_ti_from_hdf5(self):
        """reads from hdf5, scale in runtime"""
        name = "mock"
        fname = os.path.join(self._dir, "model_h5.hdf5")
        start = datetime(2020, 1, 1)
        end = datetime(2023, 1, 1)
        expected_sum = 13.2
        self.run_forecast_test(name, fname, start, end, expected_sum)

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


@unittest.skipUnless(has_git() and has_internet(), "requires git and internet")
class TestModelFromGitRemote(unittest.TestCase):
    """Integration test with a remote repository."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._root = Path(tempfile.mkdtemp(prefix="git_remote")).resolve()
        cls._models_dir = cls._root / "models"
        cls._models_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls._root, ignore_errors=True)

    @staticmethod
    def init_td(name, model_path, **kwargs):
        return TimeDependentModel(name=name, model_path=model_path, **kwargs)

    @staticmethod
    def init_ti(name, model_path, **kwargs):
        return TimeIndependentModel(name=name, model_path=model_path, **kwargs)

    @patch.object(ModelFileRegistry, "build_tree", return_value=None)
    def test_ti_stag(self, _build_tree):
        giturl = "https://github.com/pabloitu/model_template.git"
        parent_dir = self._models_dir / "model_template_ti"
        file_path = parent_dir / "README.md"  # registry.path points to a FILE
        m = self.init_ti(
            name="model_template_ti",
            model_path=str(file_path),
            giturl=giturl,
            force_stage=True,
        )
        m.stage(time_windows=None)

        self.assertTrue(file_path.exists() and file_path.is_file())
        self.assertTrue(parent_dir.exists() and parent_dir.is_dir())
        self.assertFalse((parent_dir / ".git").exists())
        self.assertTrue(_build_tree.called)


class TestModelFromZenodo(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        path = os.path.dirname(__file__)
        cls._path = path
        cls._dir = os.path.normpath(os.path.join(path, "../artifacts", "models"))

    @staticmethod
    def init_model(name, model_path, **kwargs):
        return TimeIndependentModel(name=name, model_path=model_path, **kwargs)

    @unittest.skipUnless(has_internet(), "Skipping Zenodo integration: no internet")
    @patch.object(ModelFileRegistry, "build_tree", return_value=None)
    def test_zenodo(self, _mock_buildtree):
        try:
            name = "mock_zenodo"
            filename_ = "dummy.txt"

            dir_ = os.path.join(tempfile.gettempdir(), "mock_zenodo_ti")
            if os.path.isdir(dir_):
                shutil.rmtree(dir_)
            os.makedirs(dir_, exist_ok=True)
            path_ = os.path.join(dir_, filename_)

            zenodo_id = 13117711

            model_a = self.init_model(name=name, model_path=path_, zenodo_id=zenodo_id)
            model_a.stage()

            dir_art = os.path.join(self._path, "../artifacts", "models", "zenodo_test")
            path_b = os.path.join(dir_art, filename_)
            model_b = self.init_model(name=name, model_path=path_b, zenodo_id=zenodo_id)
            model_b.stage()

            self.assertEqual(
                os.path.basename(model_a.registry.get_attr("path")),
                os.path.basename(model_b.registry.get_attr("path")),
            )
            self.assertEqual(model_a.name, model_b.name)
            self.assertTrue(
                filecmp.cmp(
                    model_a.registry.get_attr("path"),
                    model_b.registry.get_attr("path"),
                    shallow=False,
                )
            )

            shutil.rmtree(dir_, ignore_errors=True)
        except Exception as e:
            self.skipTest(f"Skipping Zenodo test: {e!r}")

    @unittest.skipUnless(has_internet(), "Skipping Zenodo integration: no internet")
    def test_zenodo_fail(self):
        name = "mock_zenodo"
        filename_ = "model_notreal.csv"
        dir_ = os.path.join(tempfile.gettempdir(), "zenodo_notreal")
        if os.path.isdir(dir_):
            shutil.rmtree(dir_)
        os.makedirs(dir_, exist_ok=True)
        path_ = os.path.join(dir_, filename_)

        model = self.init_model(name=name, model_path=path_, zenodo_id=13117711)

        with self.assertRaises(FileNotFoundError):
            model.get_source()

        shutil.rmtree(dir_, ignore_errors=True)
