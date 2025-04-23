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


class TestModelRegistryIntegration(TestCase):

    def setUp(self):
        self.time_independent_model = TimeIndependentModel(
            name="TestTIModel",
            model_path=os.path.abspath(
                os.path.join(os.path.dirname(__file__), "../artifacts/models/model.csv")
            ),
            forecast_unit=1,
            store_db=False,
        )
        self.time_dependent_model = TimeDependentModel(
            name="mock",
            model_path=os.path.abspath(
                os.path.join(os.path.dirname(__file__), "../artifacts/models/td_model")
            ),
            func="run_model",
        )

    def test_time_independent_model_stage(self):
        timewindows = [
            [datetime(2023, 1, 1), datetime(2023, 1, 2)],
        ]
        self.time_independent_model.stage(timewindows=timewindows)
        print("a", self.time_independent_model.registry.as_dict())
        self.assertIn("2023-01-01_2023-01-02", self.time_independent_model.registry.forecasts)

    def test_time_independent_model_get_forecast(self):
        tstring = "2023-01-01_2023-01-02"
        self.time_independent_model.repository.forecasts[tstring] = "forecast"
        forecast = self.time_independent_model.get_forecast(tstring)
        self.assertEqual(forecast, "forecast")

    def test_time_independent_model_get_forecast_real(self):
        tstring = "2023-01-01_2023-01-02"
        timewindows = [
            [datetime(2023, 1, 1), datetime(2023, 1, 2)],
        ]
        self.time_independent_model.stage(timewindows=timewindows)
        forecast = self.time_independent_model.get_forecast(tstring)
        self.assertIsInstance(forecast, GriddedForecast)
        self.assertAlmostEqual(forecast.data[0, 0], 0.002739726027357392)  # 1 / 365 days

    @patch("floatcsep.infrastructure.environments.VenvManager.create_environment")
    @patch("floatcsep.infrastructure.environments.CondaManager.create_environment")
    def test_time_dependent_model_stage(self, mock_venv, mock_conda):
        mock_venv.return_value = None
        mock_conda.return_value = None
        timewindows = [
            [datetime(2020, 1, 1), datetime(2020, 1, 2)],
            [datetime(2020, 1, 2), datetime(2020, 1, 3)],
        ]
        tstrings = ["2020-01-01_2020-01-02", "2020-01-02_2020-01-03"]
        self.time_dependent_model.stage(timewindows=timewindows)

        self.assertIn(tstrings[0], self.time_dependent_model.registry.forecasts)
        self.assertIn(tstrings[1], self.time_dependent_model.registry.forecasts)

    @patch("floatcsep.infrastructure.environments.VenvManager.create_environment")
    @patch("floatcsep.infrastructure.environments.CondaManager.create_environment")
    def test_time_dependent_model_get_forecast(self, mock_venv, mock_conda):
        mock_venv.return_value = None
        mock_conda.return_value = None
        timewindows = [
            [datetime(2020, 1, 1), datetime(2020, 1, 2)],
            [datetime(2020, 1, 2), datetime(2020, 1, 3)],
        ]
        self.time_dependent_model.stage(timewindows)
        tstring = "2020-01-01_2020-01-02"
        forecast = self.time_dependent_model.get_forecast(tstring)
        self.assertIsNotNone(forecast)
        self.assertAlmostEqual(list(forecast.catalogs)[1].get_longitudes()[0], 1)


class TestModelRepositoryIntegration(TestCase):

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

    def test_get_forecast_from_repository(self):
        """reads from file, scale in runtime"""
        _rates = numpy.array([[1.0, 0.1], [1.0, 0.1]])
        _mags = numpy.array([5.0, 5.1])
        origins = numpy.array([[0.0, 0.0], [0.0, 1.0]])
        _region = csep.core.regions.CartesianGrid2D.from_origins(origins)

        def forecast_(_):
            return _rates, _region, _mags

        start = datetime(1900, 1, 1)
        end = datetime(2000, 1, 1)
        timestring = timewindow2str([start, end])

        name = "mock"
        fname = os.path.join(self._dir, "model.csv")

        with patch("floatcsep.readers.ForecastParsers.csv", forecast_):
            model = self.init_model(name, fname)
            model.registry.build_tree([[start, end]])
            forecast = model.get_forecast(timestring)
            numpy.testing.assert_almost_equal(220.0, forecast.data.sum())

