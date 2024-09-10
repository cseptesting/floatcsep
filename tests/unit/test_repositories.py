import datetime
import unittest
from unittest.mock import MagicMock, patch, PropertyMock, mock_open

from csep.core.forecasts import GriddedForecast

from floatcsep.utils.readers import ForecastParsers
from floatcsep.infrastructure.registries import ForecastRegistry
from floatcsep.infrastructure.repositories import (
    CatalogForecastRepository,
    GriddedForecastRepository,
    ResultsRepository,
    CatalogRepository,
)


class TestCatalogForecastRepository(unittest.TestCase):

    def setUp(self):
        self.registry = MagicMock(spec=ForecastRegistry)
        self.registry.__call__ = MagicMock(return_value="a_duck")

    @patch("csep.load_catalog_forecast")
    def test_initialization(self, mock_load_catalog_forecast):
        repo = CatalogForecastRepository(self.registry, lazy_load=True)
        self.assertTrue(repo.lazy_load)

    @patch("csep.load_catalog_forecast")
    def test_load_forecast(self, mock_load_catalog_forecast):
        repo = CatalogForecastRepository(self.registry)
        mock_load_catalog_forecast.return_value = "forecatto"
        forecast = repo.load_forecast("2023-01-01_2023-01-02")
        self.assertEqual(forecast, "forecatto")

        # Test load_forecast with list
        forecasts = repo.load_forecast(["2023-01-01_2023-01-01", "2023-01-02_2023-01-03"])
        self.assertEqual(forecasts, ["forecatto", "forecatto"])

    @patch("csep.load_catalog_forecast")
    def test_load_single_forecast(self, mock_load_catalog_forecast):
        # Test _load_single_forecast
        repo = CatalogForecastRepository(self.registry)
        mock_load_catalog_forecast.return_value = "forecatto"
        forecast = repo._load_single_forecast("2023-01-01_2023-01-01")
        self.assertEqual(forecast, "forecatto")


class TestGriddedForecastRepository(unittest.TestCase):

    def setUp(self):
        self.registry = MagicMock(spec=ForecastRegistry)
        self.registry.fmt = "hdf5"
        self.registry.__call__ = MagicMock(return_value="a_duck")

    def test_initialization(self):
        repo = GriddedForecastRepository(self.registry, lazy_load=False)
        self.assertFalse(repo.lazy_load)

    @patch.object(ForecastParsers, "hdf5")
    def test_load_forecast(self, mock_parser):
        # Mock parser return values
        mock_parser.return_value = ("rates", "region", "mags")

        repo = GriddedForecastRepository(self.registry)
        with patch.object(
            repo, "_get_or_load_forecast", return_value="forecatto"
        ) as mock_method:
            forecast = repo.load_forecast("2023-01-01_2023-01-02")
            self.assertEqual(forecast, "forecatto")
            mock_method.assert_called_once_with("2023-01-01_2023-01-02", "", 1)

        # Test load_forecast with list
        with patch.object(
            repo, "_get_or_load_forecast", return_value="forecatto"
        ) as mock_method:
            forecasts = repo.load_forecast(["2023-01-01_2023-01-02", "2023-01-02_2023-01-03"])
            self.assertEqual(forecasts, ["forecatto", "forecatto"])
            self.assertEqual(mock_method.call_count, 2)

    @patch.object(ForecastParsers, "hdf5")
    def test_get_or_load_forecast(self, mock_parser):
        mock_parser.return_value = ("rates", "region", "mags")
        repo = GriddedForecastRepository(self.registry, lazy_load=False)
        with patch.object(
            repo, "_load_single_forecast", return_value="forecatta"
        ) as mock_method:
            # Test when forecast is not in memory
            forecast = repo._get_or_load_forecast("2023-01-01_2023-01-02", "test_name", 1)
            self.assertEqual(forecast, "forecatta")
            mock_method.assert_called_once_with("2023-01-01_2023-01-02", 1, "test_name")
            self.assertIn("2023-01-01_2023-01-02", repo.forecasts)

            # Test when forecast is in memory
            forecast = repo._get_or_load_forecast("2023-01-01_2023-01-02", "test_name", 1)
            self.assertEqual(forecast, "forecatta")
            mock_method.assert_called_once()  # Should not be called again

    @patch.object(GriddedForecast, "__init__", return_value=None)
    @patch.object(GriddedForecast, "event_count", new_callable=PropertyMock)
    @patch.object(GriddedForecast, "scale")
    @patch.object(ForecastParsers, "hdf5")
    def test_load_single_forecast(self, mock_parser, mock_scale, mock_count, mock_init):
        # Mock parser return values
        mock_count.return_value = 2
        mock_parser.return_value = ("rates", "region", "mags")
        mock_scale.return_value = mock_scale

        # Test _load_single_forecast
        repo = GriddedForecastRepository(self.registry, lazy_load=False)
        with patch("csep.utils.time_utils.decimal_year", side_effect=[2023.0, 2024.0]):
            forecast = repo._load_single_forecast("2023-01-01_2024-01-01", 1, "axe")
            self.assertIsInstance(forecast, GriddedForecast)
            mock_init.assert_called_once_with(
                name="axe",
                data="rates",
                region="region",
                magnitudes="mags",
                start_time=datetime.datetime(2023, 1, 1),
                end_time=datetime.datetime(2024, 1, 1),
            )

    @patch.object(ForecastParsers, "hdf5")
    def test_lazy_load_behavior(self, mock_parser):
        mock_parser.return_value = ("rates", "region", "mags")
        # Test lazy_load behavior
        repo = GriddedForecastRepository(self.registry, lazy_load=False)
        with patch.object(
            repo, "_load_single_forecast", return_value="forecatto"
        ) as mock_method:
            # Load forecast and check if it is stored
            forecast = repo.load_forecast("2023-01-01_2023-01-02")
            self.assertEqual(forecast, "forecatto")
            self.assertIn("2023-01-01_2023-01-02", repo.forecasts)

            # Change to lazy_load=True and check if forecast is not stored
            repo.lazy_load = True
            forecast = repo.load_forecast("2023-01-02_2023-01-03")
            self.assertEqual(forecast, "forecatto")
            self.assertNotIn("2023-01-02_2023-01-03", repo.forecasts)

    @patch("floatcsep.infrastructure.registries.ForecastRegistry")
    def test_equal(self, MockForecastRegistry):

        self.registry = MockForecastRegistry()

        self.repo1 = CatalogForecastRepository(self.registry)
        self.repo2 = CatalogForecastRepository(self.registry)
        self.repo3 = CatalogForecastRepository(self.registry)
        self.repo4 = CatalogForecastRepository(self.registry)

        self.repo1.forecasts = {"1": 1, "2": 2}
        self.repo2.forecasts = {"1": 1, "2": 2}
        self.repo3.forecasts = {"1": 2, "2": 2}
        self.repo4.forecasts = {"3": 1, "2": 2}

        self.assertEqual(self.repo1, self.repo2)
        self.assertNotEqual(self.repo1, self.repo3)
        self.assertNotEqual(self.repo1, self.repo3)


class TestResultsRepository(unittest.TestCase):

    @patch("floatcsep.infrastructure.repositories.ExperimentRegistry")
    def setUp(self, MockRegistry):
        self.mock_registry = MockRegistry()
        self.results_repo = ResultsRepository(self.mock_registry)

    def test_initialization(self):
        self.assertEqual(self.results_repo.registry, self.mock_registry)

    @patch("floatcsep.infrastructure.repositories.EvaluationResult.from_dict")
    @patch("builtins.open", new_callable=unittest.mock.mock_open, read_data='{"key": "value"}')
    def test_load_result(self, mock_open, mock_from_dict):
        mock_from_dict.return_value = "mocked_result"
        result = self.results_repo._load_result("test", "window", "model")
        self.assertEqual(result, "mocked_result")

    @patch.object(ResultsRepository, "_load_result", return_value="mocked_result")
    def test_load_results(self, mock_load_result):
        results = self.results_repo.load_results("test", "window", ["model1", "model2"])
        self.assertEqual(results, ["mocked_result", "mocked_result"])

    @patch("json.dump")
    @patch("builtins.open", new_callable=unittest.mock.mock_open)
    def test_write_result(self, mock_open, mock_json_dump):
        mock_result = MagicMock()
        self.results_repo.write_result(mock_result, "test", "model", "window")
        mock_open.assert_called_once()
        mock_json_dump.assert_called_once()


class TestCatalogRepository(unittest.TestCase):

    @patch("floatcsep.infrastructure.repositories.ExperimentRegistry")
    def setUp(self, MockRegistry):
        self.mock_registry = MockRegistry()
        self.catalog_repo = CatalogRepository(self.mock_registry)

    def test_initialization(self):
        self.assertEqual(self.catalog_repo.registry, self.mock_registry)

    @patch("floatcsep.infrastructure.repositories.isfile", return_value=True)
    def test_set_catalog(self, mock_isfile):
        # Mock the registry's rel method to return the same path for simplicity
        self.mock_registry.rel.return_value = "catalog_path"

        self.catalog_repo.set_main_catalog("catalog_path", {}, {})

        # Check if _catpath is set correctly
        self.assertEqual(self.catalog_repo.cat_path, "catalog_path")

        # Check if _catalog is set correctly
        self.assertEqual(self.catalog_repo._catalog, "catalog_path")


if __name__ == "__main__":
    unittest.main()
