import datetime
import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch, PropertyMock, mock_open

import numpy
from csep.core.catalogs import CSEPCatalog
from csep.core.forecasts import GriddedForecast
from csep.utils.time_utils import datetime_to_utc_epoch

from floatcsep.utils.helpers import read_time_cfg, read_region_cfg, read_input_cfg

from floatcsep.utils.file_io import GriddedForecastParsers
from floatcsep.infrastructure.registries import ModelFileRegistry
from floatcsep.infrastructure.repositories import (
    CatalogForecastRepository,
    GriddedForecastRepository,
    ResultsRepository,
    CatalogRepository,
)


class TestCatalogForecastRepository(unittest.TestCase):

    def setUp(self):
        self.registry = MagicMock(spec=ModelFileRegistry)  # todo: Factory registry
        self.registry.__call__ = MagicMock(return_value="a_duck")
        self.registry.fmt = "csv"

    @patch("csep.load_catalog_forecast")
    def test_initialization(self, mock_load_catalog_forecast):
        repo = CatalogForecastRepository(self.registry, lazy_load=True)
        self.assertTrue(repo.lazy_load)

    @patch("floatcsep.file_io.CatalogForecastParsers.csv")
    def test_load_forecast(self, mock_load_catalog_forecast):
        repo = CatalogForecastRepository(self.registry)

        mock_load_catalog_forecast.return_value = "forecatto"
        forecast = repo.load_forecast("2023-01-01_2023-01-02")
        self.assertEqual(forecast, "forecatto")

        # Test load_forecast with list
        forecasts = repo.load_forecast(["2023-01-01_2023-01-01", "2023-01-02_2023-01-03"])
        self.assertEqual(forecasts, ["forecatto", "forecatto"])

    @patch("floatcsep.file_io.CatalogForecastParsers.csv")
    def test_load_single_forecast(self, mock_load_catalog_forecast):
        # Test _load_single_forecast
        repo = CatalogForecastRepository(self.registry)
        mock_load_catalog_forecast.return_value = "forecatto"
        forecast = repo._load_single_forecast("2023-01-01_2023-01-01")
        self.assertEqual(forecast, "forecatto")


class TestGriddedForecastRepository(unittest.TestCase):

    def setUp(self):
        self.registry = MagicMock(spec=ModelFileRegistry)  # todo: Factory registry
        self.registry.fmt = "hdf5"
        self.registry.__call__ = MagicMock(return_value="a_duck")

    def test_initialization(self):
        repo = GriddedForecastRepository(self.registry, lazy_load=False)
        self.assertFalse(repo.lazy_load)

    @patch.object(GriddedForecastParsers, "hdf5")
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

    @patch.object(GriddedForecastParsers, "hdf5")
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
    @patch.object(GriddedForecastParsers, "hdf5")
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

    @patch.object(GriddedForecastParsers, "hdf5")
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

    @patch("floatcsep.infrastructure.registries.ModelFileRegistry")
    def test_equal(self, MockModelFileRegistry):

        self.registry = MockModelFileRegistry()

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

    @patch("floatcsep.infrastructure.repositories.ExperimentRegistry.factory")
    def setUp(self, mock_registry):
        self.mock_registry = MagicMock()
        self.mock_registry.return_value = mock_registry()
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

    @patch("floatcsep.infrastructure.repositories.ExperimentRegistry.factory")
    def setUp(self, mock_registry):
        self.mock_registry = MagicMock()
        self.mock_registry.return_value = mock_registry()
        self.catalog_repo = CatalogRepository(self.mock_registry)

    def test_initialization(self):
        self.assertEqual(self.catalog_repo.registry, self.mock_registry)

    @patch("floatcsep.infrastructure.repositories.isfile", return_value=True)
    @patch("csep.load_catalog", return_value="csep catalog")
    def test_set_catalog(self, mock_reader, mock_isfile):
        self.mock_registry.rel.return_value = "catalog_path"

        self.catalog_repo.set_main_catalog("catalog_path", {}, {})

        self.assertEqual(self.catalog_repo.cat_path, "catalog_path")
        self.assertEqual(self.catalog_repo._catalog, "csep catalog")


class TestCatalogFiltering(unittest.TestCase):
    """Input and test catalog filtering, with a synthetic catalog around the Italy region."""

    window = "2016-08-24_2016-08-25"

    @staticmethod
    def _epoch(*args):
        return datetime_to_utc_epoch(datetime.datetime(*args, tzinfo=datetime.timezone.utc))

    @classmethod
    def setUpClass(cls):
        # id, time, lat, lon, depth, mag
        rows = [
            (1, cls._epoch(2010, 1, 1), 42.5, 13.0, 10, 2.6),  # small, before window
            (2, cls._epoch(2012, 1, 1), 42.5, 13.0, 10, 4.5),  # before window
            (3, cls._epoch(2014, 1, 1), 42.5, 13.0, 40, 4.5),  # deep, before window
            (4, cls._epoch(2015, 1, 1), 48.0, 13.0, 10, 4.5),  # outside region, before window
            (5, cls._epoch(2016, 8, 24, 5), 42.5, 13.0, 10, 4.5),  # in window
            (6, cls._epoch(2016, 8, 24, 6), 42.5, 13.0, 50, 4.5),  # in window, deep
            (7, cls._epoch(2008, 1, 1), 42.5, 13.0, 10, 4.5),  # before input start_date
            (8, cls._epoch(2016, 8, 24, 7), 48.0, 13.0, 10, 4.5),  # in window, outside region
        ]
        dtype = numpy.dtype(
            [
                ("id", "S256"),
                ("origin_time", "<i8"),
                ("latitude", "<f4"),
                ("longitude", "<f4"),
                ("depth", "<f4"),
                ("magnitude", "<f4"),
            ]
        )
        data = numpy.array([(str(r[0]).encode(),) + r[1:] for r in rows], dtype=dtype)
        cls.catalog = CSEPCatalog(data=data)
        cls.time_config = read_time_cfg(
            {
                "start_date": datetime.datetime(2016, 8, 24),
                "end_date": datetime.datetime(2016, 8, 25),
                "horizon": "1days",
                "exp_class": "td",
            }
        )
        cls.region_config = read_region_cfg(
            {
                "region": "italy_csep_region",
                "mag_min": 4.0,
                "mag_max": 8.0,
                "mag_bin": 0.1,
                "depth_min": 0,
                "depth_max": 30,
            }
        )

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.registry = MagicMock()
        self.registry.rel.side_effect = lambda p: p
        self.registry.get_test_catalog_key.return_value = os.path.join(
            self.tmp.name, "test_catalog.json"
        )

    def tearDown(self):
        self.tmp.cleanup()

    def repo(self, input_config=None):
        repo = CatalogRepository(self.registry)
        repo.time_config = self.time_config
        repo.region_config = self.region_config
        repo.input_config = read_input_cfg(input_config, self.region_config)
        repo._catalog = self.catalog
        return repo

    @staticmethod
    def ids(catalog):
        return sorted(int(i) for i in catalog.get_event_ids())

    def model(self, name):
        model = MagicMock()
        model.registry.get_input_catalog_key.return_value = os.path.join(
            self.tmp.name, f"{name}_catalog.csv"
        )
        return model

    def test_input_cat_config_defaults(self):
        cfg = self.repo().input_cat_config
        self.assertEqual(4.0, cfg["mag_min"])
        self.assertEqual(8.0, cfg["mag_max"])
        self.assertIsNone(cfg["region"])
        self.assertIsNone(cfg["depth_max"])
        self.assertIsNone(cfg["start_date"])

    def test_get_input_cat_default(self):
        # legacy behaviour: testing magnitudes, no region, depth or lower time bound
        cat = self.repo().get_input_cat(self.window)
        self.assertEqual([2, 3, 4, 7], self.ids(cat))

    def test_get_input_cat_configured(self):
        input_config = {
            "catalog": {
                "mag_min": 2.5,
                "region": "italy_csep_region",
                "depth_max": 30,
                "start_date": "2009-01-01T00:00:00",
            }
        }
        cat = self.repo(input_config).get_input_cat(self.window)
        self.assertEqual([1, 2], self.ids(cat))

    def test_get_input_cat_magnitude_only(self):
        cat = self.repo({"catalog": {"mag_min": 2.5}}).get_input_cat(self.window)
        self.assertEqual([1, 2, 3, 4, 7], self.ids(cat))

    def test_set_input_cats_writes_every_model(self):
        repo = self.repo({"catalog": {"mag_min": 2.5, "region": "italy_csep_region"}})
        models = [self.model("a"), self.model("b")]
        repo.set_input_cats(self.window, models)
        for model in models:
            path = model.registry.get_input_catalog_key.return_value
            with open(path) as f:
                lines = f.read().splitlines()
            self.assertEqual("lon,lat,mag,time_string,depth,catalog_id,event_id", lines[0])
            self.assertEqual([1, 2, 3, 7], sorted(int(i.split(",")[-1]) for i in lines[1:]))

    def test_set_test_cats_filters_depth_and_region(self):
        repo = self.repo()
        repo.set_test_cats(self.window)
        with open(self.registry.get_test_catalog_key.return_value) as f:
            events = json.load(f)["catalog"]
        self.assertEqual(["5"], [i[0] for i in events])

    def test_set_test_cats_ignores_input_config(self):
        repo = self.repo({"catalog": {"mag_min": 2.5, "region": None}})
        repo.set_test_cats(self.window)
        with open(self.registry.get_test_catalog_key.return_value) as f:
            events = json.load(f)["catalog"]
        self.assertEqual(["5"], [i[0] for i in events])

    def test_query_bounds(self):
        bounds = self.repo().query_bounds()
        self.assertEqual(datetime.datetime(2016, 8, 24), bounds["start_time"])
        self.assertAlmostEqual(4.0, bounds["min_magnitude"])
        self.assertAlmostEqual(30.0, bounds["max_depth"])

        bounds = self.repo(
            {
                "catalog": {
                    "mag_min": 2.5,
                    "depth_max": 100,
                    "start_date": "2005-01-01T00:00:00",
                    "region": "italy_csep_collection_region",
                }
            }
        ).query_bounds()
        self.assertEqual(datetime.datetime(2005, 1, 1), bounds["start_time"])
        self.assertAlmostEqual(2.5, bounds["min_magnitude"])
        self.assertAlmostEqual(100.0, bounds["max_depth"])
        collection_bbox = self.region_config["region"].get_bbox()
        self.assertLessEqual(bounds["min_longitude"], collection_bbox[0])
        self.assertGreaterEqual(bounds["max_latitude"], collection_bbox[3])


if __name__ == "__main__":
    unittest.main()
