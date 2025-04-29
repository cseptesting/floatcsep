import os
import unittest
from datetime import datetime
from unittest.mock import patch, MagicMock
from floatcsep.infrastructure.registries import ModelFileRegistry, ExperimentFileRegistry


class TestModelFileRegistry(unittest.TestCase):

    def setUp(self):
        self.registry_for_filebased_model = ModelFileRegistry(
            workdir="/test/workdir", path="/test/workdir/model.txt"
        )
        self.registry_for_folderbased_model = ModelFileRegistry(
            workdir="/test/workdir", path="/test/workdir/model"
        )

    def test_call(self):
        self.registry_for_filebased_model._parse_arg = MagicMock(return_value="path")
        result = self.registry_for_filebased_model.get_attr("path")
        self.assertEqual(result, "/test/workdir/model.txt")

    @patch("os.path.isdir")
    def test_dir(self, mock_isdir):
        mock_isdir.return_value = False
        self.assertEqual(self.registry_for_filebased_model.dir, "/test/workdir")

        mock_isdir.return_value = True
        self.assertEqual(self.registry_for_folderbased_model.dir, "/test/workdir/model")

    def test_fmt(self):
        self.registry_for_filebased_model.database = "test.db"
        self.assertEqual(self.registry_for_filebased_model.fmt, "db")
        self.registry_for_filebased_model.database = None
        self.assertEqual(self.registry_for_filebased_model.fmt, "txt")

    def test_parse_arg(self):
        self.assertEqual(self.registry_for_filebased_model._parse_arg("arg"), "arg")
        self.assertRaises(Exception, self.registry_for_filebased_model._parse_arg, 123)

    def test_as_dict(self):
        self.assertEqual(
            self.registry_for_filebased_model.as_dict(),
            {
                "args_file": None,
                "database": None,
                "forecasts": {},
                "input_cat": None,
                "path": "/test/workdir/model.txt",
                "workdir": "/test/workdir",
            },
        )

    def test_abs(self):
        result = self.registry_for_filebased_model.abs("file.txt")
        self.assertTrue(result.endswith("/test/workdir/file.txt"))

    def test_abs_dir(self):
        result = self.registry_for_filebased_model.abs_dir("model.txt")
        self.assertTrue(result.endswith("/test/workdir"))

    @patch("floatcsep.infrastructure.registries.exists")
    def test_file_exists(self, mock_exists):
        mock_exists.return_value = True
        self.registry_for_filebased_model.get_attr = MagicMock(return_value="/test/path/file.txt")
        self.assertTrue(self.registry_for_filebased_model.file_exists("file.txt"))

    @patch("os.makedirs")
    @patch("os.listdir")
    def test_build_tree_time_independent(self, mock_listdir, mock_makedirs):
        time_windows = [[datetime(2023, 1, 1), datetime(2023, 1, 2)]]
        self.registry_for_filebased_model.build_tree(
            time_windows=time_windows, model_class="TimeIndependentModel"
        )
        self.assertIn("2023-01-01_2023-01-02", self.registry_for_filebased_model.forecasts)
        # self.assertIn("2023-01-01_2023-01-02", self.registry_for_filebased_model.inventory)

    @patch("os.makedirs")
    @patch("os.listdir")
    def test_build_tree_time_dependent(self, mock_listdir, mock_makedirs):
        mock_listdir.return_value = ["forecast_1.csv"]
        time_windows = [
            [datetime(2023, 1, 1), datetime(2023, 1, 2)],
            [datetime(2023, 1, 2), datetime(2023, 1, 3)],
        ]
        self.registry_for_folderbased_model.build_tree(
            time_windows=time_windows, model_class="TimeDependentModel", prefix="forecast"
        )
        self.assertIn("2023-01-01_2023-01-02", self.registry_for_folderbased_model.forecasts)
        self.assertIn("2023-01-02_2023-01-03", self.registry_for_folderbased_model.forecasts)


class TestExperimentFileRegistry(unittest.TestCase):

    def setUp(self):
        self.registry = ExperimentFileRegistry(workdir="/test/workdir")

    def test_initialization(self):
        self.assertEqual(self.registry.workdir, "/test/workdir")
        self.assertEqual(self.registry.run_dir, "results")
        self.assertEqual(self.registry.results, {})
        self.assertEqual(self.registry.test_catalogs, {})
        self.assertEqual(self.registry.figures, {})
        self.assertEqual(self.registry.model_registries, {})

    def test_add_and_get_model_registry(self):
        model_mock = MagicMock()
        model_mock.name = "TestModel"
        model_mock.registry = MagicMock(spec=ModelFileRegistry)

        self.registry.add_model_registry(model_mock)
        self.assertIn("TestModel", self.registry.model_registries)
        self.assertEqual(self.registry.get_model_registry("TestModel"), model_mock.registry)

    @patch("os.makedirs")
    def test_build_tree(self, mock_makedirs):
        time_windows = [[datetime(2023, 1, 1), datetime(2023, 1, 2)]]
        models = [MagicMock(name="Model1"), MagicMock(name="Model2")]
        tests = [MagicMock(name="Test1")]

        self.registry.build_tree(time_windows, models, tests)

        timewindow_str = "2023-01-01_2023-01-02"
        self.assertIn(timewindow_str, self.registry.results)
        self.assertIn(timewindow_str, self.registry.test_catalogs)
        self.assertIn(timewindow_str, self.registry.figures)

    def test_get_test_catalog_key(self):
        self.registry.test_catalogs = {"2023-01-01_2023-01-02": "some/path/to/catalog.json"}
        result = self.registry.get_test_catalog_key("2023-01-01_2023-01-02")
        self.assertTrue(result.endswith("results/some/path/to/catalog.json"))

    def test_get_result_key(self):
        self.registry.results = {
            "2023-01-01_2023-01-02": {
                "Test1": {
                    "Model1": "some/path/to/result.json"
                }
            }
        }
        result = self.registry.get_result_key("2023-01-01_2023-01-02", "Test1", "Model1")
        self.assertTrue(result.endswith("results/some/path/to/result.json"))

    def test_get_figure_key(self):
        self.registry.figures = {
            "2023-01-01_2023-01-02": {
                "Test1": "some/path/to/figure.png",
                "catalog_map": "some/path/to/catalog_map.png",
                "catalog_time": "some/path/to/catalog_time.png",
                "forecasts": {"Model1": "some/path/to/forecast.png"}
            }
        }
        result = self.registry.get_figure_key("2023-01-01_2023-01-02", "Test1")
        self.assertTrue(result.endswith("results/some/path/to/figure.png"))

    @patch("floatcsep.infrastructure.registries.exists")
    def test_result_exist(self, mock_exists):
        mock_exists.return_value = True
        self.registry.results = {
            "2023-01-01_2023-01-02": {
                "Test1": {
                    "Model1": "some/path/to/result.json"
                }
            }
        }
        result = self.registry.result_exist("2023-01-01_2023-01-02", "Test1", "Model1")
        self.assertTrue(result)
        mock_exists.assert_called()


if __name__ == "__main__":
    unittest.main()
