import unittest
from datetime import datetime
from unittest.mock import patch, MagicMock
from floatcsep.infrastructure.registries import ForecastRegistry


class TestForecastRegistry(unittest.TestCase):

    def setUp(self):
        self.registry_file = ForecastRegistry(
            workdir="/test/workdir", path="/test/workdir/model.txt"
        )
        self.registry_folder = ForecastRegistry(
            workdir="/test/workdir", path="/test/workdir/model"
        )

    def test_call(self):
        self.registry_file._parse_arg = MagicMock(return_value="path")
        result = self.registry_file.get("path")
        self.assertEqual(result, "/test/workdir/model.txt")

    @patch("os.path.isdir")
    def test_dir(self, mock_isdir):
        mock_isdir.return_value = False
        self.assertEqual(self.registry_file.dir, "/test/workdir")

        mock_isdir.return_value = True
        self.assertEqual(self.registry_folder.dir, "/test/workdir/model")

    def test_fmt(self):
        self.registry_file.database = "test.db"
        self.assertEqual(self.registry_file.fmt, "db")
        self.registry_file.database = None
        self.assertEqual(self.registry_file.fmt, "txt")

    def test_parse_arg(self):
        self.assertEqual(self.registry_file._parse_arg("arg"), "arg")
        self.assertRaises(Exception, self.registry_file._parse_arg, 123)

    def test_as_dict(self):
        self.assertEqual(
            self.registry_file.as_dict(),
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
        result = self.registry_file.abs("file.txt")
        self.assertTrue(result.endswith("/test/workdir/file.txt"))

    def test_absdir(self):
        result = self.registry_file.abs_dir("model.txt")
        self.assertTrue(result.endswith("/test/workdir"))

    @patch("floatcsep.infrastructure.registries.exists")
    def test_fileexists(self, mock_exists):
        mock_exists.return_value = True
        self.registry_file.get = MagicMock(return_value="/test/path/file.txt")
        self.assertTrue(self.registry_file.file_exists("file.txt"))

    @patch("os.makedirs")
    @patch("os.listdir")
    def test_build_tree_time_independent(self, mock_listdir, mock_makedirs):
        timewindows = [[datetime(2023, 1, 1), datetime(2023, 1, 2)]]
        self.registry_file.build_tree(
            timewindows=timewindows, model_class="TimeIndependentModel"
        )
        self.assertIn("2023-01-01_2023-01-02", self.registry_file.forecasts)
        # self.assertIn("2023-01-01_2023-01-02", self.registry_file.inventory)

    @patch("os.makedirs")
    @patch("os.listdir")
    def test_build_tree_time_dependent(self, mock_listdir, mock_makedirs):
        mock_listdir.return_value = ["forecast_1.csv"]
        timewindows = [
            [datetime(2023, 1, 1), datetime(2023, 1, 2)],
            [datetime(2023, 1, 2), datetime(2023, 1, 3)],
        ]
        self.registry_folder.build_tree(
            timewindows=timewindows, model_class="TimeDependentModel", prefix="forecast"
        )
        self.assertIn("2023-01-01_2023-01-02", self.registry_folder.forecasts)
        # self.assertTrue(self.registry_folder.inventory["2023-01-01_2023-01-02"])
        self.assertIn("2023-01-02_2023-01-03", self.registry_folder.forecasts)
        # self.assertTrue(self.registry_folder.inventory["2023-01-02_2023-01-03"])


if __name__ == "__main__":
    unittest.main()
