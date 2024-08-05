import os.path
from datetime import datetime
from unittest import TestCase
from unittest.mock import patch, MagicMock, mock_open

import csep.core.regions
import numpy.testing

from floatcsep.model import TimeIndependentModel, TimeDependentModel
from floatcsep.utils import str2timewindow


class TestModel(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        path = os.path.dirname(__file__)
        cls._path = path
        cls._dir = os.path.normpath(os.path.join(path, "../artifacts", "models"))

    @staticmethod
    def assertEqualModel(exp_a, exp_b):

        keys_a = list(exp_a.__dict__.keys())
        keys_b = list(exp_a.__dict__.keys())

        if keys_a != keys_b:
            raise AssertionError("Models are not equal")

        for i in keys_a:
            if not (getattr(exp_a, i) == getattr(exp_b, i)):
                raise AssertionError("Models are not equal")


class TestTimeIndependentModel(TestModel):

    @staticmethod
    def init_model(name, model_path, **kwargs):
        """Instantiates a model without using the @register deco,
        but mocks Model.Registry() attrs"""

        model = TimeIndependentModel(name=name, model_path=model_path, **kwargs)

        return model

    def test_from_filesystem(self):
        """init from file, check base attributes"""
        name = "mock"
        fname = os.path.join(self._dir, "model.csv")

        # Initialize without Registry
        model = self.init_model(name=name, model_path=fname)

        self.assertEqual(name, model.name)
        self.assertEqual(fname, model.registry.path)
        self.assertEqual(1, model.forecast_unit)

    @patch("os.makedirs")
    @patch("floatcsep.model.TimeIndependentModel.get_source")
    @patch("floatcsep.registry.ForecastRegistry.build_tree")
    def test_stage_creates_directory(self, mock_build_tree, mock_get_source, mock_makedirs):
        """Test stage method creates directory."""
        model = self.init_model("mock", "mockfile.csv")
        model.force_stage = True  # Simulate forcing the staging process
        model.stage()
        mock_makedirs.assert_called_once()
        mock_get_source.assert_called_once()

    def test_from_dict(self):
        """test that '__init__' and 'from_dict' instantiates
        identical objets"""

        name = "mock"
        fname = os.path.join(self._dir, "model.csv")

        dict_ = {
            "mock": {
                "model_path": fname,
                "forecast_unit": 5,
                "authors": ["Darwin, C.", "Bell, J.", "Et, Al."],
                "doi": "10.1010/10101010",
                "giturl": "should not be accessed, bc filesystem exists",
                "zenodo_id": "should not be accessed, bc filesystem " "exists",
            }
        }

        # Has to be instantiated with registry
        model_a = TimeIndependentModel.from_dict(dict_)

        # Import from normal py dict structure
        py_dict = {"name": "mock", **dict_["mock"]}
        model_b = TimeIndependentModel.from_dict(py_dict)

        self.assertEqual(name, model_a.name)
        self.assertEqual(fname, model_a.registry.path)
        self.assertEqual("csv", model_a.registry.fmt)
        self.assertEqual(self._dir, model_a.registry.dir)

        self.assertEqualModel(model_a, model_b)

        with self.assertRaises(IndexError):
            TimeIndependentModel.from_dict({"model": 1, "no_name": 2})
        with self.assertRaises(IndexError):
            TimeIndependentModel.from_dict(
                {"model_1": {"name": "quack"}, "model_2": {"name": "moo"}}
            )

    @patch("floatcsep.model.TimeIndependentModel.forecast_from_file")
    def test_create_forecast(self, mock_file):

        model = self.init_model("mock", "mockfile.csv")
        model.create_forecast("2020-01-01_2021-01-01")
        self.assertTrue(mock_file.called)

    def test_forecast_from_file(self):
        """reads from file, scale in runtime"""
        _rates = numpy.array([[1.0, 0.1], [1.0, 0.1]])
        _mags = numpy.array([5.0, 5.1])
        origins = numpy.array([[0.0, 0.0], [0.0, 1.0]])
        _region = csep.core.regions.CartesianGrid2D.from_origins(origins)

        def forecast_(_):
            return _rates, _region, _mags

        name = "mock"
        fname = os.path.join(self._dir, "model.csv")

        with patch("floatcsep.readers.ForecastParsers.csv", forecast_):
            model = self.init_model(name, fname)
            start = datetime(1900, 1, 1)
            end = datetime(2000, 1, 1)
            model.registry.build_tree([[start, end]])
            model.forecast_from_file(start, end)
            numpy.testing.assert_almost_equal(
                220.0, model.forecasts["1900-01-01_2000-01-01"].data.sum()
            )

    def test_get_forecast(self):

        model = self.init_model("mock", "mockfile.csv")
        model.forecasts = {"a": 1, "moo": 1, "cuack": 1}

        self.assertEqual(1, model.get_forecast("a"))
        self.assertEqual([1, 1], model.get_forecast(["moo", "cuack"]))
        with self.assertRaises(KeyError):
            model.get_forecast(["woof"])
        with self.assertRaises(ValueError):
            model.get_forecast("meaow")

    def test_todict(self):

        fname = os.path.join(self._dir, "model.csv")
        dict_ = {
            "forecast_unit": 5,
            "authors": ["Darwin, C.", "Bell, J.", "Et, Al."],
            "doi": "10.1010/10101010",
            "giturl": "should not be accessed, bc filesystem exists",
            "zenodo_id": "should not be accessed, bc filesystem exists",
        }
        model = self.init_model(name="mock", model_path=fname, **dict_)
        model_dict = model.as_dict()
        eq = True

        for k, v in dict_.items():
            if k not in list(model_dict["mock"].keys()):
                eq = False
            else:
                if v != model_dict["mock"][k]:
                    eq = False
        excl = ["path", "giturl", "forecast_unit"]
        keys = list(model.as_dict(excluded=excl).keys())

        for i in excl:
            if i in keys and i != "path":  # path always gets printed
                eq = False
        self.assertTrue(eq)

    @patch("os.path.isfile", return_value=False)
    @patch("floatcsep.model.HDF5Serializer.grid2hdf5")
    def test_init_db(self, mock_grid2hdf5, mock_isfile):
        """Test init_db method creates database."""
        filepath = os.path.join(self._dir, "model.csv")
        model = self.init_model("mock", filepath)
        model.init_db(force=True)


class TestTimeDependentModel(TestModel):

    @staticmethod
    def init_model(name, model_path, **kwargs):
        """Instantiates a model without using the @register deco,
        but mocks Model.Registry() attrs"""

        model = TimeDependentModel(name=name, model_path=model_path, **kwargs)

        return model

    @patch("floatcsep.model.TimeDependentModel.forecast_from_func")
    def test_create_forecast(self, mock_func):

        model = self.init_model("mock", "mockbins", model_class="td")

        model.registry.build_tree([str2timewindow("2020-01-01_2021-01-01")])
        model.create_forecast("2020-01-01_2021-01-01")
        self.assertTrue(mock_func.called)

    @patch("csep.load_catalog_forecast")  # Mocking the load_catalog_forecast function
    def test_get_forecast_single(self, mock_load_forecast):
        # Arrange
        model_path = os.path.join(self._dir, "td_model")
        model = self.init_model(name="mock", model_path=model_path)
        tstring = "2020-01-01_2020-01-02"  # Example time window string
        model.stage([str2timewindow(tstring)])

        region = "TestRegion"

        # Mock the return value of load_catalog_forecast
        mock_forecast = MagicMock()
        mock_load_forecast.return_value = mock_forecast

        # Act
        result = model.get_forecast(tstring=tstring, region=region)

        # Assert
        mock_load_forecast.assert_called_once_with(
            model.registry("forecasts", tstring),
            region=region,
            apply_filters=True,
            filter_spatial=True,
        )
        self.assertEqual(result, mock_forecast)

    @patch("csep.load_catalog_forecast")
    def test_get_forecast_multiple(self, mock_load_forecast):

        model_path = os.path.join(self._dir, "td_model")
        model = self.init_model(name="mock", model_path=model_path)
        tstrings = [
            "2020-01-01_2020-01-02",
            "2020-01-02_2020-01-03",
        ]  # Example list of time window strings
        region = "TestRegion"
        model.stage(str2timewindow(tstrings))
        # Mock the return values of load_catalog_forecast for each forecast
        mock_forecast1 = MagicMock()
        mock_forecast2 = MagicMock()
        mock_load_forecast.side_effect = [mock_forecast1, mock_forecast2]

        # Act
        result = model.get_forecast(tstring=tstrings, region=region)

        # Assert
        self.assertEqual(len(result), 2)
        mock_load_forecast.assert_any_call(
            model.registry("forecasts", tstrings[0]),
            region=region,
            apply_filters=True,
            filter_spatial=True,
        )
        mock_load_forecast.assert_any_call(
            model.registry("forecasts", tstrings[1]),
            region=region,
            apply_filters=True,
            filter_spatial=True,
        )
        self.assertEqual(result[0], mock_forecast1)
        self.assertEqual(result[1], mock_forecast2)

    def test_argprep(self):
        model_path = os.path.join(self._dir, "td_model")
        with open(os.path.join(model_path, "input", "args.txt"), "w") as args:
            args.write("start_date = foo\nend_date = bar")

        model = self.init_model("a", model_path, func="func", build="docker")
        start = datetime(2000, 1, 1)
        end = datetime(2000, 1, 2)
        model.stage([[start, end]])
        model.prepare_args(start, end)

        with open(os.path.join(model_path, "input", "args.txt"), "r") as args:
            self.assertEqual(args.readline(), f"start_date = {start.isoformat()}\n")
            self.assertEqual(args.readline(), f"end_date = {end.isoformat()}\n")
        model.prepare_args(start, end, n_sims=400)
        with open(os.path.join(model_path, "input", "args.txt"), "r") as args:
            self.assertEqual(args.readlines()[2], f"n_sims = 400\n")

        model.prepare_args(start, end, n_sims=200)
        with open(os.path.join(model_path, "input", "args.txt"), "r") as args:
            self.assertEqual(args.readlines()[2], f"n_sims = 200\n")

    @patch("floatcsep.model.open", new_callable=mock_open, read_data='{"key": "value"}')
    @patch("json.dump")
    def test_argprep_json(self, mock_json_dump, mock_file):
        model = self.init_model(
            name="TestModel", model_path=os.path.join(self._dir, "td_model")
        )
        model.registry = MagicMock(return_value="path/to/model/args.json")
        start = MagicMock()
        end = MagicMock()
        start.isoformat.return_value = "2023-01-01"
        end.isoformat.return_value = "2023-01-31"

        kwargs = {"key1": "value1", "key2": "value2"}

        # Act
        model.prepare_args(start, end, **kwargs)

        # Assert
        # Check that the file was opened for reading
        mock_file.assert_any_call("path/to/model/args.json", "r")

        # Check that the file was opened for writing
        mock_file.assert_any_call("path/to/model/args.json", "w")

        # Check that the JSON data was updated correctly
        expected_data = {
            "key": "value",
            "start_date": "2023-01-01",
            "end_date": "2023-01-31",
            "key1": "value1",
            "key2": "value2",
        }

        # Check that json.dump was called with the expected data
        mock_json_dump.assert_called_with(expected_data, mock_file(), indent=2)
