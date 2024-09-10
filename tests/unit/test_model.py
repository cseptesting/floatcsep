import os.path
from unittest import TestCase

from floatcsep.model import TimeIndependentModel
from floatcsep.infrastructure.registries import ForecastRegistry
from floatcsep.infrastructure.repositories import GriddedForecastRepository
from unittest.mock import patch, MagicMock, mock_open
from floatcsep.model import TimeDependentModel
from datetime import datetime


class TestModel(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        path = os.path.dirname(__file__)
        cls._path = path
        cls._dir = os.path.normpath(os.path.join(path, "../artifacts", "models"))

    @staticmethod
    def assertEqualModel(model_a, model_b):

        keys_a = list(model_a.__dict__.keys())
        keys_b = list(model_a.__dict__.keys())

        if keys_a != keys_b:
            raise AssertionError("Models are not equal")

        for i in keys_a:
            if isinstance(getattr(model_a, i), ForecastRegistry):
                continue
            if not (getattr(model_a, i) == getattr(model_b, i)):
                print(getattr(model_a, i), getattr(model_b, i))
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
    @patch("floatcsep.infrastructure.registries.ForecastRegistry.build_tree")
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

        # print(model_a.__dict__, model_b.__dict__)
        self.assertEqualModel(model_a, model_b)

        with self.assertRaises(IndexError):
            TimeIndependentModel.from_dict({"model": 1, "no_name": 2})
        with self.assertRaises(IndexError):
            TimeIndependentModel.from_dict(
                {"model_1": {"name": "quack"}, "model_2": {"name": "moo"}}
            )

    @patch.object(GriddedForecastRepository, "load_forecast")
    def test_get_forecast(self, repo_mock):

        repo_mock.return_value = 1
        model = self.init_model("mock", "mockfile.csv")
        self.assertEqual(1, model.get_forecast("1900-01-01_2000-01-01"))
        repo_mock.assert_called_once_with(
            "1900-01-01_2000-01-01", name="mock", region=None, forecast_unit=1
        )

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

    def setUp(self):
        # Patches
        self.patcher_registry = patch("floatcsep.model.ForecastRegistry")
        self.patcher_repository = patch("floatcsep.model.ForecastRepository.factory")
        self.patcher_environment = patch("floatcsep.model.EnvironmentFactory.get_env")
        self.patcher_get_source = patch(
            "floatcsep.model.Model.get_source"
        )  # Patch the get_source method on Model

        # Start patches
        self.mock_registry = self.patcher_registry.start()
        self.mock_repository_factory = self.patcher_repository.start()
        self.mock_environment = self.patcher_environment.start()
        self.mock_get_source = self.patcher_get_source.start()

        # Mock instances
        self.mock_registry_instance = MagicMock()
        self.mock_registry.return_value = self.mock_registry_instance

        self.mock_repository_instance = MagicMock()
        self.mock_repository_factory.return_value = self.mock_repository_instance

        self.mock_environment_instance = MagicMock()
        self.mock_environment.return_value = self.mock_environment_instance

        # Set attributes on the mock objects
        self.mock_registry_instance.workdir = "/path/to/workdir"
        self.mock_registry_instance.path = "/path/to/model"
        self.mock_registry_instance.get.return_value = (
            "/path/to/args_file.txt"  # Mocking the return of the registry call
        )

        # Test data
        self.name = "TestModel"
        self.model_path = "/path/to/model"
        self.func = "run_forecast"

        # Instantiate the model
        self.model = TimeDependentModel(
            name=self.name, model_path=self.model_path, func=self.func
        )

    def tearDown(self):
        patch.stopall()

    def test_init(self):
        # Assertions to check if the components were instantiated correctly
        self.mock_registry.assert_called_once_with(
            os.getcwd(), self.model_path
        )  # Ensure the registry is initialized correctly
        self.mock_repository_factory.assert_called_once_with(
            self.mock_registry_instance, model_class="TimeDependentModel"
        )
        self.mock_environment.assert_called_once_with(
            None, self.name, self.mock_registry_instance.abs(self.model_path)
        )

        self.assertEqual(self.model.name, self.name)
        self.assertEqual(self.model.func, self.func)
        self.assertEqual(self.model.registry, self.mock_registry_instance)
        self.assertEqual(self.model.repository, self.mock_repository_instance)
        self.assertEqual(self.model.environment, self.mock_environment_instance)

    @patch("os.makedirs")
    def test_stage(self, mk):
        self.model.force_stage = True  # Force staging to occur

        self.model.stage(timewindows=["2020-01-01_2020-12-31"])

        self.mock_get_source.assert_called_once_with(
            self.model.zenodo_id, self.model.giturl, branch=self.model.repo_hash
        )
        self.mock_registry_instance.build_tree.assert_called_once_with(
            timewindows=["2020-01-01_2020-12-31"],
            model_class="TimeDependentModel",
            prefix=self.model.__dict__.get("prefix", self.name),
            args_file=self.model.__dict__.get("args_file", None),
            input_cat=self.model.__dict__.get("input_cat", None),
        )
        self.mock_environment_instance.create_environment.assert_called_once()

    def test_get_forecast(self):
        tstring = "2020-01-01_2020-12-31"
        self.model.get_forecast(tstring)

        self.mock_repository_instance.load_forecast.assert_called_once_with(
            tstring, region=None
        )

    @patch("floatcsep.model.TimeDependentModel.prepare_args")
    def test_create_forecast(self, prep_args_mock):
        tstring = "2020-01-01_2020-12-31"
        prep_args_mock.return_value = None
        self.model.registry.forecast_exists.return_value = False
        self.model.create_forecast(tstring, force=True)

        self.mock_environment_instance.run_command.assert_called_once_with(
            f'{self.func} {self.model.registry.get("args_file")}'
        )

    @patch("builtins.open", new_callable=mock_open)
    @patch("json.load")
    @patch("json.dump")
    def test_prepare_args(self, mock_json_dump, mock_json_load, mock_open_file):
        start_date = datetime(2020, 1, 1)
        end_date = datetime(2020, 12, 31)

        # Mock json.load to return a dictionary
        mock_json_load.return_value = {
            "start_date": "2020-01-01T00:00:00",
            "end_date": "2020-12-31T00:00:00",
            "custom_arg": "value",
        }

        # Simulate reading a .txt file
        mock_open_file().readlines.return_value = [
            "start_date = 2020-01-01T00:00:00\n",
            "end_date = 2020-12-31T00:00:00\n",
            "custom_arg = value\n",
        ]

        # Call the method
        args_file_path = self.model.registry.get("args_file")
        self.model.prepare_args(start_date, end_date, custom_arg="value")
        mock_open_file.assert_any_call(args_file_path, "r")
        mock_open_file.assert_any_call(args_file_path, "w")
        handle = mock_open_file()
        handle.writelines.assert_any_call(
            [
                "start_date = 2020-01-01T00:00:00\n",
                "end_date = 2020-12-31T00:00:00\n",
                "custom_arg = value\n",
            ]
        )

        json_file_path = "/path/to/args_file.json"
        self.model.registry.get.return_value = json_file_path
        self.model.prepare_args(start_date, end_date, custom_arg="value")

        mock_open_file.assert_any_call(json_file_path, "r")
        mock_json_load.assert_called_once()
        mock_open_file.assert_any_call(json_file_path, "w")
        mock_json_dump.assert_called_once_with(
            {
                "start_date": "2020-01-01T00:00:00",
                "end_date": "2020-12-31T00:00:00",
                "custom_arg": "value",
            },
            mock_open_file(),
            indent=2,
        )
