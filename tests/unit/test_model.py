import os.path
from pathlib import Path
from unittest import TestCase

from floatcsep.model import TimeIndependentModel
from floatcsep.infrastructure.registries import ModelRegistry
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
            if isinstance(getattr(model_a, i), ModelRegistry):
                continue
            if not (getattr(model_a, i) == getattr(model_b, i)):
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
        self.assertEqual(Path(fname), model.registry.path)
        self.assertEqual(1, model.forecast_unit)

    @patch("os.makedirs")
    @patch("floatcsep.model.TimeIndependentModel.get_source")
    @patch("floatcsep.infrastructure.registries.ModelFileRegistry.build_tree")
    def test_stage(self, mock_build_tree, mock_get_source, mock_makedirs):
        model = self.init_model("mock", "mockfile.csv")
        model.force_stage = True
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
        self.assertEqual(Path(fname), model_a.registry.path)
        self.assertEqual(".csv", model_a.registry.fmt)
        self.assertEqual(Path(self._dir), model_a.registry.dir)

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


class TestTimeDependentModel(TestModel):

    def setUp(self):
        # Patches
        self.patcher_registry = patch("floatcsep.model.ModelRegistry.factory")
        self.patcher_repository = patch("floatcsep.model.ForecastRepository.factory")
        self.patcher_environment = patch("floatcsep.model.EnvironmentFactory.get_env")
        self.patcher_get_source = patch(
            "floatcsep.model.Model.get_source"
        )  # Patch the get_source method on Model

        # Start patches
        self.mock_registry_factory = self.patcher_registry.start()
        self.mock_repository_factory = self.patcher_repository.start()
        self.mock_environment = self.patcher_environment.start()
        self.mock_get_source = self.patcher_get_source.start()

        # Mock instances
        self.mock_registry_instance = MagicMock()
        self.mock_registry_factory.return_value = self.mock_registry_instance

        self.mock_repository_instance = MagicMock()
        self.mock_repository_factory.return_value = self.mock_repository_instance

        self.mock_environment_instance = MagicMock()
        self.mock_environment.return_value = self.mock_environment_instance

        # Set attributes on the mock objects
        self.mock_registry_instance.workdir = Path("/path/to/workdir")
        self.mock_registry_instance.path = Path("/path/to/model")
        self.mock_registry_instance.get_input_dir = MagicMock()
        self.mock_registry_instance.get_input_dir.return_value = Path("input")
        self.mock_registry_instance.get_forecast_dir = MagicMock()
        self.mock_registry_instance.get_forecast_dir.return_value = "forecasts"

        self.mock_registry_instance.get_args_key.return_value = (
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
        self.mock_registry_factory.assert_called_once_with(
            model_name=self.name,
            workdir=os.getcwd(),
            path=self.model_path,
            fmt="csv",
            args_file="args.txt",
            input_cat="catalog.csv",
        )  # Ensure the registry is initialized correctly
        self.mock_repository_factory.assert_called_once_with(
            self.mock_registry_instance, model_class="TimeDependentModel"
        )
        self.mock_environment.assert_called_once_with(
            None, self.name, self.mock_registry_instance.path.as_posix()
        )

        self.assertEqual(self.model.name, self.name)
        self.assertEqual(self.model.func, self.func)
        self.assertEqual(self.model.registry, self.mock_registry_instance)
        self.assertEqual(self.model.repository, self.mock_repository_instance)
        self.assertEqual(self.model.environment, self.mock_environment_instance)

    @patch("floatcsep.model.TimeDependentModel.get_source")
    @patch("os.makedirs")
    def test_stage(self, mock_mkdirs, mock_get_source):
        self.model.force_stage = True

        self.model.stage(time_windows=["2020-01-01_2020-12-31"])

        mock_get_source.assert_called_once_with(
            self.model.zenodo_id, self.model.giturl, branch=self.model.repo_hash
        )

        self.mock_registry_instance.build_tree.assert_called_once_with(
            time_windows=["2020-01-01_2020-12-31"],
            model_class="TimeDependentModel",
            prefix=self.model.__dict__.get("prefix", self.name),
            run_mode="sequential",
            stage_dir="results",
            run_id="run",
        )

        self.mock_environment_instance.create_environment.assert_called_once()

    @patch("pathlib.Path.mkdir")
    @patch("pathlib.Path.exists", return_value=True)
    @patch("pathlib.Path.is_dir", return_value=True)
    @patch("floatcsep.model.from_git")
    @patch("floatcsep.model.from_zenodo")
    def test_get_source(
        self,
        mock_from_zenodo,
        mock_from_git,
        mock_is_dir,
        mock_exists,
        mock_mkdir,
    ):
        self.model.giturl = "https://example.com/repo.git"
        self.model.repo_hash = "main"
        self.model.zenodo_id = None
        self.model.force_stage = False

        self.model.get_source(
            self.model.zenodo_id, self.model.giturl, branch=self.model.repo_hash
        )

        mock_from_git.assert_called_once_with(
            self.model.giturl,
            self.mock_registry_instance.path.as_posix(),
            branch=self.model.repo_hash,
            force=False,
        )
        mock_from_zenodo.assert_not_called()

        mock_from_git.reset_mock()
        mock_from_zenodo.reset_mock()

        self.model.giturl = None
        self.model.zenodo_id = 12345

        self.model.get_source(self.model.zenodo_id, self.model.giturl)

        mock_from_zenodo.assert_called_once_with(
            self.model.zenodo_id,
            self.mock_registry_instance.path.as_posix(),
            force=False,
        )
        mock_from_git.assert_not_called()

    def test_get_forecast(self):
        tstring = "2020-01-01_2020-12-31"
        self.model.get_forecast(tstring)

        self.mock_repository_instance.load_forecast.assert_called_once_with(
            tstring, name=self.name, region=None
        )

    @patch("floatcsep.model.TimeDependentModel.prepare_args")
    @patch("floatcsep.model.TimeDependentModel.prepare_extra_input")
    def test_create_forecast(self, preps_extra_mock, prep_args_mock):
        tstring = "2020-01-01_2020-12-31"
        prep_args_mock.return_value = None
        preps_extra_mock.return_value = None
        self.model.registry.forecast_exists.return_value = False
        self.model.create_forecast(tstring, force=True)

        self.mock_environment_instance.run_command.assert_called_once_with(
            command=f"{self.func}",
            input_volume=Path("input"),
            run_label=f"{self.name}_{tstring}",
            forecast_volume="forecasts",
        )

    @patch("pathlib.Path.exists", return_value=True)
    @patch("pathlib.Path.mkdir")
    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data=(
            "start_date = 2000-01-01T00:00:00\n"
            "end_date = 2000-01-02T00:00:00\n"
            "custom_arg = old\n"
        ),
    )
    def test_prepare_args_txt(self, m_open, m_mkdir, m_exists):
        tpl_path = Path("/path/to/model/input/args.txt")
        dest_path = Path("/tmp/run/input/test/args.txt")

        self.mock_registry_instance.get_args_template_path.return_value = tpl_path
        self.mock_registry_instance.get_args_key.return_value = dest_path

        start_date = datetime(2020, 1, 1)
        end_date = datetime(2020, 12, 31)

        self.model.prepare_args(start_date, end_date, custom_arg="value")

        def _was_opened(path, mode):
            return any(
                (args[0] == path or args[0] == str(path)) and args[1] == mode
                for args, _ in m_open.call_args_list
            )

        assert _was_opened(tpl_path, "r"), "template should be opened for read"
        assert _was_opened(dest_path, "w"), "dest should be opened for write"

        handle = m_open()
        chunks = []
        for call in handle.write.mock_calls:
            chunks.append(call.args[0])
        for call in handle.writelines.mock_calls:
            arg0 = call.args[0]
            if isinstance(arg0, (list, tuple)):
                chunks.extend(arg0)
            else:
                chunks.append(arg0)

        written = "".join(chunks)

        assert "start_date = 2020-01-01T00:00:00\n" in written
        assert "end_date = 2020-12-31T00:00:00\n" in written
        assert "custom_arg = value\n" in written

    @patch("pathlib.Path.exists", return_value=True)
    @patch("builtins.open", new_callable=mock_open)
    @patch("json.load")
    @patch("json.dump")
    def test_prepare_args_json(self, m_json_dump, m_json_load, m_open, m_exists):
        tpl_path = Path("/path/to/model/input/args.json")
        dest_path = Path("/tmp/run/input/test/args.json")

        self.mock_registry_instance.get_args_template_path.return_value = tpl_path
        self.mock_registry_instance.get_args_key.return_value = dest_path

        m_json_load.return_value = {"custom_arg": "value"}

        start_date = datetime(2020, 1, 1)
        end_date = datetime(2020, 12, 31)

        self.model.prepare_args(start_date, end_date, extra="X")

        def _was_opened(path, mode):
            return any(
                (args[0] == path or args[0] == str(path)) and args[1] == mode
                for args, _ in m_open.call_args_list
            )

        assert _was_opened(tpl_path, "r")
        assert _was_opened(dest_path, "w")

        (dumped_dict, _fh), kwargs = m_json_dump.call_args
        assert dumped_dict["start_date"] == "2020-01-01T00:00:00"
        assert dumped_dict["end_date"] == "2020-12-31T00:00:00"
        assert dumped_dict["custom_arg"] == "value"
        assert dumped_dict["extra"] == "X"
        assert kwargs.get("indent") == 2

    @patch("pathlib.Path.exists", return_value=True)
    @patch("builtins.open", new_callable=mock_open)
    @patch("yaml.safe_load")
    @patch("yaml.safe_dump")
    def test_prepare_args_yaml(self, m_yaml_dump, m_yaml_load, m_open, m_exists):
        tpl_path = Path("/path/to/model/input/args.yml")
        dest_path = Path("/tmp/run/input/test/args.yml")

        self.mock_registry_instance.get_args_template_path.return_value = tpl_path
        self.mock_registry_instance.get_args_key.return_value = dest_path

        m_yaml_load.return_value = {"nested": {"a": 1}}

        self.model.func_kwargs = {"nested": {"b": 2}}
        start_date = datetime(2020, 1, 1)
        end_date = datetime(2020, 1, 2)

        self.model.prepare_args(start_date, end_date, nested={"c": 3})

        def _was_opened(path, mode):
            return any(
                (args[0] == path or args[0] == str(path)) and args[1] == mode
                for args, _ in m_open.call_args_list
            )

        assert _was_opened(tpl_path, "r")
        assert _was_opened(dest_path, "w")

        (dumped_dict, _fh), kwargs = m_yaml_dump.call_args
        assert dumped_dict["start_date"] == "2020-01-01T00:00:00"
        assert dumped_dict["end_date"] == "2020-01-02T00:00:00"
        assert dumped_dict["nested"] == {"a": 1, "b": 2, "c": 3}
        assert kwargs.get("indent") == 2
