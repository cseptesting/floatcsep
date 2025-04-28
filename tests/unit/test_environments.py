import os
import venv
import unittest
import subprocess
from unittest.mock import patch, MagicMock, call, mock_open
import shutil
import hashlib
import logging
from floatcsep.infrastructure.environments import (
    CondaManager,
    EnvironmentFactory,
    VenvManager,
    DockerManager,
)
try:
    import docker
    from docker.errors import ImageNotFound, APIError
    DOCKER_SDK_AVAILABLE = True
except ImportError:
    DOCKER_SDK_AVAILABLE = False


class TestCondaEnvironmentManager(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not shutil.which("conda"):
            raise unittest.SkipTest("Conda is not available in the environment.")

    def setUp(self):
        self.manager = CondaManager(base_name="test_env", model_directory="/tmp/test_model")
        os.makedirs("/tmp/test_model", exist_ok=True)
        with open("/tmp/test_model/environment.yml", "w") as f:
            f.write("name: test_env\ndependencies:\n  - python=3.8\n  - numpy")
        with open("/tmp/test_model/setup.py", "w") as f:
            f.write("from setuptools import setup\nsetup(name='test_model', version='0.1')")

    def tearDown(self):
        if self.manager.env_exists():
            subprocess.run(
                ["conda", "env", "remove", "--name", self.manager.env_name, "--yes"],
                check=True,
            )
        if os.path.exists("/tmp/test_model"):
            shutil.rmtree("/tmp/test_model")

    @patch("subprocess.run")
    @patch("shutil.which", return_value="conda")
    def test_generate_env_name(self, mock_which, mock_run):
        manager = CondaManager("test_base", "/path/to/model")
        expected_name = "test_base_" + hashlib.md5("/path/to/model".encode()).hexdigest()[:8]
        self.assertEqual(manager.generate_env_name(), expected_name)

    @patch("subprocess.run")
    def test_env_exists(self, mock_run):
        hashed = hashlib.md5("/path/to/model".encode()).hexdigest()[:8]
        mock_run.return_value.stdout = f"test_base_{hashed}\n".encode()

        manager = CondaManager("test_base", "/path/to/model")
        self.assertTrue(manager.env_exists())

    @patch("subprocess.run")
    @patch("os.path.exists", return_value=True)
    def test_create_environment(self, mock_exists, mock_run):
        manager = CondaManager("test_base", "/path/to/model")
        manager.create_environment(force=False)
        package_manager = manager.detect_package_manager()
        expected_calls = [
            call(["conda", "env", "list"], stdout=-1),
            call().stdout.decode(),
            call().stdout.decode().__contains__(manager.env_name),
            call(
                [
                    package_manager,
                    "env",
                    "create",
                    "--name",
                    manager.env_name,
                    "--file",
                    "/path/to/model/environment.yml",
                ]
            ),
            call(
                [
                    package_manager,
                    "run",
                    "-n",
                    manager.env_name,
                    "pip",
                    "install",
                    "-e",
                    "/path/to/model",
                ],
                check=True,
            ),
        ]

        self.assertEqual(mock_run.call_count, 3)
        mock_run.assert_has_calls(expected_calls, any_order=False)

    @patch("subprocess.run")
    def test_create_environment_force(self, mock_run):
        manager = CondaManager("test_base", "/path/to/model")
        manager.env_exists = MagicMock()
        manager.env_exists.side_effect = [True, False]
        manager.create_environment(force=True)
        self.assertEqual(mock_run.call_count, 3)  # One for remove, one for create

    @patch("subprocess.run")
    @patch.object(CondaManager, "detect_package_manager", return_value="conda")
    def test_install_dependencies(self, mock_detect_package_manager, mock_run):
        manager = CondaManager("test_base", "/path/to/model")
        manager.install_dependencies()
        mock_run.assert_called_once_with(
            [
                "conda",
                "run",
                "-n",
                manager.env_name,
                "pip",
                "install",
                "-e",
                "/path/to/model",
            ],
            check=True,
        )

    @patch("shutil.which", return_value="conda")
    @patch("os.path.exists", side_effect=[False, False, True])
    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data="[metadata]\nname = test\n\n[options]\ninstall_requires =\n    "
        "numpy\npython_requires = >=3.9,<3.12\n",
    )
    def test_detect_python_version_setup_cfg(self, mock_open, mock_exists, mock_which):
        manager = CondaManager("test_base", "../artifacts/models/td_model")
        python_version = manager.detect_python_version()

        # Extract major and minor version parts
        major_minor_version = ".".join(python_version.split(".")[:2])

        self.assertIn(
            major_minor_version, ["3.9", "3.10", "3.11"]
        )  # Check if it falls within the specified range

    def test_create_and_delete_environment(self):
        # Create the environment
        self.manager.create_environment(force=True)

        # Check if the environment was created
        result = subprocess.run(["conda", "env", "list"], stdout=subprocess.PIPE, check=True)
        self.assertIn(self.manager.env_name, result.stdout.decode())

        # Check if numpy is installed
        result = subprocess.run(
            [
                "conda",
                "run",
                "-n",
                self.manager.env_name,
                "python",
                "-c",
                "import numpy",
            ],
            check=True,
        )
        self.assertEqual(result.returncode, 0)

        # Delete the environment
        self.manager.create_environment(
            force=True
        )  # This should remove and recreate the environment

        # Check if the environment was recreated
        result = subprocess.run(["conda", "env", "list"], stdout=subprocess.PIPE, check=True)
        self.assertIn(self.manager.env_name, result.stdout.decode())


class TestEnvironmentFactory(unittest.TestCase):

    @patch("os.path.abspath", return_value="/absolute/path/to/model")
    @patch.object(EnvironmentFactory, "check_environment_type", return_value="conda")
    def test_get_env_conda(self, mock_check_env, mock_abspath):
        env_manager = EnvironmentFactory.get_env(
            build="conda", model_name="test_model", model_path="/path/to/model"
        )
        self.assertIsInstance(env_manager, CondaManager)
        self.assertEqual(env_manager.base_name, "test_model")
        self.assertEqual(env_manager.model_directory, "/absolute/path/to/model")

    @patch("os.path.abspath", return_value="/absolute/path/to/model")
    @patch.object(EnvironmentFactory, "check_environment_type", return_value="venv")
    def test_get_env_venv(self, mock_check_env, mock_abspath):
        env_manager = EnvironmentFactory.get_env(
            build="venv", model_name="test_model", model_path="/path/to/model"
        )
        self.assertIsInstance(env_manager, VenvManager)
        self.assertEqual(env_manager.base_name, "test_model")
        self.assertEqual(env_manager.model_directory, "/absolute/path/to/model")

    @patch("os.path.abspath", return_value="/absolute/path/to/model")
    @patch.object(EnvironmentFactory, "check_environment_type", return_value="micromamba")
    def test_get_env_micromamba(self, mock_check_env, mock_abspath):
        env_manager = EnvironmentFactory.get_env(
            build="micromamba", model_name="test_model", model_path="/path/to/model"
        )
        self.assertIsInstance(
            env_manager, CondaManager
        )  # Assuming Micromamba uses CondaManager
        self.assertEqual(env_manager.base_name, "test_model")
        self.assertEqual(env_manager.model_directory, "/absolute/path/to/model")

    @patch("docker.from_env")
    @patch("os.path.abspath", return_value="/absolute/path/to/model")
    @patch.object(EnvironmentFactory, "check_environment_type", return_value=None)
    def test_get_env_docker(self, mock_check_env, mock_abspath, mock_from_env):
        env_manager = EnvironmentFactory.get_env(
            build="docker", model_name="test_model", model_path="/path/to/model"
        )
        self.assertIsInstance(env_manager, DockerManager)
        self.assertEqual(env_manager.base_name, "test_model")
        self.assertEqual(env_manager.model_directory, "/absolute/path/to/model")

    @patch("os.path.abspath", return_value="/absolute/path/to/model")
    @patch.object(EnvironmentFactory, "check_environment_type", return_value="conda")
    def test_get_env_default_conda(self, mock_check_env, mock_abspath):
        env_manager = EnvironmentFactory.get_env(
            build=None, model_name="test_model", model_path="/path/to/model"
        )
        self.assertIsInstance(env_manager, CondaManager)
        self.assertEqual(env_manager.base_name, "test_model")
        self.assertEqual(env_manager.model_directory, "/absolute/path/to/model")

    @patch("os.path.abspath", return_value="/absolute/path/to/model")
    @patch.object(EnvironmentFactory, "check_environment_type", return_value="venv")
    def test_get_env_default_venv(self, mock_check_env, mock_abspath):
        env_manager = EnvironmentFactory.get_env(
            build=None, model_name="test_model", model_path="/path/to/model"
        )
        self.assertIsInstance(env_manager, VenvManager)
        self.assertEqual(env_manager.base_name, "test_model")
        self.assertEqual(env_manager.model_directory, "/absolute/path/to/model")

    @patch("os.path.abspath", return_value="/absolute/path/to/model")
    @patch.object(EnvironmentFactory, "check_environment_type", return_value=None)
    def test_get_env_invalid(self, mock_check_env, mock_abspath):
        with self.assertRaises(Exception) as context:
            EnvironmentFactory.get_env(
                build="invalid", model_name="test_model", model_path="/path/to/model"
            )
        self.assertTrue("Wrong environment selection" in str(context.exception))

    @patch("os.path.abspath", return_value="/absolute/path/to/model")
    @patch.object(EnvironmentFactory, "check_environment_type", return_value="venv")
    @patch("logging.Logger.warning")
    def test_get_env_warning(self, mock_log_warning, mock_check_env, mock_abspath):
        EnvironmentFactory.get_env(
            build="conda", model_name="test_model", model_path="/path/to/model"
        )
        mock_log_warning.assert_called_once_with(
            f"Selected build environment (conda) for this model is different than that of"
            f" the experiment run. Consider selecting the same environment."
        )


class TestVenvEnvironmentManager(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Check if venv is available (Python standard library)
        if not hasattr(venv, "create"):
            raise unittest.SkipTest("Venv is not available in the environment.")

    def setUp(self):
        self.model_directory = "/tmp/test_model"
        self.manager = VenvManager(base_name="test_env", model_directory=self.model_directory)
        os.makedirs(self.model_directory, exist_ok=True)
        with open(os.path.join(self.model_directory, "setup.py"), "w") as f:
            f.write("from setuptools import setup\nsetup(name='test_model', version='0.1')")
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        if self.manager.env_exists():
            shutil.rmtree(self.manager.env_path)
        if os.path.exists(self.model_directory):
            shutil.rmtree(self.model_directory)

    def test_create_and_delete_environment(self):
        # Create the environment
        self.manager.create_environment(force=True)

        # Check if the environment was created
        self.assertTrue(self.manager.env_exists())

        # Check if pip is available in the environment
        pip_executable = os.path.join(self.manager.env_path, "bin", "pip")
        result = subprocess.run(
            [pip_executable, "list"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        self.assertEqual(result.returncode, 0)  # pip should run without errors

        # Delete the environment
        self.manager.create_environment(
            force=True
        )  # This should remove and recreate the environment

        # Check if the environment was recreated
        self.assertTrue(self.manager.env_exists())

    def test_init(self):
        self.assertEqual(self.manager.base_name, "test_env")
        self.assertEqual(self.manager.model_directory, self.model_directory)
        self.assertTrue(self.manager.env_name.startswith("test_env_"))

    def test_env_exists(self):
        self.assertFalse(self.manager.env_exists())
        self.manager.create_environment(force=True)
        self.assertTrue(self.manager.env_exists())

    def test_create_environment(self):
        self.manager.create_environment(force=True)
        self.assertTrue(self.manager.env_exists())

    def test_create_environment_force(self):
        self.manager.create_environment(force=True)
        env_path_before = self.manager.env_path
        self.manager.create_environment(force=True)
        self.assertTrue(self.manager.env_exists())
        self.assertEqual(env_path_before, self.manager.env_path)  # Ensure it's a new path

    def test_install_dependencies(self):
        self.manager.create_environment(force=True)
        pip_executable = os.path.join(self.manager.env_path, "bin", "pip")
        result = subprocess.run(
            [pip_executable, "install", "-e", self.model_directory],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.assertEqual(result.returncode, 0)  # pip should run without errors

    @patch("subprocess.Popen")
    def test_run_command(self, mock_popen):
        # Arrange
        mock_process = MagicMock()
        mock_process.stdout = iter(["Output line 1\n", "Output line 2\n"])
        mock_process.wait.return_value = None
        mock_popen.return_value = mock_process

        command = "echo test_command"

        # Act
        self.manager.run_command(command)

        output_cmd = f"bash -c 'source {os.path.join(self.manager.env_path, 'bin', 'activate')}' && {command}"
        # Assert
        mock_popen.assert_called_once_with(
            output_cmd,
            shell=True,
            env=unittest.mock.ANY,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )


@unittest.skipUnless(DOCKER_SDK_AVAILABLE, "docker SDK is not installed")
class TestDockerManagerWithSDK(unittest.TestCase):
    def setUp(self):
        # Prepare a fake model directory with a Dockerfile
        self.model_dir = "/tmp/test_model"
        os.makedirs(self.model_dir, exist_ok=True)
        with open(os.path.join(self.model_dir, "Dockerfile"), "w") as f:
            f.write("FROM scratch\n")

        self.patcher = patch("docker.from_env")
        self.mock_from_env = self.patcher.start()
        self.mock_client = MagicMock()
        self.mock_from_env.return_value = self.mock_client

        self.manager = DockerManager("My Model", self.model_dir)

    def tearDown(self):
        self.patcher.stop()
        shutil.rmtree(self.model_dir, ignore_errors=True)

    def test_init(self):
        # base_name slugged
        self.assertEqual(self.manager.base_name, "My_Model")
        # tags
        self.assertEqual(self.manager.image_tag, "my_model_image")
        self.assertEqual(self.manager.container_name, "my_model_container")
        # client setup
        self.mock_from_env.assert_called_once()
        self.assertIs(self.manager.client, self.mock_client)

    def test_env_exists_true(self):
        # images.get returns without error
        self.mock_client.images.get.return_value = MagicMock()
        self.assertTrue(self.manager.env_exists())
        self.mock_client.images.get.assert_called_once_with(self.manager.image_tag)

    def test_env_exists_false(self):
        # images.get raises ImageNotFound
        self.mock_client.images.get.side_effect = ImageNotFound("not found")
        self.assertFalse(self.manager.env_exists())
        self.mock_client.images.get.assert_called_once_with(self.manager.image_tag)

    def test_create_environment_builds_when_missing(self):

        # no image exists
        self.manager.env_exists = MagicMock(return_value=False)
        fake_logs = [{"stream": "Step 1/2\n"}, {"stream": "Step 2/2\n"}]
        self.mock_client.api.build.return_value = fake_logs

        with patch("floatcsep.environments.log") as mock_log:
            self.manager.create_environment(force=False)

        self.mock_client.api.build.assert_called_once_with(
            path=self.manager.model_directory,
            tag=self.manager.image_tag,
            rm=True,
            decode=True,
            buildargs={
                "USER_UID": str(os.getuid()),
                "USER_GID": str(os.getgid()),
            },
            nocache=False
        )
        # success logged only once
        infos = [call_args[0][0] for call_args in mock_log.info.call_args_list]
        self.assertEqual(sum("Successfully built" in msg for msg in infos), 1)

    def test_create_environment_skips_when_exists_and_not_forced(self):
        self.manager.env_exists = MagicMock(return_value=True)
        self.manager.create_environment(force=False)
        self.mock_client.images.remove.assert_not_called()
        self.mock_client.api.build.assert_not_called()

    def test_create_environment_force(self):
        self.manager.env_exists = MagicMock(return_value=True)
        fake_logs = [{"stream": ""}]
        self.mock_client.api.build.return_value = fake_logs

        with patch("floatcsep.environments.log") as mock_log:
            self.manager.create_environment(force=True)

        # remove then build
        self.mock_client.images.remove.assert_called_once_with(self.manager.image_tag, force=True)
        self.mock_client.api.build.assert_called_once()
        infos = [call_args[0][0] for call_args in mock_log.info.call_args_list]
        self.assertEqual(sum("Successfully built" in msg for msg in infos), 1)

    def test_create_environment_build_error(self):
        self.manager.env_exists = MagicMock(return_value=False)
        self.mock_client.api.build.side_effect = APIError("fail", response=None, explanation="err")
        with self.assertRaises(APIError):
            self.manager.create_environment(force=False)

    def test_run_command_success(self):
        fake_container = MagicMock()
        fake_container.logs.return_value = [b"out1\n", b"out2\n"]
        fake_container.wait.return_value = {"StatusCode": 0}
        self.mock_client.containers.run.return_value = fake_container

        with patch("floatcsep.environments.log") as mock_log:
            self.manager.run_command()

        uid, gid = os.getuid(), os.getgid()
        expected_volumes = {
            os.path.join(self.model_dir, "input"): {'bind': '/app/input', 'mode': 'rw'},
            os.path.join(self.model_dir, "forecasts"): {'bind': '/app/forecasts', 'mode': 'rw'},
        }
        self.mock_client.containers.run.assert_called_once_with(
            self.manager.image_tag,
            remove=False,
            volumes=expected_volumes,
            detach=True,
            user=f"{uid}:{gid}",
        )
        fake_container.logs.assert_called_once_with(stream=True)
        fake_container.wait.assert_called_once()

        info_msgs = [args[0] for args, _ in mock_log.info.call_args_list]
        self.assertTrue(any("out1" in m for m in info_msgs))

    def test_run_command_failure(self):
        fake_container = MagicMock()
        fake_container.logs.return_value = []
        fake_container.wait.return_value = {"StatusCode": 5}
        self.mock_client.containers.run.return_value = fake_container

        with self.assertRaises(RuntimeError):
            self.manager.run_command()

    def test_install_dependencies_noop(self):
        # No exception should be raised
        self.manager.install_dependencies()

if __name__ == "__main__":
    unittest.main()
