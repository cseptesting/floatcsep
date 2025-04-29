import shutil
import unittest
from pathlib import Path
from unittest.mock import patch

import docker
from floatcsep.model import TimeDependentModel

## Uncomment to output logs
# import logging
# from floatcsep.infrastructure.logger import setup_logger, set_console_log_level
# setup_logger()
# log = logging.getLogger("floatLogger")
# set_console_log_level("DEBUG")


try:
    # Try to create a docker client
    client = docker.from_env()
    client.ping()  # Check if Docker daemon is responding
    DOCKER_AVAILABLE = True
except Exception:
    DOCKER_AVAILABLE = False

skip_docker = unittest.skipUnless(DOCKER_AVAILABLE, "Docker is not available")

@skip_docker
class TestModelDockerIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.base_path = Path(__file__).resolve().parent.parent / "artifacts" / "models" / "docker_test"
        cls.client = docker.from_env()

    def tearDown(self):
        # Clean up containers and images matching the test prefix
        prefix = "testdocker_"
        for container in self.client.containers.list(all=True):
            if container.name.startswith(prefix):
                try:
                    container.remove(force=True)
                except Exception:
                    pass

        for image in self.client.images.list():
            for tag in image.tags:
                if tag.startswith(prefix):
                    try:
                        self.client.images.remove(image.id, force=True)
                    except Exception:
                        pass

        # Clean up created input/ and forecasts/ directories
        for test_dir in self.base_path.glob("*"):
            input_dir = test_dir / "input"
            output_dir = test_dir / "forecasts"
            for path in (input_dir, output_dir):
                if path.exists() and path.is_dir():
                    try:
                        for child in path.glob("*"):
                            if child.is_file():
                                child.unlink()
                            elif child.is_dir():
                                shutil.rmtree(child)
                        path.rmdir()
                    except Exception:
                        pass


    def _make_model(self, subfolder: str, tag: str):
        model_dir = self.base_path / subfolder
        return TimeDependentModel(
            name=tag,
            model_path=".",
            force_build=True,
            func="echo Done",  # This won't be used unless the Dockerfile defines an entrypoint
            build="docker",
            workdir=str(model_dir),
        )

    @patch("floatcsep.infrastructure.registries.ModelFileRegistry.build_tree")
    def test_valid_model(self, mock_registry):
        model = self._make_model("valid", "testdocker_valid")
        model.stage()
        model.environment.run_command()  # Should succeed with no exceptions

    @patch("floatcsep.infrastructure.registries.ModelFileRegistry.build_tree")
    def test_invalid_image_build_fails(self, mock_registry):
        model = self._make_model("invalid_image", "testdocker_invalid_image")
        with self.assertRaises(RuntimeError) as err:
            model.environment.create_environment(force=True)
        self.assertIn("Docker build error", str(err.exception))

    @patch("floatcsep.infrastructure.registries.ModelFileRegistry.build_tree")
    def test_invalid_entrypoint_fails_to_run(self, mock_registry):
        model = self._make_model("invalid_entrypoint", "testdocker_invalid_entrypoint")
        model.stage()
        with self.assertRaises(RuntimeError) as err:
            model.environment.run_command()
        self.assertIn("exited with code", str(err.exception))

    @patch("floatcsep.infrastructure.registries.ModelFileRegistry.build_tree")
    def test_invalid_permission_fails_to_run(self, mock_registry):
        model = self._make_model("invalid_permission", "testdocker_invalid_permission")
        model.stage()
        with self.assertRaises(RuntimeError) as err:
            model.environment.run_command()
        self.assertIn("exited with code", str(err.exception))

    @patch("floatcsep.infrastructure.registries.ModelFileRegistry.build_tree")
    def test_valid_custom_uid_gid(self, mock_registry):
        # todo: look into it
        model = self._make_model("valid_custom_uid-gid", "testdocker_uid_gid")
        model.stage()
        model.environment.run_command()  # Should succeed and output UID=1234


if __name__ == "__main__":
    unittest.main()