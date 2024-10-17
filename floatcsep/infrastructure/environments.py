import configparser
import hashlib
import logging
import os
import shutil
import subprocess
import sys
import venv
from abc import ABC, abstractmethod
from typing import Union

from packaging.specifiers import SpecifierSet

log = logging.getLogger("floatLogger")


class EnvironmentManager(ABC):
    """
    Abstract base class for managing different types of environments. This class defines the
    interface for creating, checking existence, running commands, and installing dependencies in
    various environment types.
    """

    @abstractmethod
    def __init__(self, base_name: str, model_directory: str):
        """
        Initializes the environment manager with a base name and model directory.

        Args:
            base_name (str): The base name for the environment.
            model_directory (str): The directory containing the model files.
        """
        self.base_name = base_name
        self.model_directory = model_directory

    @abstractmethod
    def create_environment(self, force=False):
        """
        Creates the environment. If 'force' is True, it will remove any existing environment
        with the same name before creating a new one.

        Args:
            force (bool): Whether to forcefully remove an existing environment and create it
             again
        """
        pass

    @abstractmethod
    def env_exists(self):
        """
        Checks if the environment already exists.

        Returns:
            bool: True if the environment exists, False otherwise.
        """
        pass

    @abstractmethod
    def run_command(self, command):
        """
        Executes a command within the context of the environment.

        Args:
            command (str): The command to be executed.
        """
        pass

    @abstractmethod
    def install_dependencies(self):
        """
        Installs the necessary dependencies for the environment based on the specified
        configuration or requirements.
        """
        pass

    def generate_env_name(self) -> str:
        """
        Generates a unique environment name by hashing the model directory and appending it
        to the base name.

        Returns:
            str: A unique name for the environment.
        """
        dir_hash = hashlib.md5(self.model_directory.encode()).hexdigest()[:8]
        return f"{self.base_name}_{dir_hash}"


class CondaManager(EnvironmentManager):
    """
    Manages a conda (or mamba) environment, providing methods to create, check and manipulate
    conda environments specifically.
    """

    def __init__(self, base_name: str, model_directory: str):
        """
        Initializes the Conda environment manager with the specified base name and model
        directory. It also generates the environment name and detects the package manager (conda
        or mamba) to install dependencies.

        Args:
            base_name (str): The base name, i.e., model name, for the conda environment.
            model_directory (str): The directory containing the model files.
        """
        self.base_name = base_name.replace(" ", "_")
        self.model_directory = model_directory
        self.env_name = self.generate_env_name()
        self.package_manager = self.detect_package_manager()

    @staticmethod
    def detect_package_manager():
        """
        Detects whether 'mamba' or 'conda' is available as the package manager.

        Returns:
            str: The name of the detected package manager ('mamba' or 'conda').
        """
        if shutil.which("mamba"):
            log.info("Mamba detected, using mamba as package manager.")
            return "mamba"
        log.info("Mamba not detected, using conda as package manager.")
        return "conda"

    def create_environment(self, force=False):
        """
        Creates a conda environment using either an environment.yml file or the specified
        Python version in setup.py/setup.cfg or project/toml. If 'force' is True, any existing
        environment with the same name will be removed first.

        Args:
            force (bool): Whether to forcefully remove an existing environment.
        """
        if force and self.env_exists():
            log.info(f"Removing existing conda environment: {self.env_name}")
            subprocess.run(
                [
                    self.package_manager,
                    "env",
                    "remove",
                    "--name",
                    self.env_name,
                    "--yes",
                ]
            )

        if not self.env_exists():
            env_file = os.path.join(self.model_directory, "environment.yml")
            if os.path.exists(env_file):
                log.info(f"Creating sub-conda environment {self.env_name} from environment.yml")
                subprocess.run(
                    [
                        self.package_manager,
                        "env",
                        "create",
                        "--name",
                        self.env_name,
                        "--file",
                        env_file,
                    ]
                )
            else:
                python_version = self.detect_python_version()
                log.info(f"Creating sub-conda env {self.env_name} with Python {python_version}")
                subprocess.run(
                    [
                        self.package_manager,
                        "create",
                        "--name",
                        self.env_name,
                        "--yes",
                        f"python={python_version}",
                    ]
                )
            log.info(f"\tSub-conda environment created: {self.env_name}")
            self.install_dependencies()

    def env_exists(self) -> bool:
        """
        Checks if the conda environment exists by querying the list of existing conda
        environments.

        Returns:
            bool: True if the conda environment exists, False otherwise.
        """
        result = subprocess.run(["conda", "env", "list"], stdout=subprocess.PIPE)
        return self.env_name in result.stdout.decode()

    def detect_python_version(self) -> str:
        """
        Determines the required Python version from setup files in the model directory. It
        checks 'setup.py', 'pyproject.toml', and 'setup.cfg' (in that order), for version
        specifications.

        Returns:
            str: The detected or default Python version.
        """
        setup_py = os.path.join(self.model_directory, "setup.py")
        pyproject_toml = os.path.join(self.model_directory, "pyproject.toml")
        setup_cfg = os.path.join(self.model_directory, "setup.cfg")
        current_python_version = (
            f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        )

        def parse_version(version_str):
            # Extract the first valid version number
            import re

            match = re.search(r"\d+(.\d+)*", version_str)
            return match.group(0) if match else current_python_version

        def is_version_compatible(requirement, current_version):
            try:
                specifier = SpecifierSet(requirement)
                return current_version in specifier
            except Exception as e:
                log.error(f"Invalid specifier: {requirement}. Error: {e}")
                return False

        if os.path.exists(setup_py):
            with open(setup_py) as f:
                for line in f:
                    if "python_requires" in line:
                        required_version = line.split("=")[1].strip()
                        if is_version_compatible(required_version, current_python_version):
                            log.info(f"Using current Python version: {current_python_version}")
                            return current_python_version
                        return parse_version(required_version)

        if os.path.exists(pyproject_toml):
            with open(pyproject_toml) as f:
                for line in f:
                    if "python" in line and "=" in line:
                        required_version = line.split("=")[1].strip()
                        if is_version_compatible(required_version, current_python_version):
                            log.info(f"Using current Python version: {current_python_version}")
                            return current_python_version
                        return parse_version(required_version)

        if os.path.exists(setup_cfg):
            config = configparser.ConfigParser()
            config.read(setup_cfg)
            if "options" in config and "python_requires" in config["options"]:
                required_version = config["options"]["python_requires"].strip()
                if is_version_compatible(required_version, current_python_version):
                    log.info(f"Using current Python version: {current_python_version}")
                    return current_python_version
                return parse_version(required_version)

        return current_python_version

    def install_dependencies(self) -> None:
        """
        Installs dependencies in the conda environment using pip, based on the model setup
        file.
        """
        log.info(f"Installing dependencies in conda environment: {self.env_name}")
        cmd = [
            self.package_manager,
            "run",
            "-n",
            self.env_name,
            "pip",
            "install",
            "-e",
            self.model_directory,
        ]
        subprocess.run(cmd, check=True)

    def run_command(self, command) -> None:
        """
        Runs a specified command within the conda environment.

        Args:
            command (str): The command to be executed in the conda environment.
        """
        cmd = [
            "bash",
            "-c",
            f"{self.package_manager} run --live-stream -n {self.env_name} {command}",
        ]

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )
        for line in process.stdout:
            stripped_line = line.strip()
            log.info(f"[{self.base_name}]: " + stripped_line)
        process.wait()


class VenvManager(EnvironmentManager):
    """
    Manages a virtual environment created using Python's venv module. Provides methods to
    create, check, and manipulate virtual environments.
    """

    def __init__(self, base_name: str, model_directory: str) -> None:
        """
        Initializes the virtual environment manager with the specified base name and model
        directory.

        Args:
            base_name (str): The base name (i.e., model name) for the virtual environment.
            model_directory (str): The directory containing the model files.
        """

        self.base_name = base_name.replace(" ", "_")
        self.model_directory = model_directory
        self.env_name = self.generate_env_name()
        self.env_path = os.path.join(model_directory, self.env_name)

    def create_environment(self, force=False):
        """
        Creates a virtual environment in the specified model directory. If 'force' is True,
        any existing virtual environment will be removed before creation.

        Args:
            force (bool): Whether to forcefully remove an existing virtual environment.
        """
        if force and self.env_exists():
            log.info(f"Removing existing virtual environment: {self.env_name}")
            shutil.rmtree(self.env_path)

        if not self.env_exists():
            log.info(f"Creating virtual environment: {self.env_name}")
            venv.create(self.env_path, with_pip=True)
            log.info(f"\tVirtual environment created: {self.env_name}")
            self.install_dependencies()

    def env_exists(self) -> bool:
        """
        Checks if the virtual environment exists by verifying the presence of its directory.

        Returns:
            bool: True if the virtual environment exists, False otherwise.
        """
        return os.path.isdir(self.env_path)

    def install_dependencies(self) -> None:
        """
        Installs dependencies in the virtual environment using pip, based on the model
        directory's configuration.
        """
        log.info(f"Installing dependencies in virtual environment: {self.env_name}")
        pip_executable = os.path.join(self.env_path, "bin", "pip")
        cmd = f"{pip_executable} install -e {os.path.abspath(self.model_directory)}"
        self.run_command(cmd)

    def run_command(self, command) -> None:
        """
        Executes a specified command in the virtual environment and logs the output.

        Args:
            command (str): The command to be executed in the virtual environment.
        """
        activate_script = os.path.join(self.env_path, "bin", "activate")

        virtualenv = os.environ.copy()
        virtualenv.pop("PYTHONPATH", None)
        virtualenv["VIRTUAL_ENV"] = self.env_path
        virtualenv["PATH"] = (
            os.path.join(self.env_path, "bin") + os.pathsep + virtualenv.get("PATH", "")
        )

        full_command = f"bash -c 'source \"{activate_script}\"' && {command}"

        process = subprocess.Popen(
            full_command,
            shell=True,
            env=virtualenv,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )
        for line in process.stdout:
            stripped_line = line.strip()
            log.info(f"[{self.base_name}]: " + stripped_line)
        process.wait()


class DockerManager(EnvironmentManager):
    """
    Manages a Docker environment, providing methods to create, check and manipulate Docker
    containers for the environment.
    """

    def __init__(self, base_name: str, model_directory: str) -> None:
        self.base_name = base_name
        self.model_directory = model_directory

    def create_environment(self, force=False) -> None:
        pass

    def env_exists(self) -> None:
        pass

    def run_command(self, command) -> None:
        pass

    def install_dependencies(self) -> None:
        pass


class EnvironmentFactory:
    """Factory class for creating instances of environment managers based on the specified
    type."""

    @staticmethod
    def get_env(
        build: str = None, model_name: str = "model", model_path: str = None
    ) -> EnvironmentManager:
        """
        Returns an instance of an environment manager based on the specified build type. It
        checks the current environment type and can return a conda, venv, or Docker environment
        manager.

        Args:
            build (str): The desired type of environment ('conda', 'venv', or 'docker').
            model_name (str): The name of the model for which the environment is being created.
            model_path (str): The path to the model directory.

        Returns:
            EnvironmentManager: An instance of the appropriate environment manager.

        Raises:
            Exception: If an invalid environment type is specified.
        """
        run_env = EnvironmentFactory.check_environment_type()
        if run_env != build and build and build != "docker":
            log.warning(
                f"Selected build environment ({build}) for this model is different than that of"
                f" the experiment run. Consider selecting the same environment."
            )
        if build in ["conda", "micromamba"] or (
            not build and run_env in ["conda", "micromamba"]
        ):
            return CondaManager(
                base_name=f"{model_name}",
                model_directory=os.path.abspath(model_path),
            )
        elif build == "venv" or (not build and run_env == "venv"):
            return VenvManager(
                base_name=f"{model_name}",
                model_directory=os.path.abspath(model_path),
            )
        elif build == "docker":
            return DockerManager(
                base_name=f"{model_name}",
                model_directory=os.path.abspath(model_path),
            )
        else:
            raise Exception(
                "Wrong environment selection. Please choose between "
                '"conda", "venv" or "docker".'
            )

    @staticmethod
    def check_environment_type() -> Union[str, None]:
        if "VIRTUAL_ENV" in os.environ:
            log.info("Detected virtual environment.")
            return "venv"
        try:
            result = subprocess.run(
                ["conda", "info"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if result.returncode == 0:
                log.info("Detected conda environment.")
                return "conda"
            else:
                log.warning(
                    "Conda command failed with return code: {}".format(result.returncode)
                )
        except FileNotFoundError:
            log.warning("Conda not found in PATH.")

        try:
            result = subprocess.run(
                ["micromamba", "info"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if result.returncode == 0:
                log.info("Detected micromamba environment.")
                return "micromamba"
            else:
                log.warning(
                    "Micromamba command failed with return code: {}".format(result.returncode)
                )
        except FileNotFoundError:
            log.warning("Micromamba not found in PATH.")

        return None
