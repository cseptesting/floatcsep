"""Node.js runtime management for floatCSEP Next.js dashboard."""

import logging
import os
import platform
import re
import shutil
import stat
import subprocess
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
from urllib import request

logger = logging.getLogger(__name__)

MIN_NODE_VERSION = (18, 17, 0)
BUNDLED_NODE_VERSION = "20.11.1"


@dataclass
class NodeRuntime:
    """Represents a runnable Node.js installation."""

    node_path: Path
    npm_path: Path
    bin_dir: Path
    source: str

    def apply_to_env(self, env: dict) -> dict:
        """Return a copy of the environment with this runtime prepended to PATH."""

        current_path = env.get("PATH", "")
        updated = env.copy()
        updated["PATH"] = (
            f"{self.bin_dir}{os.pathsep}{current_path}"
            if current_path
            else str(self.bin_dir)
        )
        return updated


def parse_node_version(raw: str) -> Optional[tuple[int, int, int]]:
    match = re.match(r"v?(\d+)\.(\d+)\.(\d+)", raw.strip())
    if not match:
        return None
    return tuple(int(part) for part in match.groups())


def get_system_node_runtime() -> Optional[NodeRuntime]:
    node_cmd = shutil.which("node")
    npm_cmd = shutil.which("npm")
    if not node_cmd or not npm_cmd:
        return None
    try:
        result = subprocess.run(
            [node_cmd, "--version"], capture_output=True, text=True, check=True
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    version = parse_node_version(result.stdout)
    if not version or version < MIN_NODE_VERSION:
        return None
    return NodeRuntime(
        node_path=Path(node_cmd),
        npm_path=Path(npm_cmd),
        bin_dir=Path(node_cmd).parent,
        source="system",
    )


def _node_dist_name() -> tuple[str, str]:
    system = platform.system().lower()
    machine = platform.machine().lower()
    if system == "linux":
        if machine in ("x86_64", "amd64"):
            return "linux-x64", ".tar.xz"
        if machine in ("aarch64", "arm64"):
            return "linux-arm64", ".tar.xz"
    elif system == "darwin":
        if machine == "arm64":
            return "darwin-arm64", ".tar.xz"
        if machine in ("x86_64", "amd64"):
            return "darwin-x64", ".tar.xz"
    elif system == "windows":
        if machine in ("x86_64", "amd64"):
            return "win-x64", ".zip"
    raise RuntimeError(
        f"Unsupported platform '{platform.system()} {platform.machine()}'. "
        "Please install Node.js 20+ manually."
    )


def _download_node_archive(target: Path, url: str) -> None:
    logger.info("Downloading Node.js runtime from %s", url)
    target.parent.mkdir(parents=True, exist_ok=True)
    with request.urlopen(url) as response, open(target, "wb") as handle:
        shutil.copyfileobj(response, handle)


def _extract_node_archive(archive: Path, destination: Path) -> Path:
    logger.info("Extracting Node.js runtime to %s", destination)
    destination.mkdir(parents=True, exist_ok=True)
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(destination)
    else:
        # Handles .tar.xz
        with tarfile.open(archive, mode="r:*") as tf:
            tf.extractall(destination)
    # Find the extracted directory (node-vXX-<platform>)
    for child in destination.iterdir():
        if child.is_dir() and child.name.startswith(f"node-v{BUNDLED_NODE_VERSION}"):
            return child
    raise RuntimeError("Failed to locate extracted Node.js runtime")


def ensure_bundled_node(nextjs_dir: Path) -> NodeRuntime:
    platform_tag, archive_ext = _node_dist_name()
    cache_dir = nextjs_dir / ".cache" / "node-runtime"
    extract_root = cache_dir / f"node-v{BUNDLED_NODE_VERSION}-{platform_tag}"
    if extract_root.exists():
        logger.info("Using cached Node.js runtime at %s", extract_root)
    else:
        archive_name = f"node-v{BUNDLED_NODE_VERSION}-{platform_tag}{archive_ext}"
        download_url = (
            f"https://nodejs.org/dist/v{BUNDLED_NODE_VERSION}/{archive_name}"
        )
        archive_path = cache_dir / archive_name
        _download_node_archive(archive_path, download_url)
        extracted = _extract_node_archive(archive_path, cache_dir)
        extracted.rename(extract_root)
        archive_path.unlink(missing_ok=True)

    if platform.system().lower() == "windows":
        node_path = extract_root / "node.exe"
        npm_path = extract_root / "npm.cmd"
        bin_dir = extract_root
    else:
        bin_dir = extract_root / "bin"
        node_path = bin_dir / "node"
        npm_path = bin_dir / "npm"
    for path in (node_path, npm_path):
        if not path.exists():
            raise RuntimeError(f"Bundled Node.js binary missing: {path}")
        mode = path.stat().st_mode
        path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return NodeRuntime(
        node_path=node_path, npm_path=npm_path, bin_dir=bin_dir, source="bundled"
    )


def ensure_node_runtime(nextjs_dir: Path) -> NodeRuntime:
    runtime = get_system_node_runtime()
    if runtime:
        logger.info("Detected Node.js %s from system PATH", runtime.node_path)
        return runtime
    logger.warning(
        "Node.js %s or newer not found. Downloading a scoped runtime (v%s).",
        ".".join(str(part) for part in MIN_NODE_VERSION),
        BUNDLED_NODE_VERSION,
    )
    return ensure_bundled_node(nextjs_dir)


def ensure_nextjs_dependencies(
    nextjs_dir: Path, npm_cmd: List[str], env: dict
) -> None:
    """Install Node dependencies if needed."""
    node_modules = nextjs_dir / "node_modules"
    if node_modules.exists():
        return
    logger.info("Installing Next.js dependencies (this may take a few minutes)...")
    try:
        subprocess.run(
            npm_cmd + ["install"],
            cwd=nextjs_dir,
            check=True,
            env=env,
        )
    except subprocess.CalledProcessError as exc:
        logger.error("Failed to install dependencies: %s", exc)
        raise RuntimeError(
            "Could not install Next.js dependencies automatically. "
            "Please ensure network access is available or install them manually."
        )
