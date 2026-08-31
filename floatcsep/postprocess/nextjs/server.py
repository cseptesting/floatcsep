"""Python server launcher for Next.js dashboard."""

import json
import logging
import os
import socket
import subprocess
import threading
import time
import webbrowser
from pathlib import Path
from typing import Optional, Any

from ..panel.manifest import build_manifest
from .runtime import ensure_node_runtime, ensure_nextjs_dependencies
from .schemas import ManifestModel

logger = logging.getLogger(__name__)


def find_free_port() -> int:
    """Find an available port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


def wait_for_server(address: str, port: int, timeout: int = 60) -> bool:
    """
    Wait for the server to be ready by checking if HTTP requests succeed.

    This function first waits for the port to be open, then waits for
    the server to respond with a successful HTTP status.

    Args:
        address: Host address
        port: Port number
        timeout: Maximum time to wait in seconds

    Returns:
        True if server is ready, False if timeout
    """
    import urllib.request
    import urllib.error

    start_time = time.time()
    url = f"http://{address}:{port}/api/manifest"

    # First wait for port to be open
    logger.info("Waiting for port to be open...")
    while time.time() - start_time < timeout:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(1)
                result = sock.connect_ex((address, port))
                if result == 0:
                    logger.info("Port is open, waiting for server to be fully ready...")
                    break
        except (socket.error, OSError):
            pass
        time.sleep(0.5)
    else:
        logger.warning("Port did not open in time")
        return False

    # Now wait for HTTP to respond successfully
    while time.time() - start_time < timeout:
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as response:
                if response.status == 200:
                    logger.info("Server is fully ready!")
                    return True
        except urllib.error.HTTPError as e:
            # Server responded but with an error - might still be compiling
            logger.debug(f"HTTP error {e.code}, waiting...")
        except (urllib.error.URLError, socket.error, OSError) as e:
            # Connection error - server not ready yet
            logger.debug(f"Connection error: {e}, waiting...")
        except Exception as e:
            logger.debug(f"Unexpected error: {e}, waiting...")
        time.sleep(1)

    logger.warning("Server did not become fully ready in time")
    return False


def run_nextjs_app(
    experiment: Any,
    port: int = 0,
    address: str = "localhost",
    show: bool = True,
    title: Optional[str] = None,
    mode: str = "dev",
) -> None:
    """
    Launch the Next.js dashboard for the experiment.

    Args:
        experiment: Experiment instance
        port: Port number (0 = auto-select)
        address: Host address
        show: Open browser automatically
        title: Window title (unused in Next.js)
        mode: 'dev' or 'start' (production)
    """
    # Build manifest
    logger.info("Building experiment manifest...")
    manifest = build_manifest(experiment)

    # Validate using Pydantic
    manifest_model = ManifestModel.model_validate(manifest)

    # Get Next.js app directory
    nextjs_dir = Path(__file__).resolve().parent

    runtime = ensure_node_runtime(nextjs_dir)
    base_env = runtime.apply_to_env(os.environ)
    # Ensure dependencies installed using the detected runtime
    ensure_nextjs_dependencies(
        nextjs_dir,
        npm_cmd=[str(runtime.npm_path)],
        env=base_env,
    )

    # Select port
    if port == 0:
        port = find_free_port()

    # Write manifest to cache for API access
    manifest_path = nextjs_dir / ".cache" / "manifest.json"
    manifest_path.parent.mkdir(exist_ok=True)

    logger.info(f"Writing manifest to {manifest_path}...")
    try:
        with open(manifest_path, "w") as f:
            # Serialize using Pydantic
            json.dump(manifest_model.model_dump(mode="json"), f, indent=2)
        logger.info(
            f"Manifest written successfully ({manifest_path.stat().st_size} bytes)"
        )
    except Exception as e:
        logger.error(f"Failed to write manifest: {e}")
        raise

    # Environment for the Next.js process
    env = base_env.copy()
    env["MANIFEST_PATH"] = str(manifest_path.absolute())
    env["APP_ROOT"] = manifest.app_root
    env["HOSTNAME"] = address
    env["PORT"] = str(port)

    # Construct command
    cmd = [str(runtime.npm_path), "run", mode]

    logger.info(f"Starting Next.js dashboard at http://{address}:{port}")
    logger.info(f"Mode: {mode}")
    logger.debug(f"MANIFEST_PATH: {env['MANIFEST_PATH']}")
    logger.debug(f"APP_ROOT: {env['APP_ROOT']}")

    # Start the Next.js server as a subprocess
    try:
        # Start process without capturing output - let it inherit parent's stdout/stderr
        # This prevents buffering issues and allows real-time output
        process = subprocess.Popen(
            cmd,
            cwd=nextjs_dir,
            env=env,
        )

        # Open browser after server is ready
        if show:

            def open_browser_when_ready():
                logger.info("Waiting for server to be ready...")
                if wait_for_server(address, port, timeout=30):
                    logger.info(f"Opening browser at http://{address}:{port}")
                    webbrowser.open(f"http://{address}:{port}")
                else:
                    logger.warning(
                        "Server did not become ready in time. Browser not opened automatically."
                    )

            threading.Thread(target=open_browser_when_ready, daemon=True).start()

        # Wait for the process to complete
        try:
            return_code = process.wait()
            if return_code != 0:
                logger.error(f"Next.js server exited with code {return_code}")
                raise subprocess.CalledProcessError(return_code, cmd)
        except KeyboardInterrupt:
            logger.info("\nShutting down Next.js server...")
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning(
                    "Server did not terminate gracefully, forcing shutdown..."
                )
                process.kill()
                process.wait()
            logger.info("Server stopped.")

    except FileNotFoundError:
        logger.error(f"Command not found: {cmd[0]}")
        raise
    except Exception as e:
        logger.error(f"Failed to start Next.js server: {e}")
        raise
