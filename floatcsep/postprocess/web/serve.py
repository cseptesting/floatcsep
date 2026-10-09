import functools
import logging
import threading
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Union

log = logging.getLogger("floatLogger")


class _Handler(SimpleHTTPRequestHandler):
    extensions_map = {
        **SimpleHTTPRequestHandler.extensions_map,
        ".js": "text/javascript",
        ".mjs": "text/javascript",
        ".json": "application/json",
    }

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, fmt, *args):
        log.debug("%s " + fmt, self.address_string(), *args)


def serve(directory: Union[str, Path], port: int = 8765, open_browser: bool = True) -> None:
    """
    Serves an exported dashboard folder over HTTP and blocks until Ctrl-C.

    Parameters
    ----------
    directory : str or Path
        Folder written by :func:`export_experiment`.
    port : int
        TCP port; 0 picks a free one.
    open_browser : bool
        Open the default browser on the dashboard.
    """
    directory = Path(directory).resolve()
    if not (directory / "manifest.json").exists():
        raise FileNotFoundError(f"No manifest.json in {directory}; run `floatcsep export` first")

    handler = functools.partial(_Handler, directory=str(directory))
    httpd = ThreadingHTTPServer(("localhost", port), handler)
    url = f"http://localhost:{httpd.server_address[1]}/"
    log.info(f"Serving {directory} at {url} (Ctrl-C to stop)")
    if open_browser:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
