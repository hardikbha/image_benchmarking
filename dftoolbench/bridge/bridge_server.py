"""
bridge_server.py — HTTP forwarding proxy for DFToolBench-I.

The server sits between the agent runner (which speaks the OpenAI HTTP API)
and the actual LLM APIs.  Requests are serialised as UUID-named JSON files
and picked up by :mod:`dftoolbench.bridge.bridge_worker`.  Responses are
written back by the worker using the same UUID, and the server returns them
to the agent.

Environment variables
---------------------
BRIDGE_BIND
    Hostname / IP to bind to (default: ``127.0.0.1``).
BRIDGE_PORT
    TCP port to listen on (default: ``9099``).
BRIDGE_REQ_DIR
    Directory into which the server writes request JSON files
    (default: ``/tmp/dftoolbench_bridge/requests``).
BRIDGE_RESP_DIR
    Directory from which the server reads response JSON files
    (default: ``/tmp/dftoolbench_bridge/responses``).
BRIDGE_POLL_INTERVAL
    Seconds between filesystem polls while waiting for a response
    (default: ``0.1``).
BRIDGE_TIMEOUT
    Maximum seconds to wait for a worker response before returning 504
    (default: ``300``).

Routes
------
GET  /healthz
    Returns ``200 OK`` with body ``ok``.
POST /openai/*
    Forwarded to ``https://api.openai.com/<rest>``.
POST /anthropic/*
    Forwarded to ``https://api.anthropic.com/<rest>``.

All other paths return ``404 Not Found``.

Usage
-----
Run as a module::

    python -m dftoolbench.bridge.bridge_server

Or import and call :func:`run_server`::

    from dftoolbench.bridge.bridge_server import run_server
    run_server(host="0.0.0.0", port=9099)
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration (environment variables with defaults)
# ---------------------------------------------------------------------------

_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 9099
_DEFAULT_REQ_DIR = "/tmp/dftoolbench_bridge/requests"
_DEFAULT_RESP_DIR = "/tmp/dftoolbench_bridge/responses"
_DEFAULT_POLL_INTERVAL = 0.1  # seconds
_DEFAULT_TIMEOUT = 300  # seconds


def _get_config() -> dict:
    """Read runtime configuration from environment variables."""
    return {
        "host": os.environ.get("BRIDGE_BIND", _DEFAULT_HOST),
        "port": int(os.environ.get("BRIDGE_PORT", _DEFAULT_PORT)),
        "req_dir": Path(os.environ.get("BRIDGE_REQ_DIR", _DEFAULT_REQ_DIR)),
        "resp_dir": Path(os.environ.get("BRIDGE_RESP_DIR", _DEFAULT_RESP_DIR)),
        "poll_interval": float(
            os.environ.get("BRIDGE_POLL_INTERVAL", _DEFAULT_POLL_INTERVAL)
        ),
        "timeout": float(os.environ.get("BRIDGE_TIMEOUT", _DEFAULT_TIMEOUT)),
    }


# ---------------------------------------------------------------------------
# Atomic filesystem helpers
# ---------------------------------------------------------------------------


def _atomic_write(path: Path, data: dict) -> None:
    """Write *data* as JSON to *path* atomically using a .tmp file + rename.

    Parameters
    ----------
    path : Path
        Destination file path.
    data : dict
        JSON-serialisable payload.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    try:
        tmp_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        os.rename(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _poll_for_response(
    resp_path: Path,
    poll_interval: float,
    timeout: float,
) -> Optional[dict]:
    """Block until *resp_path* exists or *timeout* seconds have elapsed.

    Parameters
    ----------
    resp_path : Path
        Path to the response JSON file written by the worker.
    poll_interval : float
        Sleep interval between filesystem checks (seconds).
    timeout : float
        Maximum wait time (seconds).

    Returns
    -------
    dict or None
        Parsed response dict, or *None* if the timeout was reached.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if resp_path.is_file():
            try:
                payload = json.loads(resp_path.read_text(encoding="utf-8"))
                resp_path.unlink(missing_ok=True)
                return payload
            except (json.JSONDecodeError, OSError):
                # File may still be partially written; retry.
                pass
        time.sleep(poll_interval)
    return None


# ---------------------------------------------------------------------------
# Route parsing
# ---------------------------------------------------------------------------


def _parse_route(path: str) -> Tuple[Optional[str], Optional[str]]:
    """Decompose a request path into ``(provider, tail)``.

    Returns
    -------
    (provider, tail) : tuple[str | None, str | None]
        *provider* is ``"openai"`` or ``"anthropic"``, or *None* for unknown
        paths.  *tail* is the remainder after the provider prefix.
    """
    if path.startswith("/openai/"):
        return "openai", path[len("/openai"):]
    if path.startswith("/anthropic/"):
        return "anthropic", path[len("/anthropic"):]
    return None, None


# ---------------------------------------------------------------------------
# HTTP request handler
# ---------------------------------------------------------------------------


class _BridgeHandler(BaseHTTPRequestHandler):
    """BaseHTTPRequestHandler that implements the bridge logic."""

    # Injected by run_server() so the handler can read config without globals.
    _cfg: dict = {}

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log_message(self, fmt: str, *args: object) -> None:  # noqa: N802
        logger.debug("[bridge] " + fmt, *args)

    # ------------------------------------------------------------------
    # GET /healthz
    # ------------------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/healthz":
            self._send_text(200, "ok")
        else:
            self._send_text(404, "not found")

    # ------------------------------------------------------------------
    # POST /openai/* and /anthropic/*
    # ------------------------------------------------------------------

    def do_POST(self) -> None:  # noqa: N802
        provider, tail = _parse_route(self.path)
        if provider is None:
            self._send_text(404, "unknown route")
            return

        # Read the request body.
        length = int(self.headers.get("Content-Length", 0))
        body_bytes = self.rfile.read(length) if length else b""
        try:
            body = json.loads(body_bytes) if body_bytes else {}
        except json.JSONDecodeError:
            self._send_json(400, {"error": "invalid JSON body"})
            return

        # Build the request envelope written to disk.
        req_id = str(uuid.uuid4())
        cfg = self.__class__._cfg
        req_dir: Path = cfg["req_dir"]
        resp_dir: Path = cfg["resp_dir"]

        # Collect forwarding headers (strip hop-by-hop headers).
        forward_headers: dict = {}
        for header in ("Authorization", "x-api-key", "anthropic-version", "Content-Type"):
            val = self.headers.get(header)
            if val:
                forward_headers[header] = val

        envelope = {
            "id": req_id,
            "provider": provider,
            "path": tail,
            "headers": forward_headers,
            "body": body,
        }

        req_path = req_dir / f"{req_id}.json"
        resp_path = resp_dir / f"{req_id}.json"

        try:
            _atomic_write(req_path, envelope)
        except OSError as exc:
            logger.error("Failed to write request file %s: %s", req_path, exc)
            self._send_json(500, {"error": "bridge write failed", "detail": str(exc)})
            return

        # Wait for the worker to produce a response.
        result = _poll_for_response(
            resp_path,
            poll_interval=cfg["poll_interval"],
            timeout=cfg["timeout"],
        )

        # Clean up the request file if the worker didn't remove it.
        req_path.unlink(missing_ok=True)

        if result is None:
            self._send_json(504, {"error": "bridge timeout: worker did not respond"})
            return

        status_code: int = result.get("status_code", 200)
        response_body: dict = result.get("body", {})
        self._send_json(status_code, response_body)

    # ------------------------------------------------------------------
    # Response helpers
    # ------------------------------------------------------------------

    def _send_text(self, code: int, text: str) -> None:
        encoded = text.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _send_json(self, code: int, payload: dict) -> None:
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_server(
    host: Optional[str] = None,
    port: Optional[int] = None,
    req_dir: Optional[str] = None,
    resp_dir: Optional[str] = None,
) -> None:
    """Start the bridge HTTP server (blocking).

    Parameter values default to environment variable overrides and then to
    the module-level defaults.

    Parameters
    ----------
    host : str, optional
        Bind address (overrides ``BRIDGE_BIND``).
    port : int, optional
        TCP port (overrides ``BRIDGE_PORT``).
    req_dir : str, optional
        Request spool directory (overrides ``BRIDGE_REQ_DIR``).
    resp_dir : str, optional
        Response spool directory (overrides ``BRIDGE_RESP_DIR``).
    """
    cfg = _get_config()
    if host is not None:
        cfg["host"] = host
    if port is not None:
        cfg["port"] = int(port)
    if req_dir is not None:
        cfg["req_dir"] = Path(req_dir)
    if resp_dir is not None:
        cfg["resp_dir"] = Path(resp_dir)

    # Ensure spool directories exist.
    cfg["req_dir"].mkdir(parents=True, exist_ok=True)
    cfg["resp_dir"].mkdir(parents=True, exist_ok=True)

    # Inject config into the handler class.
    _BridgeHandler._cfg = cfg

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    server_address = (cfg["host"], cfg["port"])
    httpd = HTTPServer(server_address, _BridgeHandler)
    logger.info(
        "Bridge server listening on http://%s:%d  (req_dir=%s  resp_dir=%s)",
        cfg["host"],
        cfg["port"],
        cfg["req_dir"],
        cfg["resp_dir"],
    )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Bridge server stopped.")
    finally:
        httpd.server_close()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_server()
