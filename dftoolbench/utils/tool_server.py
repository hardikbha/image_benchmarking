"""
tool_server.py — AgentLego-compatible tool server for DFToolBench-I.

This module exposes a thin HTTP server that wraps registered tool instances
so the agent framework can invoke them over a simple REST interface.

Each tool is exposed at:

    POST /tools/<tool_name>/call

with a JSON body containing the tool's input arguments.  The response body
is a JSON object with a ``"result"`` key holding the tool's return value.

A health-check endpoint is available at:

    GET /healthz  → 200  {"status": "ok", "tools": [...]}

Environment variables
---------------------
TOOL_SERVER_HOST
    Bind address (default: ``127.0.0.1``).
TOOL_SERVER_PORT
    TCP port (default: ``8080``).

Usage
-----
As a library::

    from dftoolbench.utils.tool_server import ToolServer
    from dftoolbench.tools.anomaly_detection import AnomalyDetectionTool

    server = ToolServer(port=8080)
    server.register(AnomalyDetectionTool(device="cuda:0"))
    server.serve_forever()

Via the convenience function::

    from dftoolbench.utils.tool_server import launch_tool_server
    launch_tool_server(tools=[AnomalyDetectionTool()], port=8080)

Command-line (for quick testing)::

    python -m dftoolbench.utils.tool_server --port 8080 --tools AnomalyDetectionTool
"""

from __future__ import annotations

import json
import logging
import os
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List, Optional, Type

logger = logging.getLogger(__name__)

_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 8080

# URL prefix for tool calls.
_TOOLS_PREFIX = "/tools/"
_CALL_SUFFIX = "/call"
_HEALTHZ_PATH = "/healthz"


# ---------------------------------------------------------------------------
# ToolServer
# ---------------------------------------------------------------------------


class ToolServer:
    """HTTP server that exposes registered tool instances via a REST API.

    Parameters
    ----------
    host : str, optional
        Bind address (overrides ``TOOL_SERVER_HOST``; default
        ``127.0.0.1``).
    port : int, optional
        TCP port (overrides ``TOOL_SERVER_PORT``; default ``8080``).

    Examples
    --------
    >>> from dftoolbench.tools.anomaly_detection import AnomalyDetectionTool
    >>> server = ToolServer(port=8080)
    >>> server.register(AnomalyDetectionTool())
    >>> server.serve_forever()          # blocks
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
    ) -> None:
        self.host: str = host or os.environ.get("TOOL_SERVER_HOST", _DEFAULT_HOST)
        self.port: int = int(port or os.environ.get("TOOL_SERVER_PORT", _DEFAULT_PORT))
        self._tools: Dict[str, Any] = {}  # tool_name → tool instance

    # ------------------------------------------------------------------
    # Tool registration
    # ------------------------------------------------------------------

    def register(self, tool: Any, name: Optional[str] = None) -> None:
        """Register a tool instance under *name*.

        Parameters
        ----------
        tool : BaseTool-like
            Any object with an ``apply(**kwargs)`` (or ``__call__``) method.
        name : str, optional
            Override the registration name.  Defaults to
            ``type(tool).__name__``.
        """
        tool_name = name or type(tool).__name__
        if tool_name in self._tools:
            logger.warning("Overwriting previously registered tool '%s'.", tool_name)
        self._tools[tool_name] = tool
        logger.info("Registered tool: %s", tool_name)

    def register_many(self, tools: List[Any]) -> None:
        """Register a list of tool instances (name inferred from class).

        Parameters
        ----------
        tools : list
            List of tool instances.
        """
        for tool in tools:
            self.register(tool)

    # ------------------------------------------------------------------
    # Serving
    # ------------------------------------------------------------------

    def serve_forever(self) -> None:
        """Start the HTTP server and block until interrupted."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )

        # Build a handler class that has a reference to *this* ToolServer.
        server_ref = self

        class _Handler(BaseHTTPRequestHandler):
            _server_ref = server_ref

            def log_message(self, fmt: str, *args: object) -> None:  # noqa: N802
                logger.debug("[tool_server] " + fmt, *args)

            def do_GET(self) -> None:  # noqa: N802
                if self.path == _HEALTHZ_PATH:
                    payload = {
                        "status": "ok",
                        "tools": list(self._server_ref._tools.keys()),
                    }
                    self._send_json(200, payload)
                else:
                    self._send_json(404, {"error": "not found"})

            def do_POST(self) -> None:  # noqa: N802
                # Expected path: /tools/<tool_name>/call
                path = self.path.split("?")[0]  # strip query string if any
                if not (
                    path.startswith(_TOOLS_PREFIX) and path.endswith(_CALL_SUFFIX)
                ):
                    self._send_json(404, {"error": "unknown endpoint"})
                    return

                tool_name = path[len(_TOOLS_PREFIX) : -len(_CALL_SUFFIX)]
                if tool_name not in self._server_ref._tools:
                    self._send_json(
                        404,
                        {
                            "error": f"tool '{tool_name}' not registered",
                            "available": list(self._server_ref._tools.keys()),
                        },
                    )
                    return

                length = int(self.headers.get("Content-Length", 0))
                body_bytes = self.rfile.read(length) if length else b""
                try:
                    kwargs = json.loads(body_bytes) if body_bytes else {}
                except json.JSONDecodeError as exc:
                    self._send_json(400, {"error": f"invalid JSON body: {exc}"})
                    return

                tool = self._server_ref._tools[tool_name]
                try:
                    if callable(getattr(tool, "apply", None)):
                        result = tool.apply(**kwargs)
                    else:
                        result = tool(**kwargs)
                    self._send_json(200, {"result": result})
                except Exception as exc:
                    logger.error(
                        "Error calling tool '%s': %s\n%s",
                        tool_name,
                        exc,
                        traceback.format_exc(),
                    )
                    self._send_json(
                        500,
                        {
                            "error": str(exc),
                            "tool": tool_name,
                        },
                    )

            # ---- helpers ----

            def _send_json(self, code: int, payload: dict) -> None:
                encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

        httpd = HTTPServer((self.host, self.port), _Handler)
        logger.info(
            "ToolServer listening on http://%s:%d  tools=%s",
            self.host,
            self.port,
            list(self._tools.keys()),
        )
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            logger.info("ToolServer stopped.")
        finally:
            httpd.server_close()


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def launch_tool_server(
    tools: List[Any],
    host: Optional[str] = None,
    port: Optional[int] = None,
) -> None:
    """Create a :class:`ToolServer`, register *tools*, and start serving.

    This is a convenience wrapper intended for use in scripts and notebooks.

    Parameters
    ----------
    tools : list
        Tool instances to register.
    host : str, optional
        Bind address.
    port : int, optional
        TCP port.
    """
    server = ToolServer(host=host, port=port)
    server.register_many(tools)
    server.serve_forever()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import importlib

    parser = argparse.ArgumentParser(
        description="Start the DFToolBench tool server.",
    )
    parser.add_argument(
        "--host",
        default=None,
        help="Bind address (default: TOOL_SERVER_HOST env var or 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="TCP port (default: TOOL_SERVER_PORT env var or 8080).",
    )
    parser.add_argument(
        "--tools",
        nargs="*",
        default=[],
        metavar="CLASS",
        help=(
            "Tool class names to instantiate from dftoolbench.tools, e.g. "
            "AnomalyDetectionTool DeepfakeDetectionTool"
        ),
    )
    args = parser.parse_args()

    tool_instances: List[Any] = []
    for cls_name in args.tools:
        try:
            module = importlib.import_module("dftoolbench.tools")
            cls: Type[Any] = getattr(module, cls_name)
            tool_instances.append(cls())
            logger.info("Instantiated tool: %s", cls_name)
        except (ImportError, AttributeError) as exc:
            logger.error("Cannot load tool class '%s': %s", cls_name, exc)

    launch_tool_server(tool_instances, host=args.host, port=args.port)
