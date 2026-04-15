"""
bridge_worker.py — API call worker for the DFToolBench-I bridge.

The worker polls the request spool directory written by
:mod:`dftoolbench.bridge.bridge_server`, makes the actual API calls, and
writes response JSON files back to the response directory so the server can
return them to the agent.

Supported backends
------------------
* **Anthropic** — auto-detected when the model name contains ``"claude"``.
  The worker converts the incoming OpenAI-style ``/v1/chat/completions``
  body into the Anthropic Messages API format and calls
  ``https://api.anthropic.com/v1/messages``.
* **OpenAI** — used for all other models.  The request body is forwarded
  as-is to ``https://api.openai.com<path>``.

API keys
--------
Keys are resolved in the following order (for each provider):

1. Environment variable (``OPENAI_API_KEY`` / ``ANTHROPIC_API_KEY``).
2. Key file path in environment variable
   (``OPENAI_KEY_FILE`` / ``ANTHROPIC_KEY_FILE``).
3. Default key file in ``~/.config/dftoolbench/<provider>.key``.

Environment variables
---------------------
BRIDGE_REQ_DIR
    Request spool directory (default: ``/tmp/dftoolbench_bridge/requests``).
BRIDGE_RESP_DIR
    Response spool directory (default: ``/tmp/dftoolbench_bridge/responses``).
BRIDGE_POLL_INTERVAL
    Seconds to sleep between directory scans (default: ``0.2``).
OPENAI_API_KEY / OPENAI_KEY_FILE
    OpenAI credentials.
ANTHROPIC_API_KEY / ANTHROPIC_KEY_FILE
    Anthropic credentials.

Usage
-----
Run as a module::

    python -m dftoolbench.bridge.bridge_worker

Or import and call :func:`run_worker`::

    from dftoolbench.bridge.bridge_worker import run_worker
    run_worker()
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_OPENAI_BASE = "https://api.openai.com"
_ANTHROPIC_BASE = "https://api.anthropic.com"
_ANTHROPIC_MESSAGES_PATH = "/v1/messages"
_DEFAULT_ANTHROPIC_VERSION = "2023-06-01"

_DEFAULT_REQ_DIR = "/tmp/dftoolbench_bridge/requests"
_DEFAULT_RESP_DIR = "/tmp/dftoolbench_bridge/responses"
_DEFAULT_POLL_INTERVAL = 0.2  # seconds

_CLAUDE_MODEL_SUBSTRING = "claude"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def _get_config() -> dict:
    return {
        "req_dir": Path(os.environ.get("BRIDGE_REQ_DIR", _DEFAULT_REQ_DIR)),
        "resp_dir": Path(os.environ.get("BRIDGE_RESP_DIR", _DEFAULT_RESP_DIR)),
        "poll_interval": float(
            os.environ.get("BRIDGE_POLL_INTERVAL", _DEFAULT_POLL_INTERVAL)
        ),
    }


# ---------------------------------------------------------------------------
# API key helpers
# ---------------------------------------------------------------------------


def _read_key_file(path: str) -> Optional[str]:
    """Return the first non-empty line of *path*, stripped, or None."""
    try:
        text = Path(path).read_text(encoding="utf-8").strip()
        return text or None
    except OSError:
        return None


def _resolve_api_key(
    env_key: str,
    env_file: str,
    default_file: str,
) -> Optional[str]:
    """Resolve an API key from env var, key-file env var, or default file.

    Parameters
    ----------
    env_key : str
        Environment variable that may hold the key directly.
    env_file : str
        Environment variable that holds a *path* to a key file.
    default_file : str
        Fallback path (``~``-expanded) to a key file.

    Returns
    -------
    str or None
    """
    key = os.environ.get(env_key, "").strip()
    if key:
        return key
    key_file = os.environ.get(env_file, "")
    if key_file:
        key = _read_key_file(key_file) or ""
        if key:
            return key
    return _read_key_file(os.path.expanduser(default_file))


def get_openai_key() -> Optional[str]:
    """Return the OpenAI API key, or *None* if unavailable."""
    return _resolve_api_key(
        "OPENAI_API_KEY",
        "OPENAI_KEY_FILE",
        "~/.config/dftoolbench/openai.key",
    )


def get_anthropic_key() -> Optional[str]:
    """Return the Anthropic API key, or *None* if unavailable."""
    return _resolve_api_key(
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_KEY_FILE",
        "~/.config/dftoolbench/anthropic.key",
    )


# ---------------------------------------------------------------------------
# OpenAI → Anthropic format conversion
# ---------------------------------------------------------------------------


def _openai_to_anthropic(body: Dict[str, Any]) -> Dict[str, Any]:
    """Convert an OpenAI ``/v1/chat/completions`` body to Anthropic Messages API.

    Handles:
    * ``messages`` list (role / content)
    * ``max_tokens`` / ``max_completion_tokens``
    * ``temperature``, ``top_p``
    * ``tools`` (OpenAI function-calling → Anthropic tool definitions)
    * ``tool_choice``

    A ``system`` prompt is extracted from any message whose role is
    ``"system"`` and placed in the top-level ``"system"`` field.

    Parameters
    ----------
    body : dict
        OpenAI-style request body.

    Returns
    -------
    dict
        Anthropic Messages API request body.
    """
    anthropic_body: Dict[str, Any] = {}
    anthropic_body["model"] = body.get("model", "claude-3-5-sonnet-20241022")

    # max_tokens is required by Anthropic.
    max_tokens = (
        body.get("max_tokens")
        or body.get("max_completion_tokens")
        or 4096
    )
    anthropic_body["max_tokens"] = max_tokens

    if "temperature" in body:
        anthropic_body["temperature"] = body["temperature"]
    if "top_p" in body:
        anthropic_body["top_p"] = body["top_p"]

    # Split system messages from the conversation.
    system_parts: list = []
    messages: list = []
    for msg in body.get("messages", []):
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "system":
            if isinstance(content, str):
                system_parts.append(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        system_parts.append(block.get("text", ""))
        else:
            anthropic_role = "assistant" if role == "assistant" else "user"
            messages.append({"role": anthropic_role, "content": content})

    if system_parts:
        anthropic_body["system"] = "\n\n".join(system_parts)

    anthropic_body["messages"] = messages

    # Tool definitions.
    if "tools" in body:
        anthropic_tools = []
        for tool in body["tools"]:
            if tool.get("type") == "function":
                fn = tool.get("function", {})
                anthropic_tools.append(
                    {
                        "name": fn.get("name", ""),
                        "description": fn.get("description", ""),
                        "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
                    }
                )
        if anthropic_tools:
            anthropic_body["tools"] = anthropic_tools

    if "tool_choice" in body:
        tc = body["tool_choice"]
        if isinstance(tc, str):
            if tc == "auto":
                anthropic_body["tool_choice"] = {"type": "auto"}
            elif tc == "none":
                anthropic_body["tool_choice"] = {"type": "none"}
            elif tc == "required":
                anthropic_body["tool_choice"] = {"type": "any"}
        elif isinstance(tc, dict) and tc.get("type") == "function":
            anthropic_body["tool_choice"] = {
                "type": "tool",
                "name": tc["function"]["name"],
            }

    return anthropic_body


def _anthropic_to_openai_response(anthropic_resp: Dict[str, Any]) -> Dict[str, Any]:
    """Convert an Anthropic Messages API response to OpenAI chat completion format.

    Parameters
    ----------
    anthropic_resp : dict
        Raw response body from Anthropic.

    Returns
    -------
    dict
        OpenAI-compatible ``ChatCompletion`` response dict.
    """
    text_parts: list[str] = []
    tool_calls: list = []

    for block in anthropic_resp.get("content", []):
        btype = block.get("type", "")
        if btype == "text":
            text_parts.append(block.get("text", ""))
        elif btype == "tool_use":
            tool_calls.append(
                {
                    "id": block.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": block.get("name", ""),
                        "arguments": json.dumps(block.get("input", {})),
                    },
                }
            )

    message: Dict[str, Any] = {"role": "assistant", "content": "\n".join(text_parts) or None}
    if tool_calls:
        message["tool_calls"] = tool_calls

    finish_reason_map = {
        "end_turn": "stop",
        "max_tokens": "length",
        "tool_use": "tool_calls",
        "stop_sequence": "stop",
    }
    stop_reason = anthropic_resp.get("stop_reason", "end_turn")
    finish_reason = finish_reason_map.get(stop_reason, "stop")

    usage = anthropic_resp.get("usage", {})
    return {
        "id": anthropic_resp.get("id", ""),
        "object": "chat.completion",
        "model": anthropic_resp.get("model", ""),
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": usage.get("input_tokens", 0),
            "completion_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
        },
    }


# ---------------------------------------------------------------------------
# HTTP call helper
# ---------------------------------------------------------------------------


def _http_post(url: str, headers: dict, body: dict) -> Tuple[int, dict]:
    """Perform a blocking HTTP POST and return ``(status_code, response_body)``.

    Parameters
    ----------
    url : str
        Full URL to POST to.
    headers : dict
        HTTP headers to include.
    body : dict
        JSON-serialisable request body.

    Returns
    -------
    (int, dict)
        HTTP status code and parsed JSON response body.
    """
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    for k, v in headers.items():
        req.add_header(k, v)
    req.add_header("Content-Type", "application/json")

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            status = resp.status
            resp_body = json.loads(resp.read().decode("utf-8"))
        return status, resp_body
    except urllib.error.HTTPError as exc:
        try:
            err_body = json.loads(exc.read().decode("utf-8"))
        except Exception:
            err_body = {"error": str(exc)}
        return exc.code, err_body
    except Exception as exc:
        return 502, {"error": f"worker HTTP error: {exc}"}


# ---------------------------------------------------------------------------
# Request processing
# ---------------------------------------------------------------------------


def _is_claude_model(body: Dict[str, Any]) -> bool:
    """Return True if the model field looks like a Claude model."""
    model = body.get("model", "").lower()
    return _CLAUDE_MODEL_SUBSTRING in model


def _process_request(envelope: Dict[str, Any]) -> Dict[str, Any]:
    """Dispatch a request envelope and return a response envelope.

    Parameters
    ----------
    envelope : dict
        Request envelope written by the bridge server::

            {
              "id": "<uuid>",
              "provider": "openai" | "anthropic",
              "path": "/v1/chat/completions",
              "headers": {...},
              "body": {...}
            }

    Returns
    -------
    dict
        Response envelope::

            {"status_code": <int>, "body": {...}}
    """
    provider: str = envelope.get("provider", "openai")
    path: str = envelope.get("path", "/v1/chat/completions")
    headers: dict = dict(envelope.get("headers", {}))
    body: dict = dict(envelope.get("body", {}))

    # Auto-detect Claude models even when routed through /openai/*
    if provider == "openai" and _is_claude_model(body):
        provider = "anthropic"
        logger.debug("Auto-detected Claude model; switching to Anthropic backend.")

    if provider == "anthropic":
        api_key = get_anthropic_key()
        if api_key:
            headers["x-api-key"] = api_key
        headers.setdefault("anthropic-version", _DEFAULT_ANTHROPIC_VERSION)

        # Convert OpenAI format → Anthropic format if coming from /openai/*
        # or if the body looks like an OpenAI chat completions body.
        if "messages" in body and "content" not in body:
            anthropic_body = _openai_to_anthropic(body)
        else:
            anthropic_body = body

        url = _ANTHROPIC_BASE + _ANTHROPIC_MESSAGES_PATH
        status, resp_body = _http_post(url, headers, anthropic_body)

        # If the caller expects an OpenAI-compatible response, convert back.
        if status == 200 and "content" in resp_body:
            resp_body = _anthropic_to_openai_response(resp_body)

    else:
        # OpenAI (or OpenAI-compatible) endpoint.
        api_key = get_openai_key()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        url = _OPENAI_BASE + path
        status, resp_body = _http_post(url, headers, body)

    return {"status_code": status, "body": resp_body}


# ---------------------------------------------------------------------------
# Atomic write for responses
# ---------------------------------------------------------------------------


def _atomic_write(path: Path, data: dict) -> None:
    """Write *data* as JSON to *path* atomically (tmp + rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    try:
        tmp_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        os.rename(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


# ---------------------------------------------------------------------------
# Main polling loop
# ---------------------------------------------------------------------------


def run_worker(
    req_dir: Optional[str] = None,
    resp_dir: Optional[str] = None,
    poll_interval: Optional[float] = None,
) -> None:
    """Start the bridge worker polling loop (blocking).

    Parameters
    ----------
    req_dir : str, optional
        Request spool directory (overrides ``BRIDGE_REQ_DIR``).
    resp_dir : str, optional
        Response spool directory (overrides ``BRIDGE_RESP_DIR``).
    poll_interval : float, optional
        Seconds between directory scans (overrides ``BRIDGE_POLL_INTERVAL``).
    """
    cfg = _get_config()
    if req_dir is not None:
        cfg["req_dir"] = Path(req_dir)
    if resp_dir is not None:
        cfg["resp_dir"] = Path(resp_dir)
    if poll_interval is not None:
        cfg["poll_interval"] = float(poll_interval)

    req_dir_path: Path = cfg["req_dir"]
    resp_dir_path: Path = cfg["resp_dir"]
    interval: float = cfg["poll_interval"]

    req_dir_path.mkdir(parents=True, exist_ok=True)
    resp_dir_path.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger.info(
        "Bridge worker started.  Polling %s every %.2fs.",
        req_dir_path,
        interval,
    )

    while True:
        try:
            req_files = sorted(req_dir_path.glob("*.json"))
        except OSError:
            time.sleep(interval)
            continue

        for req_file in req_files:
            try:
                text = req_file.read_text(encoding="utf-8")
                envelope = json.loads(text)
            except (OSError, json.JSONDecodeError):
                # File may still be mid-write; skip and retry next cycle.
                continue

            req_id: str = envelope.get("id", req_file.stem)
            logger.info(
                "Processing request %s  provider=%s  path=%s",
                req_id,
                envelope.get("provider"),
                envelope.get("path"),
            )

            try:
                response = _process_request(envelope)
            except Exception as exc:
                logger.exception("Unhandled error processing request %s", req_id)
                response = {
                    "status_code": 500,
                    "body": {"error": f"worker unhandled exception: {exc}"},
                }

            resp_path = resp_dir_path / f"{req_id}.json"
            try:
                _atomic_write(resp_path, response)
            except OSError as exc:
                logger.error("Failed to write response for %s: %s", req_id, exc)
                continue

            # Remove the request file after successfully writing the response.
            req_file.unlink(missing_ok=True)

        time.sleep(interval)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_worker()
