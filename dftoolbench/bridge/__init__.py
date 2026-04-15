"""
dftoolbench.bridge
==================

Filesystem-based HTTP forwarding bridge between the agent runner and
external LLM APIs.

Sub-modules
-----------
bridge_server
    Lightweight HTTP proxy that accepts requests from the agent, writes them
    as JSON files to a shared directory, then polls for responses written by
    the worker.
bridge_worker
    Worker process that polls the request directory, makes the actual API
    calls (OpenAI or Anthropic), and writes responses back to disk.

Typical deployment
------------------
Start the server::

    python -m dftoolbench.bridge.bridge_server

Start the worker in a separate process::

    python -m dftoolbench.bridge.bridge_worker

Both components communicate exclusively through the shared filesystem
directories configured via environment variables (``BRIDGE_REQ_DIR`` /
``BRIDGE_RESP_DIR``).
"""

__all__ = []
