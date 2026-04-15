"""
base.py — Lightweight base class for all DFToolBench tools.

Every tool in this package inherits from ``BaseTool`` and implements
``apply()``.  CLI-based tools can use the ``_run_cli()`` helper, which
wraps ``subprocess.run`` with a consistent timeout/error contract.
"""

from __future__ import annotations

import subprocess
import sys
from abc import ABC, abstractmethod
from typing import Any


class BaseTool(ABC):
    """Abstract base class for DFToolBench tools.

    Sub-classes must set ``default_desc`` (a one-line human-readable
    description used by the agent framework) and implement ``apply()``.

    Parameters
    ----------
    device : str
        Compute device string, e.g. ``"cuda:0"`` or ``"cpu"``.
    """

    #: One-line description surfaced to the agent as the tool's capability.
    default_desc: str = ""

    def __init__(self, device: str = "cpu") -> None:
        self.device = device

    @abstractmethod
    def apply(self, *args: Any, **kwargs: Any) -> Any:
        """Run the tool and return its output."""

    # ------------------------------------------------------------------
    # Subprocess helper
    # ------------------------------------------------------------------

    def _run_cli(
        self,
        cmd: list[str],
        timeout: int = 300,
        check: bool = True,
    ) -> subprocess.CompletedProcess:
        """Run a CLI command as a subprocess.

        Parameters
        ----------
        cmd : list[str]
            Command + arguments, e.g. ``[sys.executable, "script.py", "--arg"]``.
        timeout : int
            Wall-clock timeout in seconds (default 300 s / 5 min).
        check : bool
            If *True* (default), raise ``RuntimeError`` when the process
            exits with a non-zero return code.

        Returns
        -------
        subprocess.CompletedProcess
            Completed process object; ``stdout`` and ``stderr`` are
            captured as UTF-8 strings.

        Raises
        ------
        RuntimeError
            When *check* is True and the process exits non-zero.
        subprocess.TimeoutExpired
            When the process exceeds *timeout* seconds.
        """
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )
        if check and result.returncode != 0:
            raise RuntimeError(
                f"CLI command failed (exit {result.returncode}):\n"
                f"  cmd : {' '.join(cmd)}\n"
                f"  stderr: {result.stderr.strip()}"
            )
        return result

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Alias so tools can be called directly like functions."""
        return self.apply(*args, **kwargs)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(device={self.device!r})"
