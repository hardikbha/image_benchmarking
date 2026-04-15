"""
face_detection.py — EResFD-based face detection tool.

Model: EResFD (Efficient Residual Feature-enrichment Face Detector)
  - Architecture : SSD-style single-shot detector
  - Width mult   : 0.0625 (lightweight variant)
  - Task         : Multi-scale face detection with confidence scores

The tool invokes ``eresfd_cli.py`` as a subprocess and returns a JSON
string describing every detected face.

Environment variables
---------------------
DFTOOLBENCH_TOOL_ROOT
    Root directory containing the CLI scripts.
DFTOOLBENCH_ERESFD_WEIGHTS
    Default path to the EResFD weights file.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional

from .base import BaseTool

TOOL_ROOT = os.environ.get("DFTOOLBENCH_TOOL_ROOT", os.path.dirname(__file__))


class FaceDetectionTool(BaseTool):
    """Detect faces in an image using EResFD.

    EResFD is a lightweight SSD-style face detector designed for
    efficiency.  This tool wraps the ``eresfd_cli.py`` inference script
    and returns structured JSON output.

    Parameters
    ----------
    weights_path : str, optional
        Path to the EResFD ``.pth`` checkpoint.  Falls back to
        ``DFTOOLBENCH_ERESFD_WEIGHTS`` env var, then
        ``<TOOL_ROOT>/checkpoints/eresfd/eresfd.pth``.
    device : str
        Torch device string (e.g. ``"cuda:0"`` or ``"cpu"``).
    threshold : float
        Confidence score threshold for filtering detections
        (default: 0.5).
    cli_script : str, optional
        Path to the CLI entry-point.  Defaults to
        ``<TOOL_ROOT>/eresfd_cli.py``.
    timeout : int
        Per-call subprocess timeout in seconds (default: 120).
    """

    default_desc = (
        "Detect faces in an image using EResFD (SSD-style, width_mult=0.0625). "
        "Returns a JSON string with 'boxes' ([[x1,y1,x2,y2], ...]), "
        "'scores' ([float, ...]), and 'count' (int)."
    )

    def __init__(
        self,
        weights_path: Optional[str] = None,
        device: str = "cpu",
        threshold: float = 0.5,
        cli_script: Optional[str] = None,
        timeout: int = 120,
    ) -> None:
        super().__init__(device=device)
        self.weights_path = weights_path or os.environ.get(
            "DFTOOLBENCH_ERESFD_WEIGHTS",
            os.path.join(TOOL_ROOT, "checkpoints", "eresfd", "eresfd.pth"),
        )
        self.threshold = threshold
        self.cli_script = cli_script or os.path.join(TOOL_ROOT, "eresfd_cli.py")
        self.timeout = timeout

    def apply(self, image_path: str) -> str:
        """Detect all faces in *image_path*.

        Parameters
        ----------
        image_path : str
            Path to the input image.

        Returns
        -------
        str
            JSON-encoded string with three keys:

            * ``boxes``  (list[list[int]]) — bounding boxes ``[x1, y1, x2, y2]``.
            * ``scores`` (list[float])     — confidence scores per box.
            * ``count``  (int)             — number of detected faces.

        Raises
        ------
        FileNotFoundError
            If *image_path* does not exist.
        RuntimeError
            If the CLI script exits non-zero.
        """
        image_path = str(Path(image_path).resolve())
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        cmd = [
            sys.executable,
            self.cli_script,
            "--image", image_path,
            "--weights", self.weights_path,
            "--threshold", str(self.threshold),
            "--device", self.device,
        ]
        result = self._run_cli(cmd, timeout=self.timeout)
        # CLI must write JSON to stdout
        return result.stdout.strip()
