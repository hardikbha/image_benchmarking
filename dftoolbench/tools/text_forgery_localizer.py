"""
text_forgery_localizer.py — OCR-based text forgery localisation tool.

Inspired by:
    "Towards Universal Fake Image Detection Exploiting Style Consistency"
    CVPR 2023

Pipeline
--------
1. Run an OCR backend (auto / paddle / easyocr / tesseract) via a CLI
   wrapper to extract per-word bounding boxes.
2. Compare bounding-box statistics (aspect ratio, inter-character spacing,
   font consistency) to flag suspicious word regions.
3. Rasterise the flagged boxes into a binary mask PNG
   (255 = possibly forged text, 0 = background).

Environment variables
---------------------
DFTOOLBENCH_TOOL_ROOT
    Root directory that contains the OCR CLI scripts.
DFTOOLBENCH_TEXTFORGERY_DEVICE
    Default device override (``cuda:0`` / ``cpu``).
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional

from .base import BaseTool

TOOL_ROOT = os.environ.get("DFTOOLBENCH_TOOL_ROOT", os.path.dirname(__file__))

# Supported detector backends
_VALID_BACKENDS = {"auto", "paddle", "easyocr", "tesseract"}


class TextForgeryLocalizerTool(BaseTool):
    """Localise text forgery regions in a document or scene-text image.

    The tool runs an OCR detector on the input image to extract word-level
    bounding boxes, applies a forgery-consistency scoring function on each
    box, and rasterises the suspicious boxes into a binary mask.

    Parameters
    ----------
    device : str
        Torch device string (used by neural OCR backends).
    detector_backend : str
        OCR engine to use.  One of ``"auto"`` (picks the best available
        backend), ``"paddle"``, ``"easyocr"``, or ``"tesseract"``.
    cli_script : str, optional
        Path to the OCR-forgery CLI script.  Defaults to
        ``<TOOL_ROOT>/text_forgery_cli.py``.
    timeout : int
        Per-call subprocess timeout in seconds (default: 300).
    """

    default_desc = (
        "Localise forged or tampered text regions in an image using OCR-based "
        "analysis (CVPR 2023 inspired). Returns the path to a binary mask PNG "
        "where white (255) pixels mark suspected text forgery regions."
    )

    def __init__(
        self,
        device: str = "cpu",
        detector_backend: str = "auto",
        cli_script: Optional[str] = None,
        timeout: int = 300,
    ) -> None:
        super().__init__(device=device)
        if detector_backend not in _VALID_BACKENDS:
            raise ValueError(
                f"detector_backend must be one of {sorted(_VALID_BACKENDS)}, "
                f"got {detector_backend!r}"
            )
        self.detector_backend = detector_backend
        self.cli_script = cli_script or os.path.join(
            TOOL_ROOT, "text_forgery_cli.py"
        )
        self.timeout = timeout

    def apply(self, image_path: str) -> str:
        """Produce a binary text-forgery mask for *image_path*.

        Parameters
        ----------
        image_path : str
            Path to the input image.

        Returns
        -------
        str
            Absolute path to the saved binary mask PNG.
            Pixel value 255 marks regions identified as potentially forged;
            0 marks background / unmodified text.

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

        out_mask = Path(image_path).with_suffix("").parent / (
            Path(image_path).stem + "_textforgery_mask.png"
        )

        cmd = [
            sys.executable,
            self.cli_script,
            "--image", image_path,
            "--output", str(out_mask),
            "--backend", self.detector_backend,
            "--device", self.device,
        ]
        self._run_cli(cmd, timeout=self.timeout)
        return str(out_mask)
