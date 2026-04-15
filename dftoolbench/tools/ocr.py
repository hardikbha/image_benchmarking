"""
ocr.py — EasyOCR-based text recognition tool.

Model: EasyOCR (CRAFT text detector + CRNN recogniser)
Task : Optical character recognition with bounding-box output

Each detected text region is reported as a single line:
    ``(x1,y1,x2,y2) <text>``

where ``(x1,y1)`` is the top-left corner and ``(x2,y2)`` the bottom-right
corner of the axis-aligned bounding rectangle.

Environment variables
---------------------
DFTOOLBENCH_TOOL_ROOT
    Root directory used as a fallback for locating model caches.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

from .base import BaseTool

TOOL_ROOT = os.environ.get("DFTOOLBENCH_TOOL_ROOT", os.path.dirname(__file__))


class OCRTool(BaseTool):
    """Extract text and bounding boxes from an image using EasyOCR.

    EasyOCR is loaded lazily on the first call to ``apply()`` so that
    import time stays low even when many tools are imported together.

    Parameters
    ----------
    lang : list[str]
        List of language codes to enable (default: ``["en"]``).
        Any language supported by EasyOCR is valid, e.g. ``["en", "fr"]``.
    device : str
        Torch device string.  ``"cpu"`` disables CUDA; any ``"cuda:*"``
        string enables GPU inference.
    line_group_tolerance : int
        Maximum vertical pixel distance used when grouping detections
        that belong to the same text line (default: 10).
    """

    default_desc = (
        "Perform optical character recognition (OCR) on an image using EasyOCR. "
        "Returns one detected text region per line, formatted as "
        "'(x1,y1,x2,y2) <text>'."
    )

    def __init__(
        self,
        lang: Optional[List[str]] = None,
        device: str = "cpu",
        line_group_tolerance: int = 10,
    ) -> None:
        super().__init__(device=device)
        self.lang = lang or ["en"]
        self.line_group_tolerance = line_group_tolerance
        self._reader = None  # lazy-loaded

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _get_reader(self):
        """Return (and cache) the EasyOCR Reader instance."""
        if self._reader is None:
            try:
                import easyocr  # type: ignore
            except ImportError as exc:
                raise ImportError(
                    "EasyOCR is not installed. "
                    "Install it with: pip install easyocr"
                ) from exc
            use_gpu = self.device.startswith("cuda")
            self._reader = easyocr.Reader(self.lang, gpu=use_gpu)
        return self._reader

    # ------------------------------------------------------------------
    # apply
    # ------------------------------------------------------------------

    def apply(self, image_path: str) -> str:
        """Run OCR on a single image.

        Parameters
        ----------
        image_path : str
            Path to the input image (any format supported by PIL/OpenCV).

        Returns
        -------
        str
            Multi-line string.  Each line follows the format::

                (x1,y1,x2,y2) <recognised text>

            Lines are ordered top-to-bottom, then left-to-right.
            Returns an empty string when no text is detected.

        Raises
        ------
        FileNotFoundError
            If *image_path* does not exist.
        """
        image_path = str(Path(image_path).resolve())
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        reader = self._get_reader()
        detections = reader.readtext(image_path)

        if not detections:
            return ""

        lines: List[str] = []
        for bbox_pts, text, _ in detections:
            # bbox_pts is [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
            xs = [int(p[0]) for p in bbox_pts]
            ys = [int(p[1]) for p in bbox_pts]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
            lines.append(f"({x1},{y1},{x2},{y2}) {text}")

        # Sort top-to-bottom, then left-to-right within the tolerance band.
        lines.sort(key=lambda s: (
            int(s.split(",")[1]),   # y1
            int(s.split(",")[0].lstrip("(")),  # x1
        ))
        return "\n".join(lines)
