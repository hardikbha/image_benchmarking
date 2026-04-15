"""
object_detection.py — YOLOE/YOLOv8-based object detection tool.

Model: YOLOE / YOLOv8
  - Framework : Ultralytics
  - Task      : Open-vocabulary / closed-set object detection

The tool invokes ``yolo_cli.py`` as a subprocess.  Detected objects are
drawn onto an annotated copy of the input image, and the path to that
annotated image is returned.

Environment variables
---------------------
DFTOOLBENCH_TOOL_ROOT
    Root directory containing the CLI scripts.
DFTOOLBENCH_YOLO_WEIGHTS
    Default path to the YOLO weights file (``*.pt``).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

from .base import BaseTool

TOOL_ROOT = os.environ.get("DFTOOLBENCH_TOOL_ROOT", os.path.dirname(__file__))


class ObjectDetectionTool(BaseTool):
    """Detect objects in an image using YOLOE / YOLOv8.

    The tool draws bounding-box annotations on a copy of the input image
    and returns the path to that annotated file.  The original image is
    left unmodified.

    Parameters
    ----------
    weights_path : str, optional
        Path to the Ultralytics ``.pt`` model weights.  Falls back to
        ``DFTOOLBENCH_YOLO_WEIGHTS`` env var, then
        ``<TOOL_ROOT>/checkpoints/yolo/yoloe.pt``.
    device : str
        Torch device string (e.g. ``"cuda:0"`` or ``"cpu"``).
    imgsz : int
        Inference image size (pixels); the image is letterboxed to this
        size (default: 640).
    conf : float
        Minimum confidence threshold for a detection to be kept
        (default: 0.25).
    iou : float
        IoU threshold used for NMS (default: 0.45).
    cli_script : str, optional
        Path to the CLI entry-point script.  Defaults to
        ``<TOOL_ROOT>/yolo_cli.py``.
    timeout : int
        Per-call subprocess timeout in seconds (default: 120).
    """

    default_desc = (
        "Detect objects in an image using YOLOE/YOLOv8. "
        "Returns the path to an annotated copy of the input image with "
        "bounding boxes and class labels drawn on it."
    )

    def __init__(
        self,
        weights_path: Optional[str] = None,
        device: str = "cpu",
        imgsz: int = 640,
        conf: float = 0.25,
        iou: float = 0.45,
        cli_script: Optional[str] = None,
        timeout: int = 120,
    ) -> None:
        super().__init__(device=device)
        self.weights_path = weights_path or os.environ.get(
            "DFTOOLBENCH_YOLO_WEIGHTS",
            os.path.join(TOOL_ROOT, "checkpoints", "yolo", "yoloe.pt"),
        )
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.cli_script = cli_script or os.path.join(TOOL_ROOT, "yolo_cli.py")
        self.timeout = timeout

    def apply(self, image_path: str) -> str:
        """Run object detection on *image_path*.

        Parameters
        ----------
        image_path : str
            Path to the input image.

        Returns
        -------
        str
            Absolute path to the annotated output image.  The file is
            saved in the same directory as the input, with ``_detected``
            appended to the stem (e.g. ``photo_detected.jpg``).

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

        src = Path(image_path)
        out_image = src.parent / (src.stem + "_detected" + src.suffix)

        cmd = [
            sys.executable,
            self.cli_script,
            "--image", image_path,
            "--weights", self.weights_path,
            "--output", str(out_image),
            "--device", self.device,
            "--imgsz", str(self.imgsz),
            "--conf", str(self.conf),
            "--iou", str(self.iou),
        ]
        self._run_cli(cmd, timeout=self.timeout)
        return str(out_image)
