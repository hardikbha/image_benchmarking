"""
segmentation.py — MIML/ASPC-Net image manipulation segmentation tool.

Model: MIML / ASPC-Net
  - Framework : MMSegmentation
  - Task      : Pixel-level image manipulation localisation via semantic
                segmentation

The tool spawns ``aspcnet_cli.py`` as a subprocess and writes a binary
mask PNG alongside the input image.

Environment variables
---------------------
DFTOOLBENCH_TOOL_ROOT
    Root directory containing the CLI scripts.
DFTOOLBENCH_ASPCNET_WEIGHTS
    Default path to the ASPC-Net model weights file.
DFTOOLBENCH_ASPCNET_CONFIG
    Default path to the MMSeg config file for ASPC-Net.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

from .base import BaseTool

TOOL_ROOT = os.environ.get("DFTOOLBENCH_TOOL_ROOT", os.path.dirname(__file__))


class SegmentationTool(BaseTool):
    """Localise manipulated regions via MIML/ASPC-Net segmentation.

    ASPC-Net uses an atrous spatial pyramid convolution architecture
    (implemented with MMSeg) to separate authentic from manipulated image
    regions at the pixel level.

    Parameters
    ----------
    weights_path : str, optional
        Path to the ASPC-Net ``.pth`` checkpoint.  Falls back to
        ``DFTOOLBENCH_ASPCNET_WEIGHTS`` env var, then
        ``<TOOL_ROOT>/checkpoints/aspcnet/aspcnet.pth``.
    config_path : str, optional
        Path to the MMSeg config ``.py`` file.  Falls back to
        ``DFTOOLBENCH_ASPCNET_CONFIG`` env var, then
        ``<TOOL_ROOT>/configs/aspcnet/aspcnet.py``.
    device : str
        Torch device string (e.g. ``"cuda:0"`` or ``"cpu"``).
    cli_script : str, optional
        Path to the CLI entry-point script.  Defaults to
        ``<TOOL_ROOT>/aspcnet_cli.py``.
    timeout : int
        Per-call subprocess timeout in seconds (default: 300).
    """

    default_desc = (
        "Localise manipulated regions in an image using MIML/ASPC-Net "
        "(MMSeg-based segmentation). Returns the path to a binary mask PNG "
        "where white (255) pixels indicate manipulated areas."
    )

    def __init__(
        self,
        weights_path: Optional[str] = None,
        config_path: Optional[str] = None,
        device: str = "cpu",
        cli_script: Optional[str] = None,
        timeout: int = 300,
    ) -> None:
        super().__init__(device=device)
        self.weights_path = weights_path or os.environ.get(
            "DFTOOLBENCH_ASPCNET_WEIGHTS",
            os.path.join(TOOL_ROOT, "checkpoints", "aspcnet", "aspcnet.pth"),
        )
        self.config_path = config_path or os.environ.get(
            "DFTOOLBENCH_ASPCNET_CONFIG",
            os.path.join(TOOL_ROOT, "configs", "aspcnet", "aspcnet.py"),
        )
        self.cli_script = cli_script or os.path.join(TOOL_ROOT, "aspcnet_cli.py")
        self.timeout = timeout

    def apply(self, image_path: str) -> str:
        """Produce a manipulation segmentation mask for *image_path*.

        Parameters
        ----------
        image_path : str
            Path to the input image.

        Returns
        -------
        str
            Absolute path to the binary mask PNG.
            Pixel value 255 marks manipulated regions;
            0 marks authentic regions.

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

        out_mask = Path(image_path).parent / (
            Path(image_path).stem + "_seg_mask.png"
        )

        cmd = [
            sys.executable,
            self.cli_script,
            "--image", image_path,
            "--weights", self.weights_path,
            "--config", self.config_path,
            "--output", str(out_mask),
            "--device", self.device,
        ]
        self._run_cli(cmd, timeout=self.timeout)
        return str(out_mask)
