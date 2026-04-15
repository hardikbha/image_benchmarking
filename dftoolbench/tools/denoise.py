"""
denoise.py — MaIR-based image denoising tool.

Model: MaIR (Mamba-in-Image Restoration)
  - Preset : CDN_s25 (colour denoising, sigma=25)
  - Task   : Blind Gaussian denoising / image restoration

The tool spawns ``mair_cli.py`` as a subprocess, which loads the MaIR
model and writes the denoised image to disk.

Environment variables
---------------------
DFTOOLBENCH_TOOL_ROOT
    Root directory containing the CLI scripts.
DFTOOLBENCH_MAIR_WEIGHTS
    Default path to the MaIR CDN_s25 weights file.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

from .base import BaseTool

TOOL_ROOT = os.environ.get("DFTOOLBENCH_TOOL_ROOT", os.path.dirname(__file__))


class DenoiseTool(BaseTool):
    """Denoise an image using the MaIR (Mamba-in-Image Restoration) model.

    MaIR integrates selective state-space (Mamba) blocks into a U-Net-style
    restoration network.  The CDN_s25 preset is trained for colour Gaussian
    denoising at sigma=25 but generalises well to real-world noise.

    Parameters
    ----------
    weights_path : str, optional
        Path to the MaIR CDN_s25 ``.pth`` checkpoint.  Falls back to
        ``DFTOOLBENCH_MAIR_WEIGHTS`` env var, then
        ``<TOOL_ROOT>/checkpoints/mair/cdn_s25.pth``.
    device : str
        Torch device string (e.g. ``"cuda:0"`` or ``"cpu"``).
    cli_script : str, optional
        Path to the MaIR CLI entry-point script.  Defaults to
        ``<TOOL_ROOT>/mair_cli.py``.
    timeout : int
        Per-call subprocess timeout in seconds (default: 300).
    """

    default_desc = (
        "Denoise an image using the MaIR model (Mamba-in-Image Restoration, "
        "CDN_s25 preset). Returns the path to the denoised output image."
    )

    def __init__(
        self,
        weights_path: Optional[str] = None,
        device: str = "cpu",
        cli_script: Optional[str] = None,
        timeout: int = 300,
    ) -> None:
        super().__init__(device=device)
        self.weights_path = weights_path or os.environ.get(
            "DFTOOLBENCH_MAIR_WEIGHTS",
            os.path.join(TOOL_ROOT, "checkpoints", "mair", "cdn_s25.pth"),
        )
        self.cli_script = cli_script or os.path.join(TOOL_ROOT, "mair_cli.py")
        self.timeout = timeout

    def apply(self, image_path: str) -> str:
        """Denoise *image_path* and return the path to the restored image.

        Parameters
        ----------
        image_path : str
            Path to the noisy input image.

        Returns
        -------
        str
            Absolute path to the denoised output image.  The file is saved
            in the same directory as the input, with ``_denoised`` appended
            to the stem (e.g. ``photo_denoised.png``).

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
        out_image = src.parent / (src.stem + "_denoised" + src.suffix)

        cmd = [
            sys.executable,
            self.cli_script,
            "--image", image_path,
            "--weights", self.weights_path,
            "--output", str(out_image),
            "--device", self.device,
        ]
        self._run_cli(cmd, timeout=self.timeout)
        return str(out_image)
