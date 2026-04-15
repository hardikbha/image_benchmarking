"""
fingerprinting.py — GAN fingerprint detection tool.

Model: GAN Fingerprint Autoencoder
  - Architecture : Convolutional autoencoder trained to detect and
                   separate GAN / diffusion / camera fingerprint signals.
  - Task         : Classify an image as GAN-generated, diffusion-generated,
                   or real (camera).

The tool spawns ``fingerprint_cli.py`` as a subprocess and returns a JSON
string with per-class confidence scores.

Environment variables
---------------------
DFTOOLBENCH_TOOL_ROOT
    Root directory containing the CLI scripts.
DFTOOLBENCH_FINGERPRINT_CKPT
    Default path to the fingerprinting autoencoder checkpoint directory.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional

from .base import BaseTool

TOOL_ROOT = os.environ.get("DFTOOLBENCH_TOOL_ROOT", os.path.dirname(__file__))


class FingerprintingTool(BaseTool):
    """Classify image provenance as GAN, diffusion, or real via fingerprinting.

    The autoencoder decomposes the image into a fingerprint residual and a
    clean reconstruction.  The fingerprint residual is classified by a
    shallow MLP into three provenance categories.

    Parameters
    ----------
    device : str
        Torch device string (e.g. ``"cuda:0"`` or ``"cpu"``).
    crop_size : int
        Central crop side-length (pixels) applied before inference
        (default: 256).  Must match the training resolution.
    checkpoint_dir : str, optional
        Directory containing the autoencoder and classifier checkpoints.
        Falls back to ``DFTOOLBENCH_FINGERPRINT_CKPT`` env var, then
        ``<TOOL_ROOT>/checkpoints/fingerprinting``.
    cli_script : str, optional
        Path to the CLI entry-point script.  Defaults to
        ``<TOOL_ROOT>/fingerprint_cli.py``.
    timeout : int
        Per-call subprocess timeout in seconds (default: 120).
    """

    default_desc = (
        "Detect GAN / diffusion fingerprints in an image using a convolutional "
        "autoencoder. Returns a JSON string with confidence scores for three "
        "provenance classes: 'gan', 'diffusion', and 'real'."
    )

    def __init__(
        self,
        device: str = "cpu",
        crop_size: int = 256,
        checkpoint_dir: Optional[str] = None,
        cli_script: Optional[str] = None,
        timeout: int = 120,
    ) -> None:
        super().__init__(device=device)
        self.crop_size = crop_size
        self.checkpoint_dir = checkpoint_dir or os.environ.get(
            "DFTOOLBENCH_FINGERPRINT_CKPT",
            os.path.join(TOOL_ROOT, "checkpoints", "fingerprinting"),
        )
        self.cli_script = cli_script or os.path.join(
            TOOL_ROOT, "fingerprint_cli.py"
        )
        self.timeout = timeout

    def apply(self, image_path: str) -> str:
        """Classify the provenance of *image_path*.

        Parameters
        ----------
        image_path : str
            Path to the input image.

        Returns
        -------
        str
            JSON-encoded string with three keys:

            * ``gan``       (float) — confidence that the image is GAN-generated.
            * ``diffusion`` (float) — confidence that the image is
              diffusion-generated.
            * ``real``      (float) — confidence that the image is a real
              (camera) photo.

            Values are probabilities that sum to approximately 1.0.

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
            "--checkpoint_dir", self.checkpoint_dir,
            "--crop_size", str(self.crop_size),
            "--device", self.device,
        ]
        result = self._run_cli(cmd, timeout=self.timeout)
        return result.stdout.strip()
