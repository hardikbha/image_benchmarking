"""
anomaly_detection.py — PatchGuard-based anomaly detection tool.

Model: PatchGuard
  - Backbone : DINOv2 ViT-S/14 (frozen)
  - Head     : Discriminator transformer
  - Task     : Pixel-level anomaly localisation

The tool invokes ``patchguard_cli.py`` (expected on sys.path or alongside
this file) via subprocess, then parses its JSON output.

Environment variables
---------------------
DFTOOLBENCH_TOOL_ROOT
    Root directory that contains the CLI scripts.  Defaults to the
    directory that holds this file.
DFTOOLBENCH_PATCHGUARD_CKPT
    Default checkpoint directory (overridable via constructor).
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


class AnomalyDetectionTool(BaseTool):
    """Localise anomalous regions in a single image using PatchGuard.

    PatchGuard pairs a frozen DINOv2 ViT-S/14 feature extractor with a
    lightweight Discriminator transformer to produce a binary anomaly mask
    and a scalar anomaly percentage.

    Parameters
    ----------
    checkpoint_dir : str, optional
        Path to the directory containing PatchGuard checkpoint files.
        Falls back to the ``DFTOOLBENCH_PATCHGUARD_CKPT`` environment
        variable, then to ``<TOOL_ROOT>/checkpoints/patchguard``.
    device : str
        Torch device string (e.g. ``"cuda:0"`` or ``"cpu"``).
    image_size : int
        Side length (pixels) used when resizing the input image.
        Must match the value used during training (default: 224).
    cli_script : str, optional
        Absolute path to ``patchguard_cli.py``.  Defaults to
        ``<TOOL_ROOT>/patchguard_cli.py``.
    timeout : int
        Per-call subprocess timeout in seconds (default: 300).
    """

    default_desc = (
        "Detect and localise anomalous regions in an image using PatchGuard "
        "(DINOv2 ViT-S/14 backbone + Discriminator transformer). "
        "Returns a JSON string containing 'mask_path' (path to the binary "
        "anomaly mask PNG) and 'anomaly_pct' (fraction of anomalous pixels, 0–100)."
    )

    def __init__(
        self,
        checkpoint_dir: Optional[str] = None,
        device: str = "cpu",
        image_size: int = 224,
        cli_script: Optional[str] = None,
        timeout: int = 300,
    ) -> None:
        super().__init__(device=device)
        self.checkpoint_dir = checkpoint_dir or os.environ.get(
            "DFTOOLBENCH_PATCHGUARD_CKPT",
            os.path.join(TOOL_ROOT, "checkpoints", "patchguard"),
        )
        self.image_size = image_size
        self.cli_script = cli_script or os.path.join(TOOL_ROOT, "patchguard_cli.py")
        self.timeout = timeout

    def apply(self, image_path: str) -> str:
        """Run PatchGuard on a single image.

        Parameters
        ----------
        image_path : str
            Absolute or relative path to the input image.

        Returns
        -------
        str
            JSON-encoded string with two keys:

            * ``mask_path`` (str)  — path to the saved binary mask PNG.
            * ``anomaly_pct`` (float) — percentage of pixels flagged as
              anomalous (range 0–100).

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

        with tempfile.TemporaryDirectory() as tmp_dir:
            cmd = [
                sys.executable,
                self.cli_script,
                "--image", image_path,
                "--checkpoint_dir", self.checkpoint_dir,
                "--output_dir", tmp_dir,
                "--device", self.device,
                "--image_size", str(self.image_size),
            ]
            result = self._run_cli(cmd, timeout=self.timeout)
            payload = json.loads(result.stdout.strip())

            # Move mask out of the temp dir to a stable sibling location.
            src_mask = Path(payload["mask_path"])
            stable_mask = Path(image_path).with_suffix("") .parent / (
                src_mask.stem + "_anomaly_mask.png"
            )
            src_mask.rename(stable_mask)
            payload["mask_path"] = str(stable_mask)

        return json.dumps(payload)
