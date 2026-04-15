"""
scene_change_detection.py — GeSCF-based scene change detection tool.

Model: GeSCF (General Scene Change Finder)
  - Architecture : Training-free; uses SAM ViT-H for segmentation +
                   DINOv2 for patch-level feature comparison.
  - Task         : Binary change mask between two temporally ordered images.

The tool invokes ``gescf_cli.py`` as a subprocess, which receives two
image paths (t0 and t1) and writes a binary change mask PNG.

Environment variables
---------------------
DFTOOLBENCH_TOOL_ROOT
    Root directory containing the CLI scripts.
DFTOOLBENCH_SAM_CKPT
    Default path to the SAM ViT-H checkpoint.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

from .base import BaseTool

TOOL_ROOT = os.environ.get("DFTOOLBENCH_TOOL_ROOT", os.path.dirname(__file__))


class SceneChangeDetectionTool(BaseTool):
    """Detect changed regions between two images using GeSCF.

    GeSCF is a training-free change detection method that combines SAM
    ViT-H segmentation masks with DINOv2 patch features to identify
    semantically changed regions without task-specific fine-tuning.

    Parameters
    ----------
    device : str
        Torch device string (e.g. ``"cuda:0"`` or ``"cpu"``).
    sam_backbone : str
        SAM backbone variant to use.  Must be ``"vit_h"`` (default),
        ``"vit_l"``, or ``"vit_b"``.
    feature_layer : int
        DINOv2 layer index whose features are used for comparison
        (default: 11, the penultimate layer of ViT-S/14).
    sam_checkpoint : str, optional
        Path to SAM checkpoint ``.pth``.  Falls back to
        ``DFTOOLBENCH_SAM_CKPT`` env var, then
        ``<TOOL_ROOT>/checkpoints/sam/sam_vit_h.pth``.
    cli_script : str, optional
        Path to the GeSCF CLI script.  Defaults to
        ``<TOOL_ROOT>/gescf_cli.py``.
    timeout : int
        Per-call subprocess timeout in seconds (default: 300).
    """

    default_desc = (
        "Detect changed regions between two images (t0, t1) using GeSCF "
        "(training-free; SAM ViT-H + DINOv2 features). "
        "Returns the path to a binary change mask PNG."
    )

    def __init__(
        self,
        device: str = "cpu",
        sam_backbone: str = "vit_h",
        feature_layer: int = 11,
        sam_checkpoint: Optional[str] = None,
        cli_script: Optional[str] = None,
        timeout: int = 300,
    ) -> None:
        super().__init__(device=device)
        if sam_backbone not in {"vit_h", "vit_l", "vit_b"}:
            raise ValueError(
                f"sam_backbone must be 'vit_h', 'vit_l', or 'vit_b', "
                f"got {sam_backbone!r}"
            )
        self.sam_backbone = sam_backbone
        self.feature_layer = feature_layer
        self.sam_checkpoint = sam_checkpoint or os.environ.get(
            "DFTOOLBENCH_SAM_CKPT",
            os.path.join(TOOL_ROOT, "checkpoints", "sam", "sam_vit_h.pth"),
        )
        self.cli_script = cli_script or os.path.join(TOOL_ROOT, "gescf_cli.py")
        self.timeout = timeout

    def apply(self, image_path_t0: str, image_path_t1: str) -> str:
        """Produce a binary change mask between *image_path_t0* and *image_path_t1*.

        Parameters
        ----------
        image_path_t0 : str
            Path to the earlier (reference) image.
        image_path_t1 : str
            Path to the later (query) image.

        Returns
        -------
        str
            Absolute path to the binary change mask PNG.
            White (255) pixels indicate regions that changed between t0 and t1.

        Raises
        ------
        FileNotFoundError
            If either image path does not exist.
        RuntimeError
            If the CLI script exits non-zero.
        """
        for p in (image_path_t0, image_path_t1):
            if not Path(p).exists():
                raise FileNotFoundError(f"Image not found: {p}")

        image_path_t0 = str(Path(image_path_t0).resolve())
        image_path_t1 = str(Path(image_path_t1).resolve())

        out_mask = Path(image_path_t1).parent / (
            Path(image_path_t1).stem + "_change_mask.png"
        )

        cmd = [
            sys.executable,
            self.cli_script,
            "--image_t0", image_path_t0,
            "--image_t1", image_path_t1,
            "--output", str(out_mask),
            "--sam_backbone", self.sam_backbone,
            "--sam_checkpoint", self.sam_checkpoint,
            "--feature_layer", str(self.feature_layer),
            "--device", self.device,
        ]
        self._run_cli(cmd, timeout=self.timeout)
        return str(out_mask)
