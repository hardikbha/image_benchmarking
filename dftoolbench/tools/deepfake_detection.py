"""
deepfake_detection.py — D3 deepfake detection tool.

Model: D3 (Detecting Deepfakes with Disentangled Representations)
  - Backbone : CLIP ViT-L/14 (frozen visual encoder)
  - Head     : ShuffleAttention classification head
  - Task     : Binary deepfake detection (real vs. fake)

Unlike the other CLI-based tools, D3 loads the model directly inside the
process for lower per-call overhead.  The model is lazy-loaded on the
first call to ``apply()``.

Environment variables
---------------------
DFTOOLBENCH_D3_CKPT
    Default path to the D3 checkpoint file.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from .base import BaseTool

_DEFAULT_CKPT = os.environ.get(
    "DFTOOLBENCH_D3_CKPT",
    os.path.join(
        os.environ.get("DFTOOLBENCH_TOOL_ROOT", os.path.dirname(__file__)),
        "checkpoints", "d3", "d3_clip_vitl14.pth",
    ),
)


class DeepfakeDetectionTool(BaseTool):
    """Detect whether an image is a deepfake using the D3 model.

    D3 pairs a frozen CLIP ViT-L/14 encoder with a lightweight
    ShuffleAttention head to achieve strong generalisation across GAN and
    diffusion-based deepfake generators.

    The model is loaded lazily on the first call to ``apply()`` to avoid
    long import times when only other tools are being used.

    Parameters
    ----------
    checkpoint_path : str, optional
        Path to the D3 ``.pth`` checkpoint.  Falls back to the
        ``DFTOOLBENCH_D3_CKPT`` environment variable, then
        ``<TOOL_ROOT>/checkpoints/d3/d3_clip_vitl14.pth``.
    device : str
        Torch device string (e.g. ``"cuda:0"`` or ``"cpu"``).
    """

    default_desc = (
        "Determine whether an image is a deepfake using the D3 model "
        "(CLIP ViT-L/14 backbone + ShuffleAttention head). "
        "Returns a verdict string such as 'FAKE (confidence: 0.97)' or "
        "'REAL (confidence: 0.84)'."
    )

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: str = "cpu",
    ) -> None:
        super().__init__(device=device)
        self.checkpoint_path = checkpoint_path or _DEFAULT_CKPT
        self._model = None       # lazy-loaded
        self._transform = None   # lazy-loaded

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Load the D3 model and CLIP preprocessing transform."""
        try:
            import torch  # type: ignore
            import clip    # type: ignore  (openai/clip)
        except ImportError as exc:
            raise ImportError(
                "torch and clip are required for DeepfakeDetectionTool. "
                "Install with: pip install torch openai-clip"
            ) from exc

        try:
            from d3.model import D3Classifier  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "The D3 package is not installed or not on sys.path. "
                "Please add it to your environment."
            ) from exc

        _, self._transform = clip.load("ViT-L/14", device="cpu")
        self._model = D3Classifier()
        state = torch.load(self.checkpoint_path, map_location="cpu")
        self._model.load_state_dict(state)
        self._model.to(self.device)
        self._model.eval()

    # ------------------------------------------------------------------
    # apply
    # ------------------------------------------------------------------

    def apply(self, image_path: str) -> str:
        """Classify *image_path* as real or deepfake.

        Parameters
        ----------
        image_path : str
            Path to the input image.

        Returns
        -------
        str
            Human-readable verdict, e.g.::

                FAKE (confidence: 0.97)
                REAL (confidence: 0.84)

        Raises
        ------
        FileNotFoundError
            If *image_path* does not exist.
        """
        image_path = str(Path(image_path).resolve())
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        if self._model is None:
            self._load_model()

        try:
            import torch  # type: ignore
            from PIL import Image  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "torch and Pillow are required. "
                "Install with: pip install torch Pillow"
            ) from exc

        image = Image.open(image_path).convert("RGB")
        tensor = self._transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self._model(tensor)
            probs = torch.softmax(logits, dim=-1)
            fake_prob = float(probs[0, 1])

        label = "FAKE" if fake_prob >= 0.5 else "REAL"
        confidence = fake_prob if label == "FAKE" else 1.0 - fake_prob
        return f"{label} (confidence: {confidence:.4f})"
