"""
dftoolbench.tools — Public tool registry for DFToolBench-I.

All 13 tool classes (12 forensic/vision tools + 1 utility) are exported
from this package so that agent frameworks can discover them with a simple
``from dftoolbench.tools import <ToolClass>`` import.

Tool index
----------
=================================  =============================================
Class                              Capability (brief)
=================================  =============================================
AnomalyDetectionTool               Pixel-level anomaly localisation (PatchGuard)
OCRTool                            Text recognition with bounding boxes (EasyOCR)
TextForgeryLocalizerTool           OCR-based text forgery localisation
CopyMoveLocalizationTool           Copy-move forgery mask (CIML / MMSeg)
FaceDetectionTool                  Multi-scale face detection (EResFD)
SceneChangeDetectionTool           Binary change mask from two images (GeSCF)
FingerprintingTool                 GAN / diffusion / real provenance scoring
Calculator                         Safe arithmetic expression evaluator
ObjectDetectionTool                Object detection with annotated output (YOLOE)
DeepfakeDetectionTool              Real vs. deepfake classification (D3 / CLIP)
SegmentationTool                   Manipulation segmentation mask (ASPC-Net)
DenoiseTool                        Gaussian denoising (MaIR CDN_s25)
=================================  =============================================

``ALL_TOOLS`` is a list of all tool classes, convenient for dynamic
registration loops in agent scaffolding code.
"""

from .anomaly_detection import AnomalyDetectionTool
from .calculator import Calculator
from .copy_move_localization import CopyMoveLocalizationTool
from .deepfake_detection import DeepfakeDetectionTool
from .denoise import DenoiseTool
from .face_detection import FaceDetectionTool
from .fingerprinting import FingerprintingTool
from .object_detection import ObjectDetectionTool
from .ocr import OCRTool
from .scene_change_detection import SceneChangeDetectionTool
from .segmentation import SegmentationTool
from .text_forgery_localizer import TextForgeryLocalizerTool

__all__ = [
    "AnomalyDetectionTool",
    "Calculator",
    "CopyMoveLocalizationTool",
    "DeepfakeDetectionTool",
    "DenoiseTool",
    "FaceDetectionTool",
    "FingerprintingTool",
    "ObjectDetectionTool",
    "OCRTool",
    "SceneChangeDetectionTool",
    "SegmentationTool",
    "TextForgeryLocalizerTool",
]

#: Flat list of all tool classes, useful for dynamic registration.
ALL_TOOLS = [
    AnomalyDetectionTool,
    Calculator,
    CopyMoveLocalizationTool,
    DeepfakeDetectionTool,
    DenoiseTool,
    FaceDetectionTool,
    FingerprintingTool,
    ObjectDetectionTool,
    OCRTool,
    SceneChangeDetectionTool,
    SegmentationTool,
    TextForgeryLocalizerTool,
]
