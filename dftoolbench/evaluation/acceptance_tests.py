"""
acceptance_tests.py — Gold-answer acceptance predicates for DFToolBench-I.

Based on the acceptance-testing methodology described in Section IV.D of
the paper.

Each predicate class inherits from :class:`BaseAcceptance` and implements
a single ``evaluate(prediction, gold, **kwargs) -> AcceptanceResult`` method.

Available predicates
--------------------
NumericalAcceptance
    Absolute and percentage tolerance checks for scalar numerical values.
CircularityAcceptance
    Geometric shape metrics (circularity, compactness, solidity) within ±r.
CentroidAcceptance
    Pixel-distance and relative-unit centroid distance checks.
CoverageAcceptance
    Mask coverage percentage and IoU threshold checks.
OCRAcceptance
    OCR-derived equality with text normalisation.
WhitelistAcceptance
    All required terms must appear in the prediction.
BlacklistAcceptance
    Any forbidden term causes immediate failure.
AreaAcceptance
    Area-as-percentage-of-image-pixels computation.

Convenience function
--------------------
run_all_predicates(prediction, gold, config) -> list[AcceptanceResult]
    Apply a full configured suite of acceptance predicates.

Environment variables
---------------------
No runtime API keys are required by this module.
``DFTOOLBENCH_IOU_THRESHOLD`` (float, default 0.5)
    Default IoU threshold for :class:`CoverageAcceptance`.
"""

from __future__ import annotations

import logging
import math
import re
import unicodedata
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional numeric / image imports
# ---------------------------------------------------------------------------

try:
    import numpy as np  # type: ignore
    _NUMPY_AVAILABLE = True
except ImportError:
    np = None  # type: ignore
    _NUMPY_AVAILABLE = False

# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class AcceptanceResult:
    """Result of a single acceptance predicate evaluation.

    Parameters
    ----------
    predicate : str
        Name of the predicate that produced this result.
    passed : bool
        Whether the acceptance criterion was met.
    score : float
        Continuous score in [0, 1] representing how well the criterion is met.
    details : dict
        Extra information about the evaluation (predicate-specific).
    """

    predicate: str
    passed: bool
    score: float
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dictionary."""
        return {
            "predicate": self.predicate,
            "passed": self.passed,
            "score": round(self.score, 4),
            "details": self.details,
        }


@dataclass
class AcceptanceConfig:
    """Configuration for the full acceptance test suite.

    Parameters
    ----------
    numerical_tolerance_abs : float
        Absolute tolerance for numerical checks.
    numerical_tolerance_pct : float
        Percentage tolerance (0–100) for numerical checks.
    centroid_max_pixel_dist : float
        Maximum allowed centroid distance in pixels.
    centroid_max_relative_dist : float
        Maximum allowed centroid distance as a fraction of image diagonal.
    iou_threshold : float
        Minimum IoU for mask-overlap acceptance.
    coverage_tolerance : float
        Absolute tolerance on mask coverage percentage (0–100).
    geometry_tolerance : float
        Absolute tolerance for circularity / compactness / solidity checks.
    whitelist : list[str]
        Terms that must appear in the prediction.
    blacklist : list[str]
        Terms whose presence immediately fails the prediction.
    ocr_case_sensitive : bool
        Whether OCR equality check is case-sensitive (default False).
    """

    numerical_tolerance_abs: float = 0.0
    numerical_tolerance_pct: float = 10.0
    centroid_max_pixel_dist: float = 5.0
    centroid_max_relative_dist: float = 0.02
    iou_threshold: float = field(
        default_factory=lambda: float(
            __import__("os").environ.get("DFTOOLBENCH_IOU_THRESHOLD", "0.5")
        )
    )
    coverage_tolerance: float = 5.0
    geometry_tolerance: float = 0.05
    whitelist: list[str] = field(default_factory=list)
    blacklist: list[str] = field(default_factory=list)
    ocr_case_sensitive: bool = False


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class BaseAcceptance(ABC):
    """Abstract base for acceptance predicates."""

    #: Short identifier used in :class:`AcceptanceResult`.
    name: str = "base"

    @abstractmethod
    def evaluate(
        self,
        prediction: Any,
        gold: Any,
        **kwargs: Any,
    ) -> AcceptanceResult:
        """Evaluate the prediction against the gold standard.

        Parameters
        ----------
        prediction : Any
            The model's output.
        gold : Any
            The ground-truth reference.
        **kwargs : Any
            Predicate-specific extra arguments.

        Returns
        -------
        AcceptanceResult
        """

    def __call__(self, prediction: Any, gold: Any, **kwargs: Any) -> AcceptanceResult:
        """Alias so predicates can be used as callables."""
        return self.evaluate(prediction, gold, **kwargs)


# ---------------------------------------------------------------------------
# Predicate implementations
# ---------------------------------------------------------------------------


class NumericalAcceptance(BaseAcceptance):
    """Numerical tolerance acceptance predicate.

    Checks both absolute and percentage tolerances and returns the strictest
    passing bracket.

    Parameters
    ----------
    tolerance_abs : float
        Absolute tolerance.  Pass if ``|pred - gold| <= tolerance_abs``.
    tolerance_pct : float
        Percentage tolerance (0–100).  Pass if relative deviation is within
        this percentage of the gold value.
    """

    name = "numerical"

    def __init__(
        self,
        tolerance_abs: float = 0.0,
        tolerance_pct: float = 10.0,
    ) -> None:
        self.tolerance_abs = tolerance_abs
        self.tolerance_pct = tolerance_pct

    @staticmethod
    def _to_float(value: Any) -> Optional[float]:
        """Extract a float from a value (handles numeric strings)."""
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            # Strip common non-numeric characters (%, $, units)
            cleaned = re.sub(r"[^\d.\-+eE]", "", value.strip())
            try:
                return float(cleaned)
            except ValueError:
                return None
        return None

    def evaluate(
        self,
        prediction: Any,
        gold: Any,
        tolerance_abs: Optional[float] = None,
        tolerance_pct: Optional[float] = None,
        **kwargs: Any,
    ) -> AcceptanceResult:
        """Evaluate numerical closeness.

        Parameters
        ----------
        prediction : Any
            Predicted value (numeric or string representation of a number).
        gold : Any
            Gold-standard value.
        tolerance_abs : float, optional
            Overrides instance-level tolerance_abs.
        tolerance_pct : float, optional
            Overrides instance-level tolerance_pct (0–100).
        """
        tol_abs = tolerance_abs if tolerance_abs is not None else self.tolerance_abs
        tol_pct = tolerance_pct if tolerance_pct is not None else self.tolerance_pct

        pred_f = self._to_float(prediction)
        gold_f = self._to_float(gold)

        if pred_f is None or gold_f is None:
            return AcceptanceResult(
                predicate=self.name,
                passed=False,
                score=0.0,
                details={"error": "Could not parse prediction or gold as float."},
            )

        abs_diff = abs(pred_f - gold_f)
        gold_mag = max(abs(gold_f), 1e-9)
        rel_pct = 100.0 * abs_diff / gold_mag

        # Scoring brackets (from paper)
        if abs_diff == 0.0:
            score = 1.0
            bracket = "exact"
        elif rel_pct <= 10.0:
            score = 0.9
            bracket = "within_10pct"
        elif rel_pct <= 20.0:
            score = 0.7
            bracket = "within_20pct"
        elif rel_pct <= 30.0:
            score = 0.5
            bracket = "within_30pct"
        else:
            score = 0.0
            bracket = "beyond_30pct"

        # Acceptance: use explicit tolerances if supplied, else based on bracket
        passed_abs = abs_diff <= tol_abs if tol_abs > 0 else (abs_diff == 0.0)
        passed_pct = rel_pct <= tol_pct
        passed = passed_abs or passed_pct

        return AcceptanceResult(
            predicate=self.name,
            passed=passed,
            score=score,
            details={
                "pred": pred_f,
                "gold": gold_f,
                "abs_diff": abs_diff,
                "rel_pct": round(rel_pct, 3),
                "bracket": bracket,
                "tolerance_abs": tol_abs,
                "tolerance_pct": tol_pct,
            },
        )


class CircularityAcceptance(BaseAcceptance):
    """Geometric shape metrics acceptance predicate.

    Checks circularity, compactness, and solidity values within ±r.

    Definitions
    -----------
    circularity   = 4π * area / perimeter²
    compactness   = perimeter² / area   (inverse of circularity)
    solidity      = area / convex_hull_area
    """

    name = "circularity"

    def __init__(self, tolerance: float = 0.05) -> None:
        self.tolerance = tolerance

    @staticmethod
    def _compute_circularity(area: float, perimeter: float) -> Optional[float]:
        if perimeter <= 0:
            return None
        return 4.0 * math.pi * area / (perimeter ** 2)

    @staticmethod
    def _compute_compactness(area: float, perimeter: float) -> Optional[float]:
        if area <= 0:
            return None
        return (perimeter ** 2) / area

    def evaluate(
        self,
        prediction: dict[str, float],
        gold: dict[str, float],
        tolerance: Optional[float] = None,
        **kwargs: Any,
    ) -> AcceptanceResult:
        """Evaluate geometric metric closeness.

        Parameters
        ----------
        prediction : dict[str, float]
            Dict with any subset of: ``"circularity"``, ``"compactness"``,
            ``"solidity"``.  Alternatively, may contain ``"area"`` and
            ``"perimeter"`` from which circularity and compactness are derived.
        gold : dict[str, float]
            Reference dict with the same keys.
        tolerance : float, optional
            Overrides instance-level tolerance.
        """
        tol = tolerance if tolerance is not None else self.tolerance
        details: dict[str, Any] = {}
        all_pass: list[bool] = []
        scores: list[float] = []

        # Derive circularity from area/perimeter if direct value absent
        def _get_metric(src: dict, key: str) -> Optional[float]:
            if key in src:
                return float(src[key])
            if key == "circularity" and "area" in src and "perimeter" in src:
                return self._compute_circularity(
                    float(src["area"]), float(src["perimeter"])
                )
            if key == "compactness" and "area" in src and "perimeter" in src:
                return self._compute_compactness(
                    float(src["area"]), float(src["perimeter"])
                )
            return None

        for metric in ("circularity", "compactness", "solidity"):
            pred_v = _get_metric(prediction, metric)
            gold_v = _get_metric(gold, metric)
            if pred_v is None or gold_v is None:
                continue
            diff = abs(pred_v - gold_v)
            passed = diff <= tol
            s = max(0.0, 1.0 - diff / max(tol, 1e-9))
            all_pass.append(passed)
            scores.append(s)
            details[metric] = {
                "pred": round(pred_v, 4),
                "gold": round(gold_v, 4),
                "diff": round(diff, 4),
                "passed": passed,
            }

        if not all_pass:
            return AcceptanceResult(
                predicate=self.name,
                passed=False,
                score=0.0,
                details={"error": "No geometric metrics could be evaluated."},
            )

        return AcceptanceResult(
            predicate=self.name,
            passed=all(all_pass),
            score=sum(scores) / len(scores),
            details=details,
        )


class CentroidAcceptance(BaseAcceptance):
    """Centroid distance acceptance predicate.

    Accepts a centroid prediction if either:
    - the pixel Euclidean distance is within *max_pixel_dist*, or
    - the relative distance (normalised by image diagonal) is within
      *max_relative_dist*.

    Parameters
    ----------
    max_pixel_dist : float
        Maximum acceptable pixel distance.
    max_relative_dist : float
        Maximum acceptable relative distance as a fraction of the image
        diagonal (0–1).
    """

    name = "centroid"

    def __init__(
        self,
        max_pixel_dist: float = 5.0,
        max_relative_dist: float = 0.02,
    ) -> None:
        self.max_pixel_dist = max_pixel_dist
        self.max_relative_dist = max_relative_dist

    def evaluate(
        self,
        prediction: tuple[float, float] | dict[str, float],
        gold: tuple[float, float] | dict[str, float],
        image_size: Optional[tuple[int, int]] = None,
        max_pixel_dist: Optional[float] = None,
        max_relative_dist: Optional[float] = None,
        **kwargs: Any,
    ) -> AcceptanceResult:
        """Evaluate centroid distance.

        Parameters
        ----------
        prediction : tuple (x, y) or dict with "x"/"y" keys
            Predicted centroid.
        gold : tuple (x, y) or dict with "x"/"y" keys
            Reference centroid.
        image_size : tuple (width, height), optional
            Required when using relative distance check.
        max_pixel_dist : float, optional
            Overrides instance-level maximum pixel distance.
        max_relative_dist : float, optional
            Overrides instance-level maximum relative distance.
        """
        mpd = max_pixel_dist if max_pixel_dist is not None else self.max_pixel_dist
        mrd = max_relative_dist if max_relative_dist is not None else self.max_relative_dist

        def _unpack(pt: Any) -> tuple[float, float]:
            if isinstance(pt, dict):
                return float(pt.get("x", pt.get("cx", 0))), float(pt.get("y", pt.get("cy", 0)))
            return float(pt[0]), float(pt[1])

        px, py = _unpack(prediction)
        gx, gy = _unpack(gold)

        pixel_dist = math.sqrt((px - gx) ** 2 + (py - gy) ** 2)
        passed_pixel = pixel_dist <= mpd

        passed_relative = False
        rel_dist: Optional[float] = None
        if image_size:
            w, h = image_size
            diagonal = math.sqrt(w ** 2 + h ** 2)
            if diagonal > 0:
                rel_dist = pixel_dist / diagonal
                passed_relative = rel_dist <= mrd

        passed = passed_pixel or passed_relative

        # Score: 1 if within pixel threshold, linearly decays
        score = max(0.0, 1.0 - pixel_dist / max(mpd, 1e-9)) if mpd > 0 else (1.0 if passed_pixel else 0.0)
        score = min(score, 1.0)

        return AcceptanceResult(
            predicate=self.name,
            passed=passed,
            score=score,
            details={
                "pred_centroid": (px, py),
                "gold_centroid": (gx, gy),
                "pixel_dist": round(pixel_dist, 3),
                "rel_dist": round(rel_dist, 5) if rel_dist is not None else None,
                "max_pixel_dist": mpd,
                "max_relative_dist": mrd,
            },
        )


class CoverageAcceptance(BaseAcceptance):
    """Mask coverage and IoU acceptance predicate.

    Accepts a mask prediction when:
    - the coverage percentage is within *tolerance* of the gold coverage, or
    - the IoU with the gold mask is at least *iou_threshold*.

    Parameters
    ----------
    iou_threshold : float
        Minimum IoU for acceptance (0–1).
    tolerance : float
        Absolute tolerance on mask coverage percentage (0–100).
    """

    name = "coverage"

    def __init__(
        self,
        iou_threshold: float = 0.5,
        tolerance: float = 5.0,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.tolerance = tolerance

    @staticmethod
    def _compute_iou_numpy(pred_mask: Any, gold_mask: Any) -> float:
        """Compute IoU between two binary masks using NumPy."""
        pred_bool = np.asarray(pred_mask, dtype=bool)
        gold_bool = np.asarray(gold_mask, dtype=bool)
        intersection = np.logical_and(pred_bool, gold_bool).sum()
        union = np.logical_or(pred_bool, gold_bool).sum()
        return float(intersection) / float(union) if union > 0 else 0.0

    @staticmethod
    def _coverage_pct(mask: Any) -> float:
        """Compute the fraction of mask pixels that are True, as a percentage."""
        if _NUMPY_AVAILABLE:
            arr = np.asarray(mask, dtype=bool)
            total = arr.size
            return 100.0 * float(arr.sum()) / total if total > 0 else 0.0
        # Pure-Python fallback for list-of-lists
        if isinstance(mask, list):
            flat = [v for row in mask for v in row]
            return 100.0 * sum(bool(v) for v in flat) / len(flat) if flat else 0.0
        return 0.0

    def evaluate(
        self,
        prediction: Any,
        gold: Any,
        image_total_pixels: Optional[int] = None,
        iou_threshold: Optional[float] = None,
        tolerance: Optional[float] = None,
        **kwargs: Any,
    ) -> AcceptanceResult:
        """Evaluate mask coverage and IoU.

        Parameters
        ----------
        prediction : array-like or float
            Binary mask (2-D array of 0/1 or bool) or a pre-computed
            coverage percentage (float 0–100).
        gold : array-like or float
            Reference mask or pre-computed coverage percentage.
        image_total_pixels : int, optional
            Total image pixels (used only when *prediction* / *gold* are
            scalar coverage percentages).
        iou_threshold : float, optional
            Overrides instance-level IoU threshold.
        tolerance : float, optional
            Overrides instance-level coverage tolerance.
        """
        iou_thr = iou_threshold if iou_threshold is not None else self.iou_threshold
        tol = tolerance if tolerance is not None else self.tolerance

        # Scalar coverage-percentage path
        if isinstance(prediction, (int, float)) and isinstance(gold, (int, float)):
            pred_cov = float(prediction)
            gold_cov = float(gold)
            diff = abs(pred_cov - gold_cov)
            passed = diff <= tol
            score = max(0.0, 1.0 - diff / max(tol, 1e-9))
            return AcceptanceResult(
                predicate=self.name,
                passed=passed,
                score=min(score, 1.0),
                details={
                    "pred_coverage_pct": round(pred_cov, 3),
                    "gold_coverage_pct": round(gold_cov, 3),
                    "diff": round(diff, 3),
                    "tolerance": tol,
                    "iou": None,
                },
            )

        # Mask-array path
        pred_cov = self._coverage_pct(prediction)
        gold_cov = self._coverage_pct(gold)
        cov_diff = abs(pred_cov - gold_cov)
        passed_cov = cov_diff <= tol

        iou: Optional[float] = None
        passed_iou = False
        if _NUMPY_AVAILABLE:
            try:
                iou = self._compute_iou_numpy(prediction, gold)
                passed_iou = iou >= iou_thr
            except Exception as exc:  # noqa: BLE001
                logger.warning("IoU computation failed: %s", exc)

        passed = passed_cov or passed_iou
        score = iou if iou is not None else max(0.0, 1.0 - cov_diff / max(tol, 1e-9))

        return AcceptanceResult(
            predicate=self.name,
            passed=passed,
            score=min(float(score), 1.0),
            details={
                "pred_coverage_pct": round(pred_cov, 3),
                "gold_coverage_pct": round(gold_cov, 3),
                "cov_diff": round(cov_diff, 3),
                "iou": round(iou, 4) if iou is not None else None,
                "iou_threshold": iou_thr,
                "tolerance": tol,
            },
        )


class OCRAcceptance(BaseAcceptance):
    """OCR-derived equality acceptance predicate.

    Normalises both prediction and gold text before comparing, accepting
    minor OCR artefacts (extra whitespace, punctuation, case differences).

    Parameters
    ----------
    case_sensitive : bool
        Whether the comparison is case-sensitive (default False).
    strip_punctuation : bool
        Remove punctuation before comparison (default True).
    """

    name = "ocr"

    def __init__(
        self,
        case_sensitive: bool = False,
        strip_punctuation: bool = True,
    ) -> None:
        self.case_sensitive = case_sensitive
        self.strip_punctuation = strip_punctuation

    def _normalise(self, text: str) -> str:
        """Apply normalisation pipeline to a text string."""
        # Unicode normalisation (NFC)
        text = unicodedata.normalize("NFC", text)
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()
        if not self.case_sensitive:
            text = text.lower()
        if self.strip_punctuation:
            text = re.sub(r"[^\w\s]", "", text, flags=re.UNICODE)
        return text

    def evaluate(
        self,
        prediction: str,
        gold: str,
        case_sensitive: Optional[bool] = None,
        **kwargs: Any,
    ) -> AcceptanceResult:
        """Evaluate OCR text equality after normalisation.

        Parameters
        ----------
        prediction : str
            OCR-extracted or predicted text.
        gold : str
            Gold-standard text.
        case_sensitive : bool, optional
            Overrides instance-level case_sensitive flag.
        """
        if case_sensitive is not None:
            prev = self.case_sensitive
            self.case_sensitive = case_sensitive
        else:
            prev = self.case_sensitive

        norm_pred = self._normalise(str(prediction))
        norm_gold = self._normalise(str(gold))

        self.case_sensitive = prev

        exact_match = norm_pred == norm_gold

        # Character-level similarity as a soft score
        if exact_match:
            score = 1.0
        else:
            longer = max(len(norm_pred), len(norm_gold))
            if longer == 0:
                score = 1.0
                exact_match = True
            else:
                # Levenshtein-inspired character overlap
                common = sum(a == b for a, b in zip(norm_pred, norm_gold))
                score = common / longer

        return AcceptanceResult(
            predicate=self.name,
            passed=exact_match,
            score=score,
            details={
                "norm_pred": norm_pred,
                "norm_gold": norm_gold,
                "exact_match": exact_match,
            },
        )


class WhitelistAcceptance(BaseAcceptance):
    """Whitelist term acceptance predicate.

    All required terms must appear in the prediction text for a full pass.
    Missing terms cap the maximum score at 0.5.

    Parameters
    ----------
    whitelist : list[str]
        Terms that must appear in the prediction.
    case_sensitive : bool
        Whether term matching is case-sensitive (default False).
    """

    name = "whitelist"

    def __init__(
        self,
        whitelist: Optional[list[str]] = None,
        case_sensitive: bool = False,
    ) -> None:
        self.whitelist: list[str] = whitelist or []
        self.case_sensitive = case_sensitive

    def evaluate(
        self,
        prediction: str,
        gold: Any = None,
        whitelist: Optional[list[str]] = None,
        case_sensitive: Optional[bool] = None,
        **kwargs: Any,
    ) -> AcceptanceResult:
        """Check that all whitelist terms are present in the prediction.

        Parameters
        ----------
        prediction : str
            The model's prediction text.
        gold : Any
            Not used by this predicate (kept for interface compatibility).
        whitelist : list[str], optional
            Overrides instance-level whitelist.
        case_sensitive : bool, optional
            Overrides instance-level case_sensitive flag.
        """
        wl = whitelist if whitelist is not None else self.whitelist
        cs = case_sensitive if case_sensitive is not None else self.case_sensitive

        if not wl:
            return AcceptanceResult(
                predicate=self.name,
                passed=True,
                score=1.0,
                details={"whitelist": [], "missing": []},
            )

        text = str(prediction) if cs else str(prediction).lower()
        missing = [
            term for term in wl
            if (term if cs else term.lower()) not in text
        ]
        found_count = len(wl) - len(missing)
        fraction_found = found_count / len(wl)

        # Paper rule: missing any term caps score at 0.5
        score = fraction_found if not missing else min(fraction_found, 0.5)

        return AcceptanceResult(
            predicate=self.name,
            passed=len(missing) == 0,
            score=score,
            details={
                "whitelist": wl,
                "missing": missing,
                "found_count": found_count,
                "total": len(wl),
            },
        )


class BlacklistAcceptance(BaseAcceptance):
    """Blacklist term acceptance predicate.

    If any forbidden term appears in the prediction, the score is
    immediately 0.0.

    Parameters
    ----------
    blacklist : list[str]
        Terms whose presence invalidates the prediction.
    case_sensitive : bool
        Whether term matching is case-sensitive (default False).
    """

    name = "blacklist"

    def __init__(
        self,
        blacklist: Optional[list[str]] = None,
        case_sensitive: bool = False,
    ) -> None:
        self.blacklist: list[str] = blacklist or []
        self.case_sensitive = case_sensitive

    def evaluate(
        self,
        prediction: str,
        gold: Any = None,
        blacklist: Optional[list[str]] = None,
        case_sensitive: Optional[bool] = None,
        **kwargs: Any,
    ) -> AcceptanceResult:
        """Check that no blacklist term is present in the prediction.

        Parameters
        ----------
        prediction : str
            The model's prediction text.
        gold : Any
            Not used by this predicate (kept for interface compatibility).
        blacklist : list[str], optional
            Overrides instance-level blacklist.
        case_sensitive : bool, optional
            Overrides instance-level case_sensitive flag.
        """
        bl = blacklist if blacklist is not None else self.blacklist
        cs = case_sensitive if case_sensitive is not None else self.case_sensitive

        if not bl:
            return AcceptanceResult(
                predicate=self.name,
                passed=True,
                score=1.0,
                details={"blacklist": [], "violations": []},
            )

        text = str(prediction) if cs else str(prediction).lower()
        violations = [
            term for term in bl
            if (term if cs else term.lower()) in text
        ]

        passed = len(violations) == 0
        return AcceptanceResult(
            predicate=self.name,
            passed=passed,
            score=0.0 if violations else 1.0,
            details={
                "blacklist": bl,
                "violations": violations,
            },
        )


class AreaAcceptance(BaseAcceptance):
    """Area-as-percentage-of-image acceptance predicate.

    Computes the tampered/detected area as a percentage of the total image
    pixels and checks whether it is within tolerance of the gold percentage.

    Parameters
    ----------
    tolerance : float
        Absolute tolerance on the area percentage (0–100).
    """

    name = "area"

    def __init__(self, tolerance: float = 5.0) -> None:
        self.tolerance = tolerance

    @staticmethod
    def _count_mask_pixels(mask: Any) -> int:
        """Count non-zero pixels in a mask (array or list of lists)."""
        if _NUMPY_AVAILABLE:
            arr = np.asarray(mask)
            return int((arr > 0).sum())
        if isinstance(mask, list):
            return sum(1 for row in mask for v in row if v)
        return int(mask)  # assume already a pixel count

    def compute_area_pct(
        self, mask: Any, image_total_pixels: int
    ) -> float:
        """Compute area as a percentage of total image pixels.

        Parameters
        ----------
        mask : array-like or int
            Binary mask or pre-counted pixel count.
        image_total_pixels : int
            Total number of pixels in the image (height × width).

        Returns
        -------
        float
            Area percentage in [0, 100].
        """
        if image_total_pixels <= 0:
            raise ValueError("image_total_pixels must be a positive integer.")
        mask_pixels = self._count_mask_pixels(mask)
        return 100.0 * mask_pixels / image_total_pixels

    def evaluate(
        self,
        prediction: Any,
        gold: Any,
        image_total_pixels: Optional[int] = None,
        tolerance: Optional[float] = None,
        **kwargs: Any,
    ) -> AcceptanceResult:
        """Evaluate area percentage.

        Parameters
        ----------
        prediction : array-like or float
            Predicted mask or pre-computed area percentage.
        gold : array-like or float
            Gold mask or pre-computed area percentage.
        image_total_pixels : int, optional
            Required when prediction/gold are masks rather than percentages.
        tolerance : float, optional
            Overrides instance-level tolerance.
        """
        tol = tolerance if tolerance is not None else self.tolerance

        # If scalars, treat as percentages directly
        if isinstance(prediction, (int, float)) and isinstance(gold, (int, float)):
            pred_pct = float(prediction)
            gold_pct = float(gold)
        else:
            if image_total_pixels is None:
                return AcceptanceResult(
                    predicate=self.name,
                    passed=False,
                    score=0.0,
                    details={"error": "image_total_pixels is required for mask inputs."},
                )
            pred_pct = self.compute_area_pct(prediction, image_total_pixels)
            gold_pct = self.compute_area_pct(gold, image_total_pixels)

        diff = abs(pred_pct - gold_pct)
        passed = diff <= tol
        score = max(0.0, min(1.0, 1.0 - diff / max(tol, 1e-9)))

        return AcceptanceResult(
            predicate=self.name,
            passed=passed,
            score=score,
            details={
                "pred_area_pct": round(pred_pct, 3),
                "gold_area_pct": round(gold_pct, 3),
                "diff": round(diff, 3),
                "tolerance": tol,
            },
        )


# ---------------------------------------------------------------------------
# Suite runner
# ---------------------------------------------------------------------------

def run_all_predicates(
    prediction: Any,
    gold: Any,
    config: Optional[AcceptanceConfig] = None,
    extra_kwargs: Optional[dict[str, Any]] = None,
) -> list[AcceptanceResult]:
    """Apply the standard suite of acceptance predicates.

    The predicates applied and their order:

    1. :class:`BlacklistAcceptance` (if blacklist is non-empty)
    2. :class:`WhitelistAcceptance` (if whitelist is non-empty)
    3. :class:`NumericalAcceptance` (if prediction can be parsed as a float)
    4. :class:`OCRAcceptance` (always, treating both as strings)

    Additional predicates (centroid, coverage, area, circularity) are run
    only when *extra_kwargs* provides the relevant keyword arguments.

    Parameters
    ----------
    prediction : Any
        Model output.
    gold : Any
        Gold-standard reference.
    config : AcceptanceConfig, optional
        Suite configuration.
    extra_kwargs : dict, optional
        Extra per-predicate kwargs.  Top-level keys match predicate names,
        e.g. ``{"centroid": {"image_size": (640, 480)}}``.

    Returns
    -------
    list[AcceptanceResult]
        One result per predicate that was evaluated.
    """
    if config is None:
        config = AcceptanceConfig()
    if extra_kwargs is None:
        extra_kwargs = {}

    results: list[AcceptanceResult] = []

    # 1. Blacklist
    if config.blacklist:
        res = BlacklistAcceptance(blacklist=config.blacklist).evaluate(
            prediction, gold, **extra_kwargs.get("blacklist", {})
        )
        results.append(res)
        # Short-circuit: if blacklist violated, score is 0
        if not res.passed:
            return results

    # 2. Whitelist
    if config.whitelist:
        results.append(
            WhitelistAcceptance(whitelist=config.whitelist).evaluate(
                prediction, gold, **extra_kwargs.get("whitelist", {})
            )
        )

    # 3. Numerical (if parseable)
    num_pred = NumericalAcceptance._to_float(prediction)
    num_gold = NumericalAcceptance._to_float(gold)
    if num_pred is not None and num_gold is not None:
        results.append(
            NumericalAcceptance(
                tolerance_abs=config.numerical_tolerance_abs,
                tolerance_pct=config.numerical_tolerance_pct,
            ).evaluate(prediction, gold, **extra_kwargs.get("numerical", {}))
        )

    # 4. OCR / text equality
    results.append(
        OCRAcceptance(case_sensitive=config.ocr_case_sensitive).evaluate(
            prediction, gold, **extra_kwargs.get("ocr", {})
        )
    )

    # 5. Coverage / mask IoU (opt-in)
    if "coverage" in extra_kwargs:
        results.append(
            CoverageAcceptance(
                iou_threshold=config.iou_threshold,
                tolerance=config.coverage_tolerance,
            ).evaluate(prediction, gold, **extra_kwargs["coverage"])
        )

    # 6. Area percentage (opt-in)
    if "area" in extra_kwargs:
        results.append(
            AreaAcceptance(tolerance=config.coverage_tolerance).evaluate(
                prediction, gold, **extra_kwargs["area"]
            )
        )

    # 7. Centroid distance (opt-in)
    if "centroid" in extra_kwargs:
        results.append(
            CentroidAcceptance(
                max_pixel_dist=config.centroid_max_pixel_dist,
                max_relative_dist=config.centroid_max_relative_dist,
            ).evaluate(prediction, gold, **extra_kwargs["centroid"])
        )

    # 8. Circularity / shape metrics (opt-in)
    if "circularity" in extra_kwargs:
        results.append(
            CircularityAcceptance(tolerance=config.geometry_tolerance).evaluate(
                prediction, gold, **extra_kwargs["circularity"]
            )
        )

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _main() -> None:  # pragma: no cover
    """CLI entry point for batch acceptance testing.

    Usage::

        python -m dftoolbench.evaluation.acceptance_tests samples.json \\
            [--config config.json] [--out results.json]

    The samples JSON must follow the schema::

        [
          {
            "sample_id": "s001",
            "prediction": <any>,
            "gold": <any>,
            "config_overrides": {...},   // optional AcceptanceConfig fields
            "extra_kwargs": {...}        // optional per-predicate kwargs
          },
          ...
        ]
    """
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(
        description="DFToolBench-I acceptance tests runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("samples_json", help="Path to samples JSON file.")
    parser.add_argument(
        "--config",
        default=None,
        help="Path to AcceptanceConfig JSON file.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output JSON path.  Prints to stdout if omitted.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    base_config = AcceptanceConfig()
    if args.config:
        with open(args.config, encoding="utf-8") as fh:
            cfg_data = json.load(fh)
        for k, v in cfg_data.items():
            if hasattr(base_config, k):
                setattr(base_config, k, v)

    with open(args.samples_json, encoding="utf-8") as fh:
        samples: list[dict] = json.load(fh)

    output: list[dict] = []
    for sample in samples:
        cfg = AcceptanceConfig(**{
            k: v for k, v in vars(base_config).items()
        })
        for k, v in sample.get("config_overrides", {}).items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)

        results = run_all_predicates(
            prediction=sample["prediction"],
            gold=sample["gold"],
            config=cfg,
            extra_kwargs=sample.get("extra_kwargs", {}),
        )
        output.append(
            {
                "sample_id": sample.get("sample_id", ""),
                "results": [r.to_dict() for r in results],
                "overall_passed": all(r.passed for r in results),
                "mean_score": (
                    sum(r.score for r in results) / len(results) if results else 0.0
                ),
            }
        )

    json_str = json.dumps(output, indent=2, ensure_ascii=False)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(json_str)
        print(f"Results written to {args.out}", file=sys.stderr)
    else:
        print(json_str)


if __name__ == "__main__":
    _main()
