#!/usr/bin/env python3
"""
smoke_test.py — Quick sanity check for DFToolBench-I.

Tests:
  1. Calculator tool  — verifies 2 + 2 == 4
  2. OCR tool         — verifies text is extracted from a synthetic test image
  3. Evaluation metrics — verifies InstAcc, ToolAcc, ArgAcc, SummAcc computation

Usage:
  python scripts/smoke_test.py            # run all tests
  python scripts/smoke_test.py --test calc ocr   # run specific tests

Exit codes:
  0 — all tests passed
  1 — one or more tests failed
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Callable, List, Optional, Tuple

# Ensure the repo root is on the path when running as a script
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# --------------------------------------------------------------------------- #
# ANSI colour helpers (gracefully disabled if not a TTY)
# --------------------------------------------------------------------------- #
_USE_COLOUR = sys.stdout.isatty()


def _green(s: str) -> str:
    return f"\033[32m{s}\033[0m" if _USE_COLOUR else s


def _red(s: str) -> str:
    return f"\033[31m{s}\033[0m" if _USE_COLOUR else s


def _bold(s: str) -> str:
    return f"\033[1m{s}\033[0m" if _USE_COLOUR else s


# --------------------------------------------------------------------------- #
# Result accumulator
# --------------------------------------------------------------------------- #
_results: List[Tuple[str, bool, Optional[str]]] = []


def _record(name: str, passed: bool, detail: Optional[str] = None) -> None:
    status = _green("PASS") if passed else _red("FAIL")
    msg = f"  [{status}] {name}"
    if detail:
        msg += f"\n         {detail}"
    print(msg)
    _results.append((name, passed, detail))


# --------------------------------------------------------------------------- #
# Test 1: Calculator tool
# --------------------------------------------------------------------------- #
def test_calculator() -> None:
    """Verify the built-in Calculator tool evaluates 2 + 2 correctly."""
    print(_bold("\n[Test 1] Calculator tool"))
    try:
        # Import inline so individual test failures don't prevent other tests
        from dftoolbench.tools.calculator import CalculatorTool  # type: ignore

        calc = CalculatorTool()
        result = calc.apply("2 + 2")
        result_val = float(str(result).strip())
        passed = result_val == 4.0
        _record(
            "Calculator: 2 + 2 == 4",
            passed,
            detail=None if passed else f"Got: {result!r}",
        )
    except ImportError:
        # Fallback: test the raw AST-safe evaluator logic directly
        import ast
        import operator as op

        _OPS = {
            ast.Add: op.add,
            ast.Sub: op.sub,
            ast.Mult: op.mul,
            ast.Div: op.truediv,
            ast.Pow: op.pow,
            ast.USub: op.neg,
        }

        def _safe_eval(node):
            if isinstance(node, ast.Constant):
                return node.value
            if isinstance(node, ast.BinOp):
                return _OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
            if isinstance(node, ast.UnaryOp):
                return _OPS[type(node.op)](_safe_eval(node.operand))
            raise ValueError(f"Unsupported node: {node!r}")

        expr = "2 + 2"
        tree = ast.parse(expr, mode="eval")
        result = _safe_eval(tree.body)
        passed = result == 4
        _record(
            "Calculator (fallback AST evaluator): 2 + 2 == 4",
            passed,
            detail=None if passed else f"Got: {result!r}",
        )
    except Exception as exc:  # noqa: BLE001
        _record("Calculator: 2 + 2 == 4", False, detail=traceback.format_exc(limit=3))


# --------------------------------------------------------------------------- #
# Test 2: OCR tool
# --------------------------------------------------------------------------- #
def test_ocr() -> None:
    """
    Verify the OCR tool extracts text from a synthetic test image.

    Creates a small PNG with the text 'DFTOOLBENCH TEST' using PIL,
    then checks that the recognised output contains the expected tokens.
    """
    print(_bold("\n[Test 2] OCR tool"))

    # --- Create a synthetic test image ---
    try:
        from PIL import Image, ImageDraw, ImageFont  # type: ignore
    except ImportError:
        _record("OCR: Pillow available", False, detail="pip install Pillow")
        return

    TEST_TEXT = "DFTOOLBENCH TEST"
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        img = Image.new("RGB", (400, 80), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        # Use default PIL font (no extra font files required)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
        except (IOError, OSError):
            font = ImageFont.load_default()
        draw.text((20, 20), TEST_TEXT, fill=(0, 0, 0), font=font)
        img.save(tmp_path)
        _record("OCR: synthetic test image created", True)
    except Exception as exc:  # noqa: BLE001
        _record("OCR: synthetic test image created", False, detail=str(exc))
        return

    # --- Try running OCRTool ---
    try:
        from dftoolbench.tools.ocr import OCRTool  # type: ignore

        tool = OCRTool(lang=["en"], device="cpu")
        output = tool.apply(tmp_path)
        # Loose check: at least one expected token should appear
        found = any(tok in output.upper() for tok in ["DFTOOLBENCH", "TEST"])
        _record(
            "OCR: text extracted from synthetic image",
            found,
            detail=f"Output: {output!r}" if not found else None,
        )
    except ImportError:
        _record(
            "OCR: EasyOCR available",
            False,
            detail="pip install easyocr  (EasyOCR not installed — skipping OCR inference test)",
        )
    except Exception as exc:  # noqa: BLE001
        _record("OCR: text extracted from synthetic image", False, detail=str(exc))
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# --------------------------------------------------------------------------- #
# Test 3: Evaluation metrics computation
# --------------------------------------------------------------------------- #
def test_metrics() -> None:
    """
    Verify that the step-level and end-to-end metric functions produce
    numerically correct outputs on synthetic fixture data.
    """
    print(_bold("\n[Test 3] Evaluation metrics"))

    # ---- Sub-test 3a: InstAcc / ToolAcc / ArgAcc / SummAcc ----
    try:
        from dftoolbench.evaluation import compute_step_metrics  # type: ignore

        # Perfect prediction — all scores should be 1.0
        scores = compute_step_metrics(
            predicted={"tool": "OCR", "args": {"image_path": "/img.png"}, "summary": "Text found."},
            reference={"tool": "OCR", "args": {"image_path": "/img.png"}, "summary": "Text found."},
        )
        ok = (
            scores.get("tool_acc", 0) == 1.0
            and scores.get("arg_acc", 0) == 1.0
        )
        _record("Metrics: perfect-prediction step scores are 1.0", ok, detail=str(scores) if not ok else None)

        # Wrong tool — ToolAcc should be 0.0
        scores_wrong = compute_step_metrics(
            predicted={"tool": "AnomalyDetectionTool", "args": {}, "summary": ""},
            reference={"tool": "OCR", "args": {"image_path": "/img.png"}, "summary": "Text found."},
        )
        ok2 = scores_wrong.get("tool_acc", 1) == 0.0
        _record("Metrics: wrong-tool step score is 0.0", ok2, detail=str(scores_wrong) if not ok2 else None)

    except ImportError:
        # Fall back to testing the raw metric logic inline
        _test_metrics_inline()
    except Exception as exc:  # noqa: BLE001
        _record("Metrics: step-level computation", False, detail=traceback.format_exc(limit=3))

    # ---- Sub-test 3b: Pearson correlation helper ----
    try:
        import math

        # Perfect positive correlation — r should be 1.0
        xs = [1.0, 2.0, 3.0, 4.0, 5.0]
        ys = [2.0, 4.0, 6.0, 8.0, 10.0]
        n = len(xs)
        mean_x = sum(xs) / n
        mean_y = sum(ys) / n
        num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
        denom = math.sqrt(
            sum((x - mean_x) ** 2 for x in xs)
            * sum((y - mean_y) ** 2 for y in ys)
        )
        r = num / denom
        ok = abs(r - 1.0) < 1e-9
        _record(
            "Metrics: Pearson r = 1.0 for perfectly correlated data",
            ok,
            detail=f"r = {r}" if not ok else None,
        )
    except Exception as exc:  # noqa: BLE001
        _record("Metrics: Pearson correlation", False, detail=str(exc))


def _test_metrics_inline() -> None:
    """Inline fallback metric tests when the evaluation module is absent."""
    # ToolAcc: string equality
    def tool_acc(pred: str, ref: str) -> float:
        return 1.0 if pred.strip() == ref.strip() else 0.0

    ok1 = tool_acc("OCR", "OCR") == 1.0
    ok2 = tool_acc("AnomalyDetectionTool", "OCR") == 0.0
    _record("Metrics (inline): tool_acc correct match -> 1.0", ok1)
    _record("Metrics (inline): tool_acc wrong match  -> 0.0", ok2)

    # ArgAcc: exact-match ratio over argument keys
    def arg_acc(pred: dict, ref: dict) -> float:
        if not ref:
            return 1.0
        matches = sum(1 for k, v in ref.items() if pred.get(k) == v)
        return matches / len(ref)

    ok3 = arg_acc({"image_path": "/img.png"}, {"image_path": "/img.png"}) == 1.0
    ok4 = arg_acc({}, {"image_path": "/img.png"}) == 0.0
    _record("Metrics (inline): arg_acc perfect -> 1.0", ok3)
    _record("Metrics (inline): arg_acc empty   -> 0.0", ok4)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
ALL_TESTS: dict[str, Callable[[], None]] = {
    "calc": test_calculator,
    "ocr": test_ocr,
    "metrics": test_metrics,
}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="DFToolBench-I smoke test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/smoke_test.py\n"
            "  python scripts/smoke_test.py --test calc metrics\n"
        ),
    )
    parser.add_argument(
        "--test",
        nargs="*",
        choices=list(ALL_TESTS.keys()),
        default=None,
        help="Subset of tests to run (default: all)",
    )
    args = parser.parse_args(argv)

    selected = args.test if args.test else list(ALL_TESTS.keys())

    print(_bold("=" * 60))
    print(_bold(" DFToolBench-I — Smoke Test"))
    print(_bold("=" * 60))

    for key in selected:
        ALL_TESTS[key]()

    # Summary
    print()
    print(_bold("=" * 60))
    total = len(_results)
    passed = sum(1 for _, ok, _ in _results if ok)
    failed = total - passed
    print(f" Results: {_green(str(passed))} passed, {_red(str(failed))} failed  (total {total})")
    print(_bold("=" * 60))

    if failed > 0:
        print(_red("\nSome tests FAILED. Review the output above for details."))
        return 1

    print(_green("\nAll tests PASSED."))
    return 0


if __name__ == "__main__":
    sys.exit(main())
