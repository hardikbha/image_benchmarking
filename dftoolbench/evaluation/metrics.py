"""
metrics.py — Step-level and end-to-end benchmark metrics for DFToolBench-I.

Based on the evaluation methodology described in Section V.A of the paper.

Metric definitions
------------------
**Step-level** (per predicted tool-use trace)

``inst_align``
    Instruction alignment accuracy — fraction of steps where the agent
    followed the intended sub-task instruction.
``tool_acc``
    Tool selection accuracy — fraction of steps where the agent invoked
    the correct tool.
``arg_acc``
    Argument accuracy — fraction of tool calls whose arguments match the
    reference arguments.
``answer_acc``
    Summary answer accuracy — fraction of final answers that are
    semantically correct (uses LLM judge scores).

**End-to-end** (per capability category)

``P&R``
    Perception & Restoration tasks.
``ML&A``
    Manipulation Localization & Attribution tasks.
``QR&A``
    Quantitative Reasoning & Authenticity Inference tasks.
``AnsAcc``
    Overall answer accuracy across all categories.

Tool category assignments
-------------------------
- P&R  : OCR, FaceDetectionTool, ObjectDetectionTool, DeepfakeDetectionTool,
          AnomalyDetectionTool, FingerprintingTool
- ML&A : DenoiseTool, SegmentationTool, TextForgeryLocalizerTool,
          CopyMoveLocalizationTool, SceneChangeDetectionTool
- QR&A : Calculator
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool → category mapping
# ---------------------------------------------------------------------------

#: Tools belonging to the Perception & Restoration category.
PR_TOOLS: frozenset[str] = frozenset(
    {
        "OCR",
        "FaceDetectionTool",
        "ObjectDetectionTool",
        "DeepfakeDetectionTool",
        "AnomalyDetectionTool",
        "FingerprintingTool",
    }
)

#: Tools belonging to the Manipulation Localization & Attribution category.
MLA_TOOLS: frozenset[str] = frozenset(
    {
        "DenoiseTool",
        "SegmentationTool",
        "TextForgeryLocalizerTool",
        "CopyMoveLocalizationTool",
        "SceneChangeDetectionTool",
    }
)

#: Tools belonging to the Quantitative Reasoning & Authenticity Inference category.
QRA_TOOLS: frozenset[str] = frozenset({"Calculator"})

_TOOL_TO_CATEGORY: dict[str, str] = {}
for _t in PR_TOOLS:
    _TOOL_TO_CATEGORY[_t] = "P&R"
for _t in MLA_TOOLS:
    _TOOL_TO_CATEGORY[_t] = "ML&A"
for _t in QRA_TOOLS:
    _TOOL_TO_CATEGORY[_t] = "QR&A"


def get_tool_category(tool_name: str) -> str:
    """Return the capability category for a given tool name.

    Parameters
    ----------
    tool_name : str
        Name of the tool (case-sensitive).

    Returns
    -------
    str
        ``"P&R"``, ``"ML&A"``, ``"QR&A"``, or ``"unknown"`` if the tool
        is not in any predefined category.
    """
    return _TOOL_TO_CATEGORY.get(tool_name, "unknown")


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class StepMetrics:
    """Aggregated step-level metrics for a collection of traces.

    Parameters
    ----------
    inst_align : float
        Instruction alignment accuracy in [0, 1].
    tool_acc : float
        Tool selection accuracy in [0, 1].
    arg_acc : float
        Argument accuracy in [0, 1].
    answer_acc : float
        Summary answer accuracy in [0, 1].
    n_steps : int
        Total number of steps evaluated.
    n_samples : int
        Total number of samples (traces) evaluated.
    """

    inst_align: float = 0.0
    tool_acc: float = 0.0
    arg_acc: float = 0.0
    answer_acc: float = 0.0
    n_steps: int = 0
    n_samples: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dictionary."""
        return {
            "inst_align": round(self.inst_align, 4),
            "tool_acc": round(self.tool_acc, 4),
            "arg_acc": round(self.arg_acc, 4),
            "answer_acc": round(self.answer_acc, 4),
            "n_steps": self.n_steps,
            "n_samples": self.n_samples,
        }


@dataclass
class CategoryMetrics:
    """End-to-end metrics for a single capability category.

    Parameters
    ----------
    category : str
        Category name (``"P&R"``, ``"ML&A"``, or ``"QR&A"``).
    precision : float
        Precision in [0, 1].
    recall : float
        Recall in [0, 1].
    f1 : float
        F1 score in [0, 1].
    answer_acc : float
        Answer accuracy in [0, 1].
    n_samples : int
        Number of samples in this category.
    """

    category: str = ""
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    answer_acc: float = 0.0
    n_samples: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dictionary."""
        return {
            "category": self.category,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "answer_acc": round(self.answer_acc, 4),
            "n_samples": self.n_samples,
        }


@dataclass
class BenchmarkMetrics:
    """Full benchmark metrics report.

    Parameters
    ----------
    step : StepMetrics
        Aggregated step-level metrics.
    categories : dict[str, CategoryMetrics]
        Per-category end-to-end metrics.
    ans_acc : float
        Overall answer accuracy across all categories.
    per_model : dict[str, StepMetrics]
        Step metrics broken down by agent/model name.
    per_dataset : dict[str, StepMetrics]
        Step metrics broken down by dataset name.
    pearson_correlations : dict[str, float]
        Pearson correlation coefficients between metric pairs.
    """

    step: StepMetrics = field(default_factory=StepMetrics)
    categories: dict[str, CategoryMetrics] = field(default_factory=dict)
    ans_acc: float = 0.0
    per_model: dict[str, StepMetrics] = field(default_factory=dict)
    per_dataset: dict[str, StepMetrics] = field(default_factory=dict)
    pearson_correlations: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dictionary."""
        return {
            "step": self.step.to_dict(),
            "categories": {k: v.to_dict() for k, v in self.categories.items()},
            "ans_acc": round(self.ans_acc, 4),
            "per_model": {k: v.to_dict() for k, v in self.per_model.items()},
            "per_dataset": {k: v.to_dict() for k, v in self.per_dataset.items()},
            "pearson_correlations": {
                k: round(v, 4) for k, v in self.pearson_correlations.items()
            },
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Return numerator / denominator, or *default* when denominator is zero."""
    return numerator / denominator if denominator != 0.0 else default


def _f1(precision: float, recall: float) -> float:
    """Compute F1 from precision and recall."""
    denom = precision + recall
    return _safe_div(2.0 * precision * recall, denom)


def _pearson(x: Sequence[float], y: Sequence[float]) -> float:
    """Compute Pearson correlation coefficient between two equal-length sequences.

    Returns 0.0 when the standard deviation of either sequence is zero.
    """
    n = len(x)
    if n < 2:
        return 0.0
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    std_x = math.sqrt(sum((xi - mean_x) ** 2 for xi in x))
    std_y = math.sqrt(sum((yi - mean_y) ** 2 for yi in y))
    denom = std_x * std_y
    return _safe_div(cov, denom)


def _arg_match(pred_args: Any, ref_args: Any) -> float:
    """Compute argument match score between predicted and reference arguments.

    Parameters
    ----------
    pred_args : Any
        Predicted arguments (dict, list, or scalar).
    ref_args : Any
        Reference arguments (dict, list, or scalar).

    Returns
    -------
    float
        1.0 for exact match, partial score for dicts/lists, 0.0 otherwise.
    """
    if pred_args == ref_args:
        return 1.0
    if isinstance(ref_args, dict) and isinstance(pred_args, dict):
        if not ref_args:
            return 1.0
        matched = sum(
            1 for k, v in ref_args.items() if pred_args.get(k) == v
        )
        return matched / len(ref_args)
    if isinstance(ref_args, list) and isinstance(pred_args, list):
        if not ref_args:
            return 1.0
        matched = sum(
            1 for a, b in zip(ref_args, pred_args) if a == b
        )
        return matched / max(len(ref_args), len(pred_args))
    return 0.0


# ---------------------------------------------------------------------------
# Step-level computation
# ---------------------------------------------------------------------------

def _compute_step_metrics(
    trace_list: list[dict[str, Any]],
    judge_scores: Optional[dict[str, float]] = None,
) -> StepMetrics:
    """Compute step-level metrics from a list of trace records.

    Parameters
    ----------
    trace_list : list[dict]
        Each element represents one sample and must have:

        - ``"predicted_steps"`` (list[dict]): model's tool calls.  Each step
          dict should have ``"tool"`` (str) and optionally ``"args"`` (any).
        - ``"reference_steps"`` (list[dict]): gold-standard tool calls.  Same
          schema as predicted steps.
        - ``"inst_align"`` (float, optional): per-sample instruction alignment
          score.  If not provided, computed as ``tool_acc`` for that sample.
        - ``"sample_id"`` (str, optional): used to look up *judge_scores*.

    judge_scores : dict[str, float], optional
        Mapping from sample_id to answer accuracy score (from LLM judge).
        When provided, used for the ``answer_acc`` metric.

    Returns
    -------
    StepMetrics
        Aggregated step-level metrics.
    """
    total_steps = 0
    sum_ia = 0.0
    sum_ta = 0.0
    sum_aa = 0.0
    sum_ans = 0.0
    n_samples = len(trace_list)

    for trace in trace_list:
        pred_steps: list[dict] = trace.get("predicted_steps", [])
        ref_steps: list[dict] = trace.get("reference_steps", [])

        n = max(len(pred_steps), len(ref_steps))
        if n == 0:
            continue

        # Per-step tool accuracy and arg accuracy
        step_ta = 0.0
        step_aa = 0.0
        for i, ref in enumerate(ref_steps):
            pred = pred_steps[i] if i < len(pred_steps) else {}
            correct_tool = int(pred.get("tool", "") == ref.get("tool", ""))
            step_ta += correct_tool
            step_aa += _arg_match(pred.get("args"), ref.get("args"))

        step_ta /= len(ref_steps) if ref_steps else 1
        step_aa /= len(ref_steps) if ref_steps else 1

        # Instruction alignment: use explicit field if present, else use tool_acc
        ia = trace.get("inst_align", step_ta)

        # Answer accuracy
        sid = trace.get("sample_id", "")
        if judge_scores and sid in judge_scores:
            ans = judge_scores[sid]
        else:
            ans = float(trace.get("answer_score", 0.0))

        sum_ia += ia
        sum_ta += step_ta
        sum_aa += step_aa
        sum_ans += ans
        total_steps += len(ref_steps)

    return StepMetrics(
        inst_align=_safe_div(sum_ia, n_samples),
        tool_acc=_safe_div(sum_ta, n_samples),
        arg_acc=_safe_div(sum_aa, n_samples),
        answer_acc=_safe_div(sum_ans, n_samples),
        n_steps=total_steps,
        n_samples=n_samples,
    )


# ---------------------------------------------------------------------------
# End-to-end computation
# ---------------------------------------------------------------------------

def _compute_category_metrics(
    traces_by_category: dict[str, list[dict[str, Any]]],
    judge_scores: Optional[dict[str, float]] = None,
) -> dict[str, CategoryMetrics]:
    """Compute per-category end-to-end metrics.

    For each category, precision and recall are computed over correct tool
    selections (treating the task as a binary classification: correct tool
    sequence vs. incorrect).

    Parameters
    ----------
    traces_by_category : dict[str, list[dict]]
        Traces grouped by their capability category.
    judge_scores : dict[str, float], optional
        LLM judge answer scores keyed by sample_id.

    Returns
    -------
    dict[str, CategoryMetrics]
        Category name → :class:`CategoryMetrics`.
    """
    results: dict[str, CategoryMetrics] = {}

    for cat, traces in traces_by_category.items():
        tp = fp = fn = 0
        ans_sum = 0.0

        for trace in traces:
            pred_steps: list[dict] = trace.get("predicted_steps", [])
            ref_steps: list[dict] = trace.get("reference_steps", [])
            sid = trace.get("sample_id", "")

            pred_tools = {s.get("tool", "") for s in pred_steps}
            ref_tools = {s.get("tool", "") for s in ref_steps}

            # TP: tools correctly predicted; FP: extra predicted; FN: missed
            tp += len(pred_tools & ref_tools)
            fp += len(pred_tools - ref_tools)
            fn += len(ref_tools - pred_tools)

            if judge_scores and sid in judge_scores:
                ans_sum += judge_scores[sid]
            else:
                ans_sum += float(trace.get("answer_score", 0.0))

        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _f1(precision, recall)
        n = len(traces)

        results[cat] = CategoryMetrics(
            category=cat,
            precision=precision,
            recall=recall,
            f1=f1,
            answer_acc=_safe_div(ans_sum, n),
            n_samples=n,
        )

    return results


# ---------------------------------------------------------------------------
# Pearson correlations
# ---------------------------------------------------------------------------

def _compute_pearson_correlations(
    traces: list[dict[str, Any]],
    judge_scores: Optional[dict[str, float]] = None,
) -> dict[str, float]:
    """Compute Pearson correlations between pairs of step-level metrics.

    Parameters
    ----------
    traces : list[dict]
        All trace records (see :func:`_compute_step_metrics` for schema).
    judge_scores : dict[str, float], optional
        LLM judge scores keyed by sample_id.

    Returns
    -------
    dict[str, float]
        Keys of the form ``"X_vs_Y"`` mapping to Pearson r.
    """
    ia_vals: list[float] = []
    ta_vals: list[float] = []
    aa_vals: list[float] = []
    ans_vals: list[float] = []

    for trace in traces:
        pred_steps: list[dict] = trace.get("predicted_steps", [])
        ref_steps: list[dict] = trace.get("reference_steps", [])

        if not ref_steps:
            continue

        step_ta = sum(
            1 for i, ref in enumerate(ref_steps)
            if i < len(pred_steps) and pred_steps[i].get("tool") == ref.get("tool")
        ) / len(ref_steps)

        step_aa = sum(
            _arg_match(
                pred_steps[i].get("args") if i < len(pred_steps) else None,
                ref.get("args"),
            )
            for i, ref in enumerate(ref_steps)
        ) / len(ref_steps)

        ia = trace.get("inst_align", step_ta)
        sid = trace.get("sample_id", "")
        ans = (
            judge_scores[sid]
            if judge_scores and sid in judge_scores
            else float(trace.get("answer_score", 0.0))
        )

        ia_vals.append(ia)
        ta_vals.append(step_ta)
        aa_vals.append(step_aa)
        ans_vals.append(ans)

    return {
        "inst_align_vs_tool_acc": _pearson(ia_vals, ta_vals),
        "inst_align_vs_arg_acc": _pearson(ia_vals, aa_vals),
        "inst_align_vs_answer_acc": _pearson(ia_vals, ans_vals),
        "tool_acc_vs_arg_acc": _pearson(ta_vals, aa_vals),
        "tool_acc_vs_answer_acc": _pearson(ta_vals, ans_vals),
        "arg_acc_vs_answer_acc": _pearson(aa_vals, ans_vals),
    }


# ---------------------------------------------------------------------------
# Main compute entry point
# ---------------------------------------------------------------------------

def compute_metrics(
    traces: list[dict[str, Any]],
    judge_scores: Optional[dict[str, float]] = None,
) -> BenchmarkMetrics:
    """Compute all benchmark metrics from a list of trace records.

    Each trace record should contain:

    .. code-block:: python

        {
          "sample_id":        str,                    # unique sample identifier
          "model":            str,                    # agent/model name
          "dataset":          str,                    # dataset name
          "category":         str,                    # "P&R" | "ML&A" | "QR&A" (optional)
          "predicted_steps":  [{"tool": str, "args": any}, ...],
          "reference_steps":  [{"tool": str, "args": any}, ...],
          "inst_align":       float,                  # optional, per-sample
          "answer_score":     float,                  # optional; overridden by judge_scores
        }

    If ``"category"`` is absent it is inferred from the first tool used in
    ``reference_steps``.

    Parameters
    ----------
    traces : list[dict]
        Collection of trace records.
    judge_scores : dict[str, float], optional
        LLM judge scores keyed by ``sample_id``.  When provided, these
        override any per-trace ``"answer_score"`` fields.

    Returns
    -------
    BenchmarkMetrics
        Fully populated benchmark metrics object.
    """
    # Assign category where missing
    for trace in traces:
        if "category" not in trace:
            ref_steps = trace.get("reference_steps", [])
            first_tool = ref_steps[0].get("tool", "") if ref_steps else ""
            trace["category"] = get_tool_category(first_tool)

    # ---- global step metrics -----------------------------------------------
    global_step = _compute_step_metrics(traces, judge_scores)

    # ---- group by model -------------------------------------------------------
    by_model: dict[str, list[dict]] = defaultdict(list)
    for t in traces:
        by_model[t.get("model", "unknown")].append(t)
    per_model = {
        m: _compute_step_metrics(ts, judge_scores)
        for m, ts in by_model.items()
    }

    # ---- group by dataset ----------------------------------------------------
    by_dataset: dict[str, list[dict]] = defaultdict(list)
    for t in traces:
        by_dataset[t.get("dataset", "unknown")].append(t)
    per_dataset = {
        d: _compute_step_metrics(ts, judge_scores)
        for d, ts in by_dataset.items()
    }

    # ---- group by capability category ----------------------------------------
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for t in traces:
        by_cat[t["category"]].append(t)
    category_metrics = _compute_category_metrics(by_cat, judge_scores)

    # ---- overall answer accuracy (macro-average across categories) -----------
    cat_ans = [cm.answer_acc for cm in category_metrics.values() if cm.n_samples > 0]
    ans_acc = sum(cat_ans) / len(cat_ans) if cat_ans else 0.0

    # ---- Pearson correlations -------------------------------------------------
    pearson = _compute_pearson_correlations(traces, judge_scores)

    return BenchmarkMetrics(
        step=global_step,
        categories=category_metrics,
        ans_acc=ans_acc,
        per_model=per_model,
        per_dataset=per_dataset,
        pearson_correlations=pearson,
    )


# ---------------------------------------------------------------------------
# Convenience: load traces from JSON
# ---------------------------------------------------------------------------

def load_traces(path: str) -> list[dict[str, Any]]:
    """Load a traces JSON file from disk.

    Parameters
    ----------
    path : str
        Path to a JSON file containing a list of trace records or a dict
        mapping sample_id → trace record.

    Returns
    -------
    list[dict]
        List of trace records.
    """
    import json

    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict):
        records = []
        for sid, rec in data.items():
            if "sample_id" not in rec:
                rec = dict(rec)
                rec["sample_id"] = sid
            records.append(rec)
        return records
    return list(data)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _main() -> None:  # pragma: no cover
    """CLI entry point.

    Usage::

        python -m dftoolbench.evaluation.metrics traces.json \\
            [--judge-scores judge_results.json] [--out metrics.json]
    """
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(
        description="DFToolBench-I benchmark metrics computation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("traces_json", help="Path to traces JSON file.")
    parser.add_argument(
        "--judge-scores",
        default=None,
        help="Path to LLM judge results JSON (output of llm_judge.py).",
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

    traces = load_traces(args.traces_json)

    judge_scores: Optional[dict[str, float]] = None
    if args.judge_scores:
        with open(args.judge_scores, encoding="utf-8") as fh:
            raw = json.load(fh)
        # Accept both {results: {sid: {score: ...}}} and {sid: score}
        if "results" in raw:
            judge_scores = {
                sid: v["score"] for sid, v in raw["results"].items()
            }
        else:
            judge_scores = {sid: float(v) for sid, v in raw.items()}

    metrics = compute_metrics(traces, judge_scores)
    json_str = json.dumps(metrics.to_dict(), indent=2, ensure_ascii=False)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(json_str)
        print(f"Metrics written to {args.out}", file=sys.stderr)
    else:
        print(json_str)


if __name__ == "__main__":
    _main()
