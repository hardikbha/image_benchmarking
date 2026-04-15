"""
dftoolbench.evaluation
======================

Evaluation pipeline for DFToolBench-I: Benchmarking Tool-Augmented Agents
for Image-Based Deepfake Detection.

Sub-modules
-----------
llm_judge
    LLM-as-a-Judge evaluator supporting Claude and Gemini backends.
metrics
    Step-level and end-to-end benchmark metrics (Section V.A of the paper).
react_controller
    ReAct agent controller for tool-augmented forensic reasoning.
acceptance_tests
    Gold-answer acceptance predicates (tolerances, masks, OCR, etc.).
"""

from dftoolbench.evaluation.acceptance_tests import (
    AcceptanceConfig,
    AcceptanceResult,
    AreaAcceptance,
    BlacklistAcceptance,
    CentroidAcceptance,
    CircularityAcceptance,
    CoverageAcceptance,
    NumericalAcceptance,
    OCRAcceptance,
    WhitelistAcceptance,
    run_all_predicates,
)
from dftoolbench.evaluation.llm_judge import JudgeConfig, JudgeResult, LLMJudge
from dftoolbench.evaluation.metrics import BenchmarkMetrics, StepMetrics
from dftoolbench.evaluation.react_controller import AgentMode, ReActController

__all__ = [
    # acceptance_tests
    "AcceptanceConfig",
    "AcceptanceResult",
    "AreaAcceptance",
    "BlacklistAcceptance",
    "CentroidAcceptance",
    "CircularityAcceptance",
    "CoverageAcceptance",
    "NumericalAcceptance",
    "OCRAcceptance",
    "WhitelistAcceptance",
    "run_all_predicates",
    # llm_judge
    "JudgeConfig",
    "JudgeResult",
    "LLMJudge",
    # metrics
    "BenchmarkMetrics",
    "StepMetrics",
    # react_controller
    "AgentMode",
    "ReActController",
]
