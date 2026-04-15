"""
llm_judge.py — LLM-as-a-Judge evaluator for DFToolBench-I.

Based on the methodology described in Section IV.D of the paper.

Supports two judge backends:
  - **Claude** (via the ``anthropic`` SDK; model defaults to
    ``claude-3-5-sonnet-20241022``).
  - **Gemini** (via the ``google.generativeai`` SDK; model defaults to
    ``gemini-1.5-pro``).

Scoring rubric
--------------
The judge is instructed to produce a JSON response with the following fields:

``score`` (float, 0–1)
    Composite quality score derived from the rubric below.
``reasoning`` (str)
    Free-text explanation of the score.
``whitelist_match`` (bool)
    Whether *all* required terms appear in the prediction.
``blacklist_violation`` (bool)
    Whether *any* forbidden term appears in the prediction.
``numerical_accuracy`` (str | null)
    Bracketed accuracy class when the answer contains a numerical value:
    "exact", "within_10pct", "within_20pct", "within_30pct", "beyond_30pct".

Rubric summary
~~~~~~~~~~~~~~
- **Numerical answers**: Exact → 1.0 | ±10 % → 0.9 | ±20 % → 0.7 |
  ±30 % → 0.5 | >30 % → 0.0
- **Counting answers**: Exact → 1.0 | Off-by-1 → 0.8 | Off-by-2 → 0.5 |
  Off-by-3+ → 0.0
- **Whitelist**: any missing required term caps score at 0.5
- **Blacklist**: any forbidden term → 0.0 immediately
- **Semantic equivalence** is accepted (model uses NLU).

CLI usage::

    python -m dftoolbench.evaluation.llm_judge predictions.json \\
        [--judge claude|gemini] [--model <name>] [--out results.json]

Environment variables
---------------------
ANTHROPIC_API_KEY
    Required when ``--judge claude``.
GEMINI_API_KEY
    Required when ``--judge gemini``.
DFTOOLBENCH_JUDGE_MODEL
    Default model override (overridden by ``--model``).
DFTOOLBENCH_JUDGE_BACKEND
    Default backend (``claude`` or ``gemini``; overridden by ``--judge``).
DFTOOLBENCH_JUDGE_MAX_TOKENS
    Maximum tokens for judge response (default 1024).
DFTOOLBENCH_JUDGE_TEMPERATURE
    Sampling temperature for judge (default 0.0).
"""

from __future__ import annotations

import json
import logging
import os
import re
import textwrap
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional SDK imports
# ---------------------------------------------------------------------------

try:
    import anthropic as _anthropic  # type: ignore
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _anthropic = None  # type: ignore
    _ANTHROPIC_AVAILABLE = False

try:
    import google.generativeai as _genai  # type: ignore
    _GENAI_AVAILABLE = True
except ImportError:
    _genai = None  # type: ignore
    _GENAI_AVAILABLE = False

# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

_DEFAULT_CLAUDE_MODEL = "claude-3-5-sonnet-20241022"
_DEFAULT_GEMINI_MODEL = "gemini-1.5-pro"


@dataclass
class JudgeConfig:
    """Configuration for the LLM judge.

    Parameters
    ----------
    backend : str
        Judge backend, either ``"claude"`` or ``"gemini"``.
    model : str
        Model identifier.  Defaults depend on the chosen backend.
    temperature : float
        Sampling temperature (default 0.0 for deterministic output).
    max_tokens : int
        Maximum tokens in the judge's response.
    """

    backend: str = field(
        default_factory=lambda: os.environ.get("DFTOOLBENCH_JUDGE_BACKEND", "claude")
    )
    model: str = field(default="")
    temperature: float = field(
        default_factory=lambda: float(
            os.environ.get("DFTOOLBENCH_JUDGE_TEMPERATURE", "0.0")
        )
    )
    max_tokens: int = field(
        default_factory=lambda: int(
            os.environ.get("DFTOOLBENCH_JUDGE_MAX_TOKENS", "1024")
        )
    )

    def __post_init__(self) -> None:
        if not self.model:
            env_model = os.environ.get("DFTOOLBENCH_JUDGE_MODEL", "")
            if env_model:
                self.model = env_model
            elif self.backend == "gemini":
                self.model = _DEFAULT_GEMINI_MODEL
            else:
                self.model = _DEFAULT_CLAUDE_MODEL


@dataclass
class JudgeResult:
    """Result returned by the LLM judge for a single prediction.

    Parameters
    ----------
    score : float
        Composite score in [0, 1].
    reasoning : str
        Judge's explanation.
    whitelist_match : bool
        True when all required terms are present.
    blacklist_violation : bool
        True when any forbidden term is present.
    numerical_accuracy : str or None
        Accuracy bracket for numerical answers, or ``None``.
    raw_response : str
        Unprocessed judge output (for debugging).
    """

    score: float
    reasoning: str
    whitelist_match: bool
    blacklist_violation: bool
    numerical_accuracy: Optional[str]
    raw_response: str = field(default="", repr=False)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dictionary."""
        return {
            "score": self.score,
            "reasoning": self.reasoning,
            "whitelist_match": self.whitelist_match,
            "blacklist_violation": self.blacklist_violation,
            "numerical_accuracy": self.numerical_accuracy,
        }


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = textwrap.dedent(
    """\
    You are an expert evaluator for image-based deepfake detection benchmarks.
    Your task is to compare a model's prediction against a gold-standard reference
    answer and return a structured JSON evaluation.

    ## Scoring Rubric

    ### Numerical answers
    Compute the relative deviation |pred - gold| / max(|gold|, 1e-9):
    - Exact (deviation = 0): score = 1.0
    - Within ±10 %           : score = 0.9
    - Within ±20 %           : score = 0.7
    - Within ±30 %           : score = 0.5
    - Beyond ±30 %           : score = 0.0

    ### Counting answers (non-negative integers)
    - Exact match     : score = 1.0
    - Off by 1        : score = 0.8
    - Off by 2        : score = 0.5
    - Off by 3 or more: score = 0.0

    ### Whitelist
    If a whitelist of required terms is provided, every term must appear in the
    prediction (case-insensitive, partial string match allowed).
    Any missing term caps the maximum possible score at 0.5.

    ### Blacklist
    If a blacklist of forbidden terms is provided and any forbidden term appears
    in the prediction, the score is immediately 0.0.

    ### Semantic equivalence
    Minor paraphrasing, synonyms, and re-ordering of equivalent information are
    acceptable.  Focus on factual correctness, not surface form.

    ## Output format
    Return **only** a JSON object — no markdown, no preamble — with exactly
    these keys:
    {
      "score": <float 0-1>,
      "reasoning": "<concise explanation>",
      "whitelist_match": <true|false>,
      "blacklist_violation": <true|false>,
      "numerical_accuracy": "<exact|within_10pct|within_20pct|within_30pct|beyond_30pct|null>"
    }

    Use JSON null (not the string "null") when numerical_accuracy is not
    applicable.
    """
)

_USER_TEMPLATE = textwrap.dedent(
    """\
    ## Question / Task
    {question}

    ## Gold-standard answer
    {gold}

    ## Model prediction
    {prediction}

    ## Whitelist (required terms, may be empty)
    {whitelist}

    ## Blacklist (forbidden terms, may be empty)
    {blacklist}

    ## Additional reference context (may be empty)
    {references}

    Evaluate the prediction and return the JSON score object.
    """
)


# ---------------------------------------------------------------------------
# LLM Judge
# ---------------------------------------------------------------------------

class LLMJudge:
    """LLM-as-a-Judge evaluator.

    Parameters
    ----------
    config : JudgeConfig, optional
        Judge configuration.  A default ``JudgeConfig`` is constructed when
        not supplied.

    Examples
    --------
    >>> judge = LLMJudge()
    >>> scores = judge.evaluate_with_gt(
    ...     predictions={"q1": "The image is authentic."},
    ...     gold={"q1": "Authentic"},
    ...     references={"q1": {"question": "Is this image real?"}},
    ... )
    >>> print(scores["q1"].score)
    """

    def __init__(self, config: Optional[JudgeConfig] = None) -> None:
        self.config = config or JudgeConfig()
        self._client: Any = None
        self._init_client()

    # ------------------------------------------------------------------
    # Client initialisation
    # ------------------------------------------------------------------

    def _init_client(self) -> None:
        """Initialise the SDK client for the configured backend."""
        backend = self.config.backend.lower()
        if backend == "claude":
            if not _ANTHROPIC_AVAILABLE:
                raise ImportError(
                    "The 'anthropic' package is required for the Claude backend. "
                    "Install it with: pip install anthropic"
                )
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise EnvironmentError(
                    "ANTHROPIC_API_KEY environment variable is not set."
                )
            self._client = _anthropic.Anthropic(api_key=api_key)
        elif backend == "gemini":
            if not _GENAI_AVAILABLE:
                raise ImportError(
                    "The 'google-generativeai' package is required for the Gemini "
                    "backend.  Install it with: pip install google-generativeai"
                )
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise EnvironmentError(
                    "GEMINI_API_KEY environment variable is not set."
                )
            _genai.configure(api_key=api_key)
            self._client = _genai.GenerativeModel(
                model_name=self.config.model,
                system_instruction=_SYSTEM_PROMPT,
            )
        else:
            raise ValueError(
                f"Unknown judge backend: {self.config.backend!r}. "
                "Choose 'claude' or 'gemini'."
            )

    # ------------------------------------------------------------------
    # Internal: call the judge model
    # ------------------------------------------------------------------

    def _call_claude(self, user_message: str) -> str:
        """Send a single-turn request to the Claude API and return raw text."""
        response = self._client.messages.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
        return response.content[0].text

    def _call_gemini(self, user_message: str) -> str:
        """Send a single-turn request to the Gemini API and return raw text."""
        generation_config = _genai.GenerationConfig(  # type: ignore[attr-defined]
            temperature=self.config.temperature,
            max_output_tokens=self.config.max_tokens,
        )
        response = self._client.generate_content(
            user_message,
            generation_config=generation_config,
        )
        return response.text

    def _call_judge(self, user_message: str) -> str:
        """Dispatch to the configured backend and return raw judge text."""
        backend = self.config.backend.lower()
        if backend == "claude":
            return self._call_claude(user_message)
        return self._call_gemini(user_message)

    # ------------------------------------------------------------------
    # Internal: parse judge JSON
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_response(raw: str) -> JudgeResult:
        """Parse a raw judge response into a :class:`JudgeResult`.

        Attempts strict JSON parsing first, then falls back to a regex
        extraction strategy to handle models that wrap JSON in markdown fences.

        Parameters
        ----------
        raw : str
            Raw text output from the judge model.

        Returns
        -------
        JudgeResult
            Parsed result.  On parse failure a zero-score result is returned
            with the error logged.
        """
        text = raw.strip()

        # Strip markdown code fences if present
        fence_match = re.search(r"```(?:json)?\s*([\s\S]+?)```", text)
        if fence_match:
            text = fence_match.group(1).strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Attempt to extract the first {...} block
            brace_match = re.search(r"\{[\s\S]+\}", text)
            if brace_match:
                try:
                    data = json.loads(brace_match.group(0))
                except json.JSONDecodeError:
                    logger.warning("Failed to parse judge response as JSON: %s", raw[:200])
                    return JudgeResult(
                        score=0.0,
                        reasoning="[parse error]",
                        whitelist_match=False,
                        blacklist_violation=False,
                        numerical_accuracy=None,
                        raw_response=raw,
                    )
            else:
                logger.warning("No JSON object found in judge response: %s", raw[:200])
                return JudgeResult(
                    score=0.0,
                    reasoning="[parse error]",
                    whitelist_match=False,
                    blacklist_violation=False,
                    numerical_accuracy=None,
                    raw_response=raw,
                )

        num_acc = data.get("numerical_accuracy")
        if num_acc == "null" or num_acc == "":
            num_acc = None

        return JudgeResult(
            score=float(data.get("score", 0.0)),
            reasoning=str(data.get("reasoning", "")),
            whitelist_match=bool(data.get("whitelist_match", False)),
            blacklist_violation=bool(data.get("blacklist_violation", False)),
            numerical_accuracy=num_acc,
            raw_response=raw,
        )

    # ------------------------------------------------------------------
    # Internal: build user prompt
    # ------------------------------------------------------------------

    @staticmethod
    def _build_user_message(
        prediction: str,
        gold: str,
        question: str = "",
        whitelist: Optional[list[str]] = None,
        blacklist: Optional[list[str]] = None,
        references: str = "",
    ) -> str:
        """Render the user-turn prompt from a template."""
        wl_str = ", ".join(whitelist) if whitelist else "(none)"
        bl_str = ", ".join(blacklist) if blacklist else "(none)"
        return _USER_TEMPLATE.format(
            question=question or "(not provided)",
            gold=gold,
            prediction=prediction,
            whitelist=wl_str,
            blacklist=bl_str,
            references=references or "(none)",
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate_single(
        self,
        prediction: str,
        gold: str,
        question: str = "",
        whitelist: Optional[list[str]] = None,
        blacklist: Optional[list[str]] = None,
        references: str = "",
    ) -> JudgeResult:
        """Evaluate a single prediction against a gold answer.

        Parameters
        ----------
        prediction : str
            The model's predicted answer.
        gold : str
            The ground-truth reference answer.
        question : str, optional
            The original question or task description.
        whitelist : list[str], optional
            Terms that must appear in the prediction.
        blacklist : list[str], optional
            Terms whose presence immediately invalidates the answer.
        references : str, optional
            Additional context or reference material for the judge.

        Returns
        -------
        JudgeResult
            Structured evaluation result.
        """
        user_msg = self._build_user_message(
            prediction=prediction,
            gold=gold,
            question=question,
            whitelist=whitelist,
            blacklist=blacklist,
            references=references,
        )
        raw = self._call_judge(user_msg)
        return self._parse_response(raw)

    def evaluate_with_gt(
        self,
        predictions: dict[str, str],
        gold: dict[str, str],
        references: Optional[dict[str, Any]] = None,
    ) -> dict[str, JudgeResult]:
        """Evaluate a collection of predictions against gold answers.

        Parameters
        ----------
        predictions : dict[str, str]
            Mapping from question/sample ID to the model's predicted answer.
        gold : dict[str, str]
            Mapping from question/sample ID to the gold-standard answer.
            Must share the same key set as *predictions*.
        references : dict[str, Any], optional
            Optional per-sample reference dictionaries.  Each value may
            contain any of:

            - ``"question"`` (str): original question text
            - ``"whitelist"`` (list[str]): required terms
            - ``"blacklist"`` (list[str]): forbidden terms
            - ``"context"`` (str): extra reference text for the judge

        Returns
        -------
        dict[str, JudgeResult]
            Mapping from sample ID to its :class:`JudgeResult`.
        """
        if references is None:
            references = {}

        missing = set(predictions) - set(gold)
        if missing:
            raise ValueError(
                f"Gold answers missing for {len(missing)} sample(s): "
                + ", ".join(sorted(missing)[:5])
            )

        results: dict[str, JudgeResult] = {}
        for sample_id, pred in predictions.items():
            ref = references.get(sample_id, {})
            result = self.evaluate_single(
                prediction=pred,
                gold=gold[sample_id],
                question=ref.get("question", ""),
                whitelist=ref.get("whitelist"),
                blacklist=ref.get("blacklist"),
                references=ref.get("context", ""),
            )
            results[sample_id] = result
            logger.debug("sample=%s score=%.3f", sample_id, result.score)

        return results

    def summary_stats(self, results: dict[str, JudgeResult]) -> dict[str, float]:
        """Compute aggregate statistics over a collection of results.

        Parameters
        ----------
        results : dict[str, JudgeResult]
            Output of :meth:`evaluate_with_gt`.

        Returns
        -------
        dict[str, float]
            Keys: ``mean_score``, ``median_score``, ``pct_exact``
            (fraction with score ≥ 0.99), ``pct_blacklist_violation``.
        """
        import statistics

        scores = [r.score for r in results.values()]
        if not scores:
            return {}
        return {
            "mean_score": statistics.mean(scores),
            "median_score": statistics.median(scores),
            "pct_exact": sum(1 for s in scores if s >= 0.99) / len(scores),
            "pct_blacklist_violation": sum(
                1 for r in results.values() if r.blacklist_violation
            ) / len(scores),
        }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _main() -> None:  # pragma: no cover
    """CLI entry point.

    Usage::

        python -m dftoolbench.evaluation.llm_judge predictions.json \\
            [--judge claude|gemini] [--model <name>] [--out results.json]

    The predictions JSON must follow the schema::

        {
          "<sample_id>": {
            "prediction": "...",
            "gold": "...",
            "question": "...",          // optional
            "whitelist": ["term", ...], // optional
            "blacklist": ["term", ...], // optional
            "context": "..."            // optional
          },
          ...
        }
    """
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="DFToolBench-I LLM-as-a-Judge evaluator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("pred_json", help="Path to predictions JSON file.")
    parser.add_argument(
        "--judge",
        choices=["claude", "gemini"],
        default=None,
        help="Judge backend (overrides DFTOOLBENCH_JUDGE_BACKEND).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Judge model name (overrides DFTOOLBENCH_JUDGE_MODEL).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output JSON path.  Prints to stdout if omitted.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable debug logging."
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    with open(args.pred_json, encoding="utf-8") as fh:
        data: dict[str, dict[str, Any]] = json.load(fh)

    predictions: dict[str, str] = {}
    gold: dict[str, str] = {}
    references: dict[str, Any] = {}

    for sid, entry in data.items():
        if "prediction" not in entry or "gold" not in entry:
            logger.warning("Skipping sample %s: missing 'prediction' or 'gold'.", sid)
            continue
        predictions[sid] = entry["prediction"]
        gold[sid] = entry["gold"]
        references[sid] = {
            k: entry[k]
            for k in ("question", "whitelist", "blacklist", "context")
            if k in entry
        }

    cfg = JudgeConfig()
    if args.judge:
        cfg.backend = args.judge
    if args.model:
        cfg.model = args.model
    cfg.__post_init__()  # re-resolve model default if needed

    judge = LLMJudge(config=cfg)
    results = judge.evaluate_with_gt(predictions, gold, references)
    stats = judge.summary_stats(results)

    output = {
        "summary": stats,
        "results": {sid: r.to_dict() for sid, r in results.items()},
    }

    json_str = json.dumps(output, indent=2, ensure_ascii=False)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(json_str)
        print(f"Results written to {args.out}", file=sys.stderr)
    else:
        print(json_str)


if __name__ == "__main__":
    _main()
