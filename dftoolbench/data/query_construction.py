"""
query_construction.py — Query construction and annotation pipeline helpers.

This module supports the *manual annotation + template-assisted generation*
pipeline described in Section III of the DFToolBench-I paper.

Key concepts
------------
tools_gold
    The *minimal sufficient set* of tools required to answer a query
    correctly.  Computed from the ground-truth dialog by collecting every
    tool that appears in an assistant turn's ``tool_calls`` field.

Difficulty
    ``"medium"`` — queries that require 1–2 distinct tool invocations.
    ``"hard"``   — queries that require 3 or more distinct tool invocations.

QueryAnnotation
    Structured container for a single annotated query (mirrors the
    benchmark JSON schema).

Public API
----------
generate_query_from_template(tool_meta, template, **kwargs)
    Fill a natural-language query template from tool metadata.

compute_tools_gold(dialogs)
    Derive the minimal sufficient tool set from a dialog's ground-truth
    assistant turns.

classify_difficulty(tools_gold)
    Return ``"medium"`` or ``"hard"`` based on the number of required tools.

QueryAnnotation
    Dataclass-style container for a fully annotated benchmark entry.
"""

from __future__ import annotations

import copy
import re
import string
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


# ---------------------------------------------------------------------------
# QueryAnnotation
# ---------------------------------------------------------------------------


@dataclass
class QueryAnnotation:
    """Structured annotation for a single benchmark query.

    Attributes
    ----------
    query_id : str
        Unique string identifier (e.g. ``"q001"``).
    query_text : str
        Natural-language question posed to the agent.
    tools : list[dict]
        Full tool schemas available to the agent for this query.
    files : list[dict]
        Input image/video references: ``[{"type": ..., "path": ...}, ...]``.
    dialogs : list[dict]
        Ground-truth multi-turn conversation with annotated tool calls.
    gt_answer : dict
        Gold answer in whitelist/blacklist format::

            {
              "whitelist": [<accepted_value>, ...],
              "blacklist": [<rejected_value>, ...]
            }
    tools_gold : list[str]
        Minimal sufficient set of tool names needed to answer the query.
    difficulty : str
        ``"medium"`` or ``"hard"`` (see :func:`classify_difficulty`).
    evaluation : dict
        Evaluation metadata (tolerances, type, etc.).
    metadata : dict
        Free-form dataset provenance info.
    """

    query_id: str
    query_text: str
    tools: List[Dict[str, Any]] = field(default_factory=list)
    files: List[Dict[str, Any]] = field(default_factory=list)
    dialogs: List[Dict[str, Any]] = field(default_factory=list)
    gt_answer: Dict[str, Any] = field(default_factory=dict)
    tools_gold: List[str] = field(default_factory=list)
    difficulty: str = "medium"
    evaluation: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the annotation back to the benchmark JSON schema.

        Returns
        -------
        dict
            Dict compatible with the top-level ``dataset.json`` entry format.
        """
        return {
            "tools": copy.deepcopy(self.tools),
            "files": copy.deepcopy(self.files),
            "dialogs": copy.deepcopy(self.dialogs),
            "gt_answer": copy.deepcopy(self.gt_answer),
            "evaluation": copy.deepcopy(self.evaluation),
            "metadata": {
                **copy.deepcopy(self.metadata),
                "tools_gold": self.tools_gold,
                "difficulty": self.difficulty,
            },
        }

    @classmethod
    def from_dict(cls, query_id: str, entry: Dict[str, Any]) -> "QueryAnnotation":
        """Construct a :class:`QueryAnnotation` from a raw dataset entry.

        Parameters
        ----------
        query_id : str
            The key under which this entry appears in ``dataset.json``.
        entry : dict
            A single dataset entry dict (the value, not the full file).

        Returns
        -------
        QueryAnnotation
        """
        metadata = entry.get("metadata", {})
        dialogs = entry.get("dialogs", [])

        # Extract query text: first user turn content.
        query_text = ""
        for turn in dialogs:
            if turn.get("role") == "user":
                content = turn.get("content", "")
                if isinstance(content, str):
                    query_text = content
                elif isinstance(content, list):
                    # Multi-modal content list: grab the first text block.
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            query_text = block.get("text", "")
                            break
                break

        tools_gold = metadata.get("tools_gold") or compute_tools_gold(dialogs)
        difficulty = metadata.get("difficulty") or classify_difficulty(tools_gold)

        return cls(
            query_id=query_id,
            query_text=query_text,
            tools=entry.get("tools", []),
            files=entry.get("files", []),
            dialogs=dialogs,
            gt_answer=entry.get("gt_answer", {}),
            tools_gold=tools_gold,
            difficulty=difficulty,
            evaluation=entry.get("evaluation", {}),
            metadata=metadata,
        )


# ---------------------------------------------------------------------------
# Template-based query generation
# ---------------------------------------------------------------------------

#: Regex that matches ``{placeholder}`` tokens in a template string.
_PLACEHOLDER_RE = re.compile(r"\{(\w+)\}")


def generate_query_from_template(
    tool_meta: Dict[str, Any],
    template: str,
    **kwargs: Any,
) -> str:
    """Fill a natural-language query template from tool metadata.

    Placeholders in *template* are resolved in the following order:

    1. Keyword arguments in *kwargs* (highest priority).
    2. Top-level keys of *tool_meta* (e.g. ``name``, ``description``).
    3. Flattened input parameter names from ``tool_meta["inputs"]``.

    Any placeholder that cannot be resolved is left as-is (i.e. the
    ``{placeholder}`` text is preserved verbatim).

    Parameters
    ----------
    tool_meta : dict
        Tool metadata dict (one entry from ``configs/tool_meta.json``).
        Expected keys: ``"name"``, ``"description"``, ``"inputs"``,
        ``"outputs"``.
    template : str
        Template string with ``{placeholder}`` tokens, e.g.::

            "Please run {name} on {image_path} and tell me the result."
    **kwargs
        Override values for any placeholder (e.g. ``image_path="/data/x.jpg"``).

    Returns
    -------
    str
        The query string with all resolvable placeholders substituted.

    Examples
    --------
    >>> meta = {"name": "OCR", "description": "Extract text.", "inputs": [], "outputs": []}
    >>> generate_query_from_template(meta, "Run {name}: {description}")
    'Run OCR: Extract text.'
    """
    # Build resolution map: kwargs > tool top-level keys > input param names.
    resolution: Dict[str, str] = {}

    # 2. Tool-level keys.
    for key, val in tool_meta.items():
        if isinstance(val, str):
            resolution[key] = val

    # 3. Input parameter names (use description as the value for context).
    for inp in tool_meta.get("inputs", []):
        if isinstance(inp, dict):
            param_name = inp.get("name", "")
            if param_name:
                resolution[param_name] = inp.get("desc", param_name)

    # 1. Explicit kwargs override everything.
    resolution.update({str(k): str(v) for k, v in kwargs.items()})

    def _replace(match: re.Match) -> str:  # type: ignore[type-arg]
        token = match.group(1)
        return resolution.get(token, match.group(0))  # leave unknown as-is

    return _PLACEHOLDER_RE.sub(_replace, template)


# ---------------------------------------------------------------------------
# Minimal sufficient tool set
# ---------------------------------------------------------------------------


def compute_tools_gold(dialogs: List[Dict[str, Any]]) -> List[str]:
    """Derive the minimal sufficient tool set from ground-truth dialogs.

    Scans every *assistant* turn in *dialogs* for ``tool_calls`` entries and
    collects the unique set of tool names invoked, preserving first-occurrence
    order.

    A tool call entry is expected in one of two forms:

    * OpenAI-style::

        {"type": "function", "function": {"name": "ToolName", ...}}

    * Simple dict with a top-level ``"name"`` key::

        {"name": "ToolName", ...}

    Parameters
    ----------
    dialogs : list[dict]
        Multi-turn conversation list (``"role"`` / ``"content"`` / optional
        ``"tool_calls"`` keys per turn).

    Returns
    -------
    list[str]
        Ordered list of unique tool names that appear in any assistant turn's
        ``tool_calls`` field.  Empty list if no tool calls are found.
    """
    seen: Set[str] = set()
    ordered: List[str] = []

    for turn in dialogs:
        if turn.get("role") != "assistant":
            continue
        for call in turn.get("tool_calls", []):
            # OpenAI-style function call.
            name: Optional[str] = None
            if isinstance(call, dict):
                func = call.get("function")
                if isinstance(func, dict):
                    name = func.get("name")
                else:
                    name = call.get("name")
            if name and name not in seen:
                seen.add(name)
                ordered.append(name)

    return ordered


# ---------------------------------------------------------------------------
# Difficulty classification
# ---------------------------------------------------------------------------

#: Number of required tool invocations at which difficulty becomes "hard".
_HARD_THRESHOLD: int = 3


def classify_difficulty(tools_gold: List[str]) -> str:
    """Classify query difficulty based on the minimal required tool count.

    Parameters
    ----------
    tools_gold : list[str]
        Minimal sufficient tool set (output of :func:`compute_tools_gold`).

    Returns
    -------
    str
        ``"hard"`` if ``len(tools_gold) >= 3``, otherwise ``"medium"``.

    Examples
    --------
    >>> classify_difficulty(["OCR"])
    'medium'
    >>> classify_difficulty(["DeepfakeDetectionTool", "FaceDetectionTool", "Calculator"])
    'hard'
    """
    return "hard" if len(tools_gold) >= _HARD_THRESHOLD else "medium"
