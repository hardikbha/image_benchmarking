"""
dataset_loader.py — Load and pre-process DFToolBench-I benchmark queries.

Dataset format (dataset.json)
------------------------------
A top-level JSON object whose keys are query IDs and whose values are dicts
with the following schema::

    {
      "<query_id>": {
        "tools":      [ <tool_schema>, ... ],
        "files":      [ {"type": "image"|"video", "path": "<rel_path>"}, ... ],
        "dialogs":    [ {"role": "user"|"assistant", "content": ..., ...}, ... ],
        "gt_answer":  {"whitelist": [...], "blacklist": [...]},
        "evaluation": { ... },
        "metadata":   { ... }
      },
      ...
    }

Public API
----------
load_dataset(path, dataset_file='dataset.json')
    Returns a list of query dicts, each augmented with a ``"query_id"`` key.

organize_dialogs(entry, base_path)
    Resolves relative ``files[*].path`` values against *base_path* in-place
    and returns the modified entry.

get_tool_category(tool_name)
    Returns the benchmark category string for the given tool name:
    ``'P&R'``, ``'ML&A'``, or ``'QR&A'``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Tool → category mapping (Table II of the paper)
# ---------------------------------------------------------------------------

#: Tools that perform Perception & Recognition tasks.
_PR_TOOLS: frozenset[str] = frozenset(
    [
        "OCR",
        "FaceDetectionTool",
        "ObjectDetectionTool",
        "DeepfakeDetectionTool",
        "AnomalyDetectionTool",
        "FingerprintingTool",
    ]
)

#: Tools that perform Manipulation Localisation & Analysis tasks.
_MLA_TOOLS: frozenset[str] = frozenset(
    [
        "DenoiseTool",
        "SegmentationTool",
        "TextForgeryLocalizerTool",
        "CopyMoveLocalizationTool",
        "SceneChangeDetectionTool",
    ]
)

#: Tools that perform Quantitative Reasoning & Aggregation tasks.
_QRA_TOOLS: frozenset[str] = frozenset(["Calculator"])

_CATEGORY_MAP: Dict[str, str] = (
    {t: "P&R" for t in _PR_TOOLS}
    | {t: "ML&A" for t in _MLA_TOOLS}
    | {t: "QR&A" for t in _QRA_TOOLS}
)


def get_tool_category(tool_name: str) -> str:
    """Return the benchmark category for *tool_name*.

    Parameters
    ----------
    tool_name : str
        Canonical tool name as used in the dataset (e.g.
        ``"DeepfakeDetectionTool"``).

    Returns
    -------
    str
        One of ``'P&R'`` (Perception & Recognition),
        ``'ML&A'`` (Manipulation Localisation & Analysis), or
        ``'QR&A'`` (Quantitative Reasoning & Aggregation).

    Raises
    ------
    KeyError
        If *tool_name* is not registered in any category.
    """
    try:
        return _CATEGORY_MAP[tool_name]
    except KeyError as exc:
        known = ", ".join(sorted(_CATEGORY_MAP))
        raise KeyError(
            f"Unknown tool name {tool_name!r}.  Known tools: {known}"
        ) from exc


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def organize_dialogs(entry: Dict[str, Any], base_path: str) -> Dict[str, Any]:
    """Resolve relative file paths inside *entry* against *base_path*.

    The function modifies ``entry["files"][*]["path"]`` in-place so that
    every path becomes an absolute POSIX path.  All other fields are left
    unchanged.

    Parameters
    ----------
    entry : dict
        A single query dict as returned by :func:`load_dataset`.
    base_path : str
        The directory relative to which ``files[*].path`` values are
        interpreted (typically the directory containing ``dataset.json``).

    Returns
    -------
    dict
        The same *entry* dict with resolved paths (modified in-place and
        returned for convenience).
    """
    base = Path(base_path).resolve()
    for file_entry in entry.get("files", []):
        raw_path = file_entry.get("path", "")
        if raw_path:
            resolved = (base / raw_path).resolve()
            file_entry["path"] = str(resolved)
    return entry


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def load_dataset(
    path: str,
    dataset_file: str = "dataset.json",
) -> List[Dict[str, Any]]:
    """Load all benchmark queries from a dataset directory.

    Parameters
    ----------
    path : str
        Path to the dataset *directory* (must contain *dataset_file*).
    dataset_file : str, optional
        Name of the JSON file inside *path* (default: ``"dataset.json"``).

    Returns
    -------
    list[dict]
        Ordered list of query dicts.  Each dict contains all original
        fields from the JSON (``tools``, ``files``, ``dialogs``,
        ``gt_answer``, ``evaluation``, ``metadata``) plus an injected
        ``"query_id"`` key with the string key from the top-level JSON
        object.

        File paths inside every ``"files"`` list are resolved to absolute
        paths relative to *path* via :func:`organize_dialogs`.

    Raises
    ------
    FileNotFoundError
        If *path* or the dataset JSON file does not exist.
    ValueError
        If the JSON file does not contain a top-level object (dict).

    Examples
    --------
    >>> queries = load_dataset("/data/dftoolbench")
    >>> queries[0]["query_id"]
    'q001'
    >>> queries[0]["tools"][0]["name"]
    'DeepfakeDetectionTool'
    """
    dataset_dir = Path(path).resolve()
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    json_path = dataset_dir / dataset_file
    if not json_path.is_file():
        raise FileNotFoundError(
            f"Dataset file not found: {json_path}.  "
            f"Expected a JSON file named '{dataset_file}' inside '{dataset_dir}'."
        )

    with json_path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)

    if not isinstance(raw, dict):
        raise ValueError(
            f"Expected a top-level JSON object in {json_path}, "
            f"got {type(raw).__name__}."
        )

    queries: List[Dict[str, Any]] = []
    base_path = str(dataset_dir)

    for query_id, entry in raw.items():
        # Inject the string key so downstream code never loses track of it.
        entry = dict(entry)          # shallow copy – don't mutate the raw dict
        entry["query_id"] = query_id

        # Resolve file paths.
        entry = organize_dialogs(entry, base_path)

        queries.append(entry)

    return queries
