"""
dftoolbench.data
================

Dataset loading, query construction, and annotation helpers for
DFToolBench-I.

Sub-modules
-----------
dataset_loader
    Load benchmark queries from JSON, resolve file paths, and map
    tool names to their benchmark category.
query_construction
    Template-based query generation, structured annotation format,
    minimal-sufficient-set computation, and difficulty classification.
"""

from dftoolbench.data.dataset_loader import (
    get_tool_category,
    load_dataset,
    organize_dialogs,
)
from dftoolbench.data.query_construction import (
    QueryAnnotation,
    classify_difficulty,
    compute_tools_gold,
    generate_query_from_template,
)

__all__ = [
    # dataset_loader
    "load_dataset",
    "organize_dialogs",
    "get_tool_category",
    # query_construction
    "QueryAnnotation",
    "generate_query_from_template",
    "compute_tools_gold",
    "classify_difficulty",
]
