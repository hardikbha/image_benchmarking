"""
calculator.py — Safe arithmetic expression evaluator.

No external model, no subprocess.  Expressions are evaluated with a
two-layer safety approach:

1. **AST pre-validation** — the expression is parsed into an AST and
   inspected for dangerous node types (``Import``, ``ImportFrom``,
   ``Attribute``, ``Call`` to non-whitelisted names, etc.) before any
   code runs.
2. **Restricted eval** — ``eval()`` is called with a tightly scoped
   globals dict that exposes only numeric constants and ``math`` module
   functions; ``__builtins__`` is set to an empty dict.
3. **Threading timeout guard** — evaluation runs in a daemon thread;
   expressions that exceed *timeout* seconds raise ``TimeoutError``.

Security model
--------------
* Import statements and attribute access (``os.system``, etc.) are
  rejected at AST-validation time before ``eval`` is called.
* Only names present in ``_SAFE_GLOBALS`` may be referenced at runtime.
* ``__builtins__`` is explicitly set to an empty dict to further restrict
  the runtime environment.
"""

from __future__ import annotations

import ast
import math
import threading
from typing import Any

from .base import BaseTool

# ------------------------------------------------------------------
# AST node whitelist
# ------------------------------------------------------------------
_SAFE_AST_NODES = (
    ast.Module,
    ast.Expr,
    ast.Expression,
    # Literals
    ast.Constant,
    ast.Num,       # Python < 3.8 compat alias
    ast.Str,       # Python < 3.8 compat alias (string literals rejected below)
    # Operators
    ast.BinOp,
    ast.UnaryOp,
    ast.BoolOp,
    ast.Compare,
    ast.IfExp,
    # Operator symbols
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv,
    ast.Mod, ast.Pow, ast.UAdd, ast.USub,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    ast.And, ast.Or, ast.Not,
    ast.BitAnd, ast.BitOr, ast.BitXor, ast.Invert,
    ast.LShift, ast.RShift,
    # Names (validated against whitelist at runtime)
    ast.Name,
    ast.Load,
    # Function calls (only whitelisted names allowed)
    ast.Call,
    # Tuple / list for multi-arg math functions like atan2, hypot
    ast.Tuple,
    ast.List,
)

_FORBIDDEN_AST_NODES = (
    ast.Import,
    ast.ImportFrom,
    ast.Attribute,    # blocks obj.attr access
    ast.Subscript,    # blocks indexing
    ast.Assign,
    ast.AugAssign,
    ast.Delete,
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.Lambda,
    ast.GeneratorExp,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.Yield,
    ast.YieldFrom,
    ast.Await,
    ast.Global,
    ast.Nonlocal,
)


def _validate_ast(tree: ast.AST, safe_names: set[str]) -> None:
    """Walk *tree* and raise ``ValueError`` on any forbidden construct.

    Parameters
    ----------
    tree : ast.AST
        Parsed AST of the expression.
    safe_names : set[str]
        Set of whitelisted name strings.

    Raises
    ------
    ValueError
        On any forbidden node type or unrecognised name reference.
    """
    for node in ast.walk(tree):
        if isinstance(node, _FORBIDDEN_AST_NODES):
            raise ValueError(
                f"Forbidden construct in expression: {type(node).__name__}"
            )
        if isinstance(node, ast.Name) and node.id not in safe_names:
            raise ValueError(
                f"Name {node.id!r} is not allowed in expressions."
            )
        # Reject string constants (only numeric constants make sense here)
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            raise ValueError("String literals are not allowed in expressions.")

# ------------------------------------------------------------------
# Whitelist of safe names available inside evaluated expressions
# ------------------------------------------------------------------
_SAFE_GLOBALS: dict[str, Any] = {
    # Setting __builtins__ to an empty dict restricts the runtime environment
    # in CPython; this is a defence-in-depth measure that complements the
    # primary AST pre-validation layer.
    "__builtins__": {},
    # math constants
    "pi": math.pi,
    "e": math.e,
    "inf": math.inf,
    "nan": math.nan,
    "tau": math.tau,
    # math functions
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "pow": pow,
    "sqrt": math.sqrt,
    "cbrt": math.cbrt if hasattr(math, "cbrt") else (lambda x: x ** (1 / 3)),
    "log": math.log,
    "log2": math.log2,
    "log10": math.log10,
    "exp": math.exp,
    "floor": math.floor,
    "ceil": math.ceil,
    "trunc": math.trunc,
    "factorial": math.factorial,
    "gcd": math.gcd,
    "lcm": math.lcm if hasattr(math, "lcm") else None,
    "comb": math.comb,
    "perm": math.perm,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": math.atan2,
    "degrees": math.degrees,
    "radians": math.radians,
    "sinh": math.sinh,
    "cosh": math.cosh,
    "tanh": math.tanh,
    "hypot": math.hypot,
    "isnan": math.isnan,
    "isinf": math.isinf,
    "isfinite": math.isfinite,
    "fabs": math.fabs,
    "fmod": math.fmod,
    "modf": math.modf,
    "divmod": divmod,
    "sum": sum,
}
# Remove None values (e.g. lcm on Python < 3.9)
_SAFE_GLOBALS = {k: v for k, v in _SAFE_GLOBALS.items() if v is not None}


class Calculator(BaseTool):
    """Evaluate a safe arithmetic expression and return its result.

    The evaluator exposes the full ``math`` module vocabulary (constants
    and functions) plus basic Python arithmetic operators.  It does **not**
    support string operations, list/dict literals, imports, attribute
    access, or any other construct that could pose a security risk.

    Parameters
    ----------
    timeout : int
        Maximum evaluation time in seconds (default: 5).
        Expressions that exceed this limit raise ``TimeoutError``.
    """

    default_desc = (
        "Evaluate a mathematical expression and return the numeric result. "
        "Supports standard arithmetic operators and all functions from Python's "
        "math module (sin, cos, log, sqrt, factorial, …). "
        "Input: expression string.  Output: result string."
    )

    def __init__(self, timeout: int = 5, device: str = "cpu") -> None:
        super().__init__(device=device)
        self.timeout = timeout

    def apply(self, expression: str) -> str:
        """Evaluate *expression* and return its result as a string.

        Parameters
        ----------
        expression : str
            A mathematical expression, e.g. ``"sqrt(2) * pi"`` or
            ``"factorial(10) / 2**8"``.

        Returns
        -------
        str
            String representation of the numeric result.

        Raises
        ------
        ValueError
            If the expression is empty or contains a syntax / name error.
        TimeoutError
            If evaluation exceeds *self.timeout* seconds.
        ArithmeticError
            For domain errors (e.g. ``log(-1)``).
        """
        expression = expression.strip()
        if not expression:
            raise ValueError("Expression must not be empty.")

        # --- Layer 1: AST pre-validation ---
        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError as exc:
            raise ValueError(f"Syntax error in expression: {exc}") from exc

        _safe_names = {k for k in _SAFE_GLOBALS if k != "__builtins__"}
        _validate_ast(tree, _safe_names)

        # --- Layer 2: Restricted eval with threading timeout ---
        result_box: list[Any] = [None]
        error_box: list[BaseException | None] = [None]

        def _eval() -> None:
            try:
                result_box[0] = eval(  # noqa: S307 — intentional restricted eval
                    compile(tree, "<expression>", "eval"),
                    _SAFE_GLOBALS,
                    {},
                )
            except Exception as exc:  # noqa: BLE001
                error_box[0] = exc

        thread = threading.Thread(target=_eval, daemon=True)
        thread.start()
        thread.join(timeout=self.timeout)

        if thread.is_alive():
            raise TimeoutError(
                f"Expression evaluation timed out after {self.timeout} s."
            )
        if error_box[0] is not None:
            exc = error_box[0]
            if isinstance(exc, (NameError, SyntaxError)):
                raise ValueError(f"Invalid expression: {exc}") from exc
            raise exc

        return str(result_box[0])
