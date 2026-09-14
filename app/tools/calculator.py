"""Calculator tool.

Deliberately simple, so you can check the answer by hand and see whether the
model really called the tool or just did mental arithmetic (LLMs are bad at it).

Security note: we do NOT use Python's built-in eval(). The expression string
comes from the model, and eval() on model output would let it run arbitrary
code - e.g. `__import__("os").system("...")`. Instead we parse the string into
a syntax tree and walk it, allowing only arithmetic nodes. This is the standard
safe pattern, and it applies to every tool you ever give an agent.
"""

from __future__ import annotations

import ast
import operator
from typing import Callable

# The only operations the calculator will perform. Anything else is rejected.
_BINARY_OPS: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}

_UNARY_OPS: dict[type[ast.unaryop], Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


class CalculatorError(ValueError):
    """Raised when an expression is malformed or uses a forbidden operation."""


def _evaluate(node: ast.AST) -> float:
    """Recursively evaluate one node of the parsed expression tree."""
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)

    # A plain number, e.g. 42 or 3.14
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise CalculatorError(f"Not a number: {node.value!r}")
        return float(node.value)

    # Two operands, e.g. 3 * 4
    if isinstance(node, ast.BinOp):
        op = _BINARY_OPS.get(type(node.op))
        if op is None:
            raise CalculatorError(f"Operator not allowed: {type(node.op).__name__}")
        return op(_evaluate(node.left), _evaluate(node.right))

    # One operand, e.g. -5
    if isinstance(node, ast.UnaryOp):
        unary_op = _UNARY_OPS.get(type(node.op))
        if unary_op is None:
            raise CalculatorError(f"Operator not allowed: {type(node.op).__name__}")
        return unary_op(_evaluate(node.operand))

    # Function calls, names, attribute access, etc. all land here and are refused.
    raise CalculatorError(f"Expression element not allowed: {type(node).__name__}")


def calculate(expression: str) -> float:
    """Evaluate an arithmetic expression and return the exact result.

    Args:
        expression: Arithmetic only, e.g. "12500 * 0.23" or "(90 - 12) / 3".

    Returns:
        The numeric result.

    Raises:
        CalculatorError: If the expression is malformed or uses anything
            other than + - * / ** % and numbers.
    """
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as error:
        raise CalculatorError(f"Could not parse expression: {expression!r}") from error

    try:
        return _evaluate(tree)
    except ZeroDivisionError as error:
        raise CalculatorError("Division by zero") from error


# The description the model reads when deciding whether to call this tool.
# Plain JSON Schema - no vendor types - so the same dict works for Gemini,
# Claude and OpenAI with only minor reshaping.
#
# The `description` fields are not documentation: they are the prompt the model
# uses to choose a tool. A vague description means a tool that never gets called.
CALCULATOR_SCHEMA: dict = {
    "name": "calculate",
    "description": (
        "Evaluate an arithmetic expression and return the exact numeric result. "
        "Always use this for arithmetic instead of calculating mentally, "
        "including percentages, growth rates and unit conversions."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": (
                    "A pure arithmetic expression using only numbers and "
                    "the operators + - * / ** % and parentheses. "
                    "Example: '4200 * 1.075'"
                ),
            }
        },
        "required": ["expression"],
    },
}
