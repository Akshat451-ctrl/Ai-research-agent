"""Tool registry.

Every tool the agent can use is registered here, in two parallel pieces:

  TOOL_SCHEMAS   - what the MODEL sees (names, descriptions, parameters)
  TOOL_FUNCTIONS - what YOUR CODE runs when the model asks for a tool

The agent loop looks up the requested name in TOOL_FUNCTIONS. Adding a new
tool means writing the module and adding one line to each collection below.
"""

from __future__ import annotations

from typing import Any, Callable

from app.tools.calculator import CALCULATOR_SCHEMA, calculate

TOOL_SCHEMAS: list[dict] = [
    CALCULATOR_SCHEMA,
]

TOOL_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "calculate": calculate,
}

__all__ = ["TOOL_SCHEMAS", "TOOL_FUNCTIONS"]
