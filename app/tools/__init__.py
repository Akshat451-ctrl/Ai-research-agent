# app/tools/__init__.py
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
from app.tools.document_search import DOCUMENT_SEARCH_SCHEMA, search_documents
from app.tools.web_search import WEB_SEARCH_SCHEMA, search_web

TOOL_SCHEMAS: list[dict] = [
    CALCULATOR_SCHEMA,
    DOCUMENT_SEARCH_SCHEMA,
    WEB_SEARCH_SCHEMA,
]

TOOL_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "calculate": calculate,
    "search_documents": search_documents,
    "search_web": search_web,
}

__all__ = ["TOOL_SCHEMAS", "TOOL_FUNCTIONS"]
