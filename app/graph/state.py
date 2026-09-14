"""The shared state that flows through the workflow.

In LangGraph, state is the single object every node reads from and writes to.
A node receives the current state and returns a dict of ONLY the keys it wants
to change; LangGraph merges that into the state and passes it to the next node.

This is the big shift from Phase 3. There, context lived in a local variable
inside one function. Here it is an explicit, inspectable, serialisable object -
which is what makes it possible to log runs, resume them, or branch on them.

total=False means every key is optional, so nodes can use state.get(...) for
values that earlier nodes may not have set.
"""

from __future__ import annotations

from typing import TypedDict


class ResearchState(TypedDict, total=False):
    """Everything the research workflow knows at a point in time."""

    question: str          # what the user asked (set at the start)
    plan: list[str]        # steps produced by the planner
    findings: list[str]    # one entry per completed research step
    analysis: str          # the analyst's interpretation of the findings
    final_report: str      # the writer's finished output
