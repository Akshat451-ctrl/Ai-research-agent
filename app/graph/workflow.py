"""The LangGraph research workflow.

    START -> planner -.-> research -> analyse -> write -> END
                       '-> analyse ------------^

Each node is an ordinary Python function: state in, partial state out.
The graph decides what runs next. The dotted line is a CONDITIONAL edge -
route_after_plan() inspects the state and picks the next node at runtime.

Why this beats the hand-written loop from Phase 3:
  - each stage is separately testable, with no LLM involved in the wiring
  - the control flow is data you can draw, log and reason about
  - adding a fact-checker in Phase 7 is one node plus one edge
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from app.agents import researcher
from app.graph.state import ResearchState
from app.llm import ask

MAX_PLAN_STEPS = 3

PLANNER_SYSTEM = (
    "You are a research planner. Break the user's question into at most "
    f"{MAX_PLAN_STEPS} concrete steps that a research assistant will carry "
    "out one at a time. The assistant has a calculator tool, so any figure "
    "that must be computed should be its own step, stated with the actual "
    "numbers to use. Each step must stand alone - the assistant cannot see "
    "the original question or the other steps.\n"
    "Output ONE step per line, with no numbering, bullets or commentary. "
    "Only if the question is pure opinion with nothing to look up or "
    "compute, output the single word: NONE"
)

ANALYST_SYSTEM = (
    "You are a data analyst. Given research findings, identify what they "
    "actually show: the key numbers, the trend, and anything surprising. "
    "Be concise and do not invent figures."
)

REPORT_SYSTEM = (
    "You are a report writer. Turn the analysis into a short structured "
    "report with a heading, 3-5 bullet findings, and a one-line conclusion."
)


# --------------------------------------------------------------------------
# Nodes. Each takes the state and returns only the keys it changes.
# --------------------------------------------------------------------------


def plan_node(state: ResearchState) -> ResearchState:
    """Turn the question into a list of concrete research steps."""
    print("\n[planner] breaking the question into steps...")
    raw = ask(f"Question: {state['question']}", system=PLANNER_SYSTEM)

    if raw.strip().upper().startswith("NONE"):
        print("   (no research needed)")
        return {"plan": []}

    steps = [line.strip("-* \t") for line in raw.splitlines() if line.strip()]
    steps = steps[:MAX_PLAN_STEPS]
    for step in steps:
        print(f"   - {step}")
    return {"plan": steps}


def research_node(state: ResearchState) -> ResearchState:
    """Run each planned step through the Phase 3 tool-using agent."""
    print("\n[researcher] working through the plan...")
    findings: list[str] = []
    for step in state.get("plan", []):
        print(f"\n  > {step}")
        findings.append(f"{step}\n{researcher.run(step)}")
    return {"findings": findings}


def analyse_node(state: ResearchState) -> ResearchState:
    """Interpret the findings - what do the numbers actually mean?"""
    print("\n[analyst] interpreting findings...")
    findings = state.get("findings") or ["(no research was performed)"]
    analysis = ask(
        f"Question: {state['question']}\n\nFindings:\n" + "\n\n".join(findings),
        system=ANALYST_SYSTEM,
    )
    return {"analysis": analysis}


def report_node(state: ResearchState) -> ResearchState:
    """Write the final structured report."""
    print("\n[writer] composing the report...")
    report = ask(
        f"Question: {state['question']}\n\nAnalysis:\n{state.get('analysis', '')}",
        system=REPORT_SYSTEM,
    )
    return {"final_report": report}


# --------------------------------------------------------------------------
# Routing. A conditional edge is just a function returning the next node name.
# --------------------------------------------------------------------------


def route_after_plan(state: ResearchState) -> str:
    """Skip the research stage entirely when the planner found nothing to do."""
    return "research" if state.get("plan") else "analyse"


# --------------------------------------------------------------------------
# Graph assembly
# --------------------------------------------------------------------------


def build_workflow():
    """Wire the nodes together and compile the graph."""
    graph = StateGraph(ResearchState)

    graph.add_node("planner", plan_node)
    graph.add_node("research", research_node)
    graph.add_node("analyse", analyse_node)
    graph.add_node("write", report_node)

    graph.add_edge(START, "planner")
    graph.add_conditional_edges(
        "planner",
        route_after_plan,
        {"research": "research", "analyse": "analyse"},
    )
    graph.add_edge("research", "analyse")
    graph.add_edge("analyse", "write")
    graph.add_edge("write", END)

    return graph.compile()


def run_research(question: str) -> ResearchState:
    """Run the whole workflow and return the final state."""
    return build_workflow().invoke({"question": question})
