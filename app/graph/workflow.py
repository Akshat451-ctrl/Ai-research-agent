# app/graph/workflow.py
"""The LangGraph research workflow.

    START -> planner -.-> research -> analyse -> write -> fact_check -.-> write (revise)
                       '-> analyse -------------------------^          '-> END

Each node is an ordinary Python function: state in, partial state out.
The graph decides what runs next. Dotted lines are CONDITIONAL edges.

Phase 7 adds the fact_check node and a revision cycle: after writing, every
claim is checked against the raw evidence. If problems are found and the
revision budget is not spent, the writer gets another attempt with the
problems and raw evidence in hand - not just the analyst's summary.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from app.agents import researcher
from app.agents.fact_checker import (
    ClaimCheck,
    check_report,
    format_evidence,
    format_problems,
    print_checks,
)
from app.graph.state import ResearchState
from app.llm import ask

MAX_PLAN_STEPS = 3
MAX_REVISIONS = 2

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

REVISE_SYSTEM = (
    "You are a report writer revising a report after fact-checking found "
    "problems. Fix ONLY the specific problems listed - do not rewrite parts "
    "that were not flagged. Use the raw evidence to get facts right, and "
    "keep the same structure (heading, bullets, one-line conclusion)."
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
    """Run each planned step through the tool-using agent.

    Each step gets every earlier step's findings as context. Evidence and
    calculations from every step are kept raw, for the fact-checker.
    """
    print("\n[researcher] working through the plan...")
    findings: list[str] = []
    evidence: list[dict] = []
    calculations: list[dict] = []

    for step in state.get("plan", []):
        print(f"\n  > {step}")
        context = "\n\n".join(findings) if findings else None
        result = researcher.run(step, context=context)
        findings.append(f"{step}\n{result.answer}")
        evidence.extend(result.evidence)
        calculations.extend(result.calculations)

    return {"findings": findings, "evidence": evidence, "calculations": calculations}


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
    """Write the first draft, or revise using the fact-checker's problems."""
    fact_check_dicts = state.get("fact_checks") or []

    if fact_check_dicts:
        print("\n[writer] revising the report...")
        checks = [ClaimCheck(**item) for item in fact_check_dicts]
        problems = format_problems(checks)
        evidence_block = format_evidence(state.get("evidence") or [])
        report = ask(
            f"Question: {state['question']}\n\n"
            f"Previous report:\n{state.get('final_report', '')}\n\n"
            f"Problems found:\n{problems}\n\n"
            f"Raw evidence:\n{evidence_block}",
            system=REVISE_SYSTEM,
        )
        return {"final_report": report, "revisions": state.get("revisions", 0) + 1}

    print("\n[writer] composing the report...")
    report = ask(
        f"Question: {state['question']}\n\nAnalysis:\n{state.get('analysis', '')}",
        system=REPORT_SYSTEM,
    )
    return {"final_report": report}


def fact_check_node(state: ResearchState) -> ResearchState:
    """Verify every claim in the report against the raw evidence."""
    evidence = state.get("evidence") or []
    if not evidence:
        print("\n[fact-check] no evidence gathered - skipping")
        return {"fact_checks": []}

    print("\n[fact-check] verifying claims against raw evidence...")
    checks = check_report(
        state.get("final_report", ""), evidence, state.get("calculations") or []
    )
    print_checks(checks)
    return {"fact_checks": [check.model_dump() for check in checks]}


# --------------------------------------------------------------------------
# Routing. A conditional edge is just a function returning the next node name.
# --------------------------------------------------------------------------


def route_after_plan(state: ResearchState) -> str:
    """Skip the research stage entirely when the planner found nothing to do."""
    return "research" if state.get("plan") else "analyse"


def route_after_fact_check(state: ResearchState) -> str:
    """Send the report back for one more revision, or stop."""
    checks = state.get("fact_checks") or []
    has_problems = any(check["verdict"] != "supported" for check in checks)
    if has_problems and state.get("revisions", 0) < MAX_REVISIONS:
        return "write"
    return "end"


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
    graph.add_node("fact_check", fact_check_node)

    graph.add_edge(START, "planner")
    graph.add_conditional_edges(
        "planner",
        route_after_plan,
        {"research": "research", "analyse": "analyse"},
    )
    graph.add_edge("research", "analyse")
    graph.add_edge("analyse", "write")
    graph.add_edge("write", "fact_check")
    graph.add_conditional_edges(
        "fact_check",
        route_after_fact_check,
        {"write": "write", "end": END},
    )

    return graph.compile()


def run_research(question: str) -> ResearchState:
    """Run the whole workflow and return the final state."""
    return build_workflow().invoke({"question": question})
